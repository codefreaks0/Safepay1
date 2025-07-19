import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib
from flask import Flask, request, jsonify
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced classifiers
try:
    from xgboost import XGBClassifier
    xgb_installed = True
except ImportError:
    xgb_installed = False
try:
    from lightgbm import LGBMClassifier
    lgbm_installed = True
except ImportError:
    lgbm_installed = False
try:
    from catboost import CatBoostClassifier
    catboost_installed = True
except ImportError:
    catboost_installed = False
try:
    from imblearn.over_sampling import SMOTE
    smote_installed = True
except ImportError:
    smote_installed = False

MODEL_PATH = 'models/upi_fraud_detection_model.pkl'
DATA_PATH = 'anonymized_sample_fraud_txn.csv'
LOG_PATH = 'models/upi_fraud_training_log.txt'

# Advanced Feature Engineering

def upi_id_features(upi_id):
    upi_id = str(upi_id)
    bank_code = upi_id.split('@')[-1] if '@' in upi_id else 'unknown'
    is_random = int(sum(c.isalpha() for c in upi_id) > 8 and sum(c.isdigit() for c in upi_id) > 3)
    suspicious_score = 0
    if len(upi_id) < 8 or len(upi_id) > 20:
        suspicious_score += 1
    if is_random:
        suspicious_score += 1
    rare_banks = {'fam', 'pty', 'naviaxis', 'freecharge', 'superyes', 'ptys', 'ptaxis'}
    if bank_code in rare_banks:
        suspicious_score += 1
    features = {
        'upi_id_length': len(upi_id),
        'has_number': int(any(char.isdigit() for char in upi_id)),
        'has_special': int(any(char in '@._-' for char in upi_id)),
        'bank_code': bank_code,
        'suspicious_score': suspicious_score
    }
    return features

def load_and_preprocess_data():
    data = pd.read_csv(DATA_PATH)
    data['TXN_TIMESTAMP'] = pd.to_datetime(data['TXN_TIMESTAMP'], errors='coerce')
    data['hour'] = data['TXN_TIMESTAMP'].dt.hour
    data['day_of_week'] = data['TXN_TIMESTAMP'].dt.dayofweek
    data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
    data['txn_month'] = data['TXN_TIMESTAMP'].dt.month
    data['txn_day'] = data['TXN_TIMESTAMP'].dt.day
    data['is_night'] = data['hour'].apply(lambda x: 1 if x < 6 or x > 22 else 0)
    data['is_working_hour'] = data['hour'].apply(lambda x: 1 if 9 <= x <= 18 else 0)
    payer_counts = data['PAYER_VPA'].value_counts().to_dict()
    beneficiary_counts = data['BENEFICIARY_VPA'].value_counts().to_dict()
    data['payer_freq'] = data['PAYER_VPA'].map(payer_counts)
    data['beneficiary_freq'] = data['BENEFICIARY_VPA'].map(beneficiary_counts)
    data['amount_category'] = pd.cut(
        data['AMOUNT'],
        bins=[0, 500, 2000, 5000, 10000, 20000, 50000, float('inf')],
        labels=['micro', 'small', 'medium', 'large', 'xlarge', 'xxlarge', 'xxxlarge']
    )
    failed_payer = data.groupby('PAYER_VPA')['TRN_STATUS'].apply(lambda x: (x=='FAILED').mean()).to_dict()
    data['payer_failed_ratio'] = data['PAYER_VPA'].map(failed_payer)
    data['payer_unique_beneficiaries'] = data.groupby('PAYER_VPA')['BENEFICIARY_VPA'].transform('nunique')
    data['beneficiary_unique_payers'] = data.groupby('BENEFICIARY_VPA')['PAYER_VPA'].transform('nunique')
    upi_features = data['PAYER_VPA'].apply(upi_id_features).apply(pd.Series)
    upi_features.columns = [f'payer_{col}' for col in upi_features.columns]
    data = pd.concat([data, upi_features], axis=1)
    upi_features_b = data['BENEFICIARY_VPA'].apply(upi_id_features).apply(pd.Series)
    upi_features_b.columns = [f'beneficiary_{col}' for col in upi_features_b.columns]
    data = pd.concat([data, upi_features_b], axis=1)
    for col in ['payer_bank_code', 'beneficiary_bank_code']:
        if col in data:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
    data = data.sort_values('TXN_TIMESTAMP')
    data['payer_recent_frauds'] = data.groupby('PAYER_VPA')['IS_FRAUD'].transform(lambda x: x.rolling(10, min_periods=1).sum())
    data['beneficiary_recent_frauds'] = data.groupby('BENEFICIARY_VPA')['IS_FRAUD'].transform(lambda x: x.rolling(10, min_periods=1).sum())
    # Interaction features
    data['amt_x_failed'] = data['AMOUNT'] * data['payer_failed_ratio']
    data['freq_x_failed'] = data['payer_freq'] * data['payer_failed_ratio']
    data['amt_x_suspicious'] = data['AMOUNT'] * data['payer_suspicious_score']
    # --- MAXIMUM FRAUD FOCUS: Heavily weighted fraud-related features ---
    # 1. Direct IS_FRAUD correlation features
    data['payer_fraud_ratio'] = data.groupby('PAYER_VPA')['IS_FRAUD'].transform('mean')
    data['beneficiary_fraud_ratio'] = data.groupby('BENEFICIARY_VPA')['IS_FRAUD'].transform('mean')
    data['payer_total_frauds'] = data.groupby('PAYER_VPA')['IS_FRAUD'].transform('sum')
    data['beneficiary_total_frauds'] = data.groupby('BENEFICIARY_VPA')['IS_FRAUD'].transform('sum')
    # 2. Recent fraud patterns (last 5, 10, 20 transactions)
    data['payer_recent_frauds_5'] = data.groupby('PAYER_VPA')['IS_FRAUD'].transform(lambda x: x.rolling(5, min_periods=1).sum())
    data['beneficiary_recent_frauds_5'] = data.groupby('BENEFICIARY_VPA')['IS_FRAUD'].transform(lambda x: x.rolling(5, min_periods=1).sum())
    data['payer_recent_frauds_20'] = data.groupby('PAYER_VPA')['IS_FRAUD'].transform(lambda x: x.rolling(20, min_periods=1).sum())
    data['beneficiary_recent_frauds_20'] = data.groupby('BENEFICIARY_VPA')['IS_FRAUD'].transform(lambda x: x.rolling(20, min_periods=1).sum())
    # 3. Fraud frequency features
    data['payer_fraud_frequency'] = data.groupby('PAYER_VPA')['IS_FRAUD'].transform(lambda x: x.rolling(50, min_periods=1).mean())
    data['beneficiary_fraud_frequency'] = data.groupby('BENEFICIARY_VPA')['IS_FRAUD'].transform(lambda x: x.rolling(50, min_periods=1).mean())
    # 4. Weighted features for model learning (MULTIPLY BY 50 for maximum focus)
    max_payers = data['beneficiary_unique_payers'].max()
    data['beneficiary_recent_frauds_weighted'] = data['beneficiary_recent_frauds'] * 50
    data['payer_recent_frauds_weighted'] = data['payer_recent_frauds'] * 50
    data['payer_fraud_ratio_weighted'] = data['payer_fraud_ratio'] * 100
    data['beneficiary_fraud_ratio_weighted'] = data['beneficiary_fraud_ratio'] * 100
    data['payer_total_frauds_weighted'] = data['payer_total_frauds'] * 50
    data['beneficiary_total_frauds_weighted'] = data['beneficiary_total_frauds'] * 50
    # Add total recent frauds and weighted version
    data['total_recent_frauds'] = data['beneficiary_recent_frauds'] + data['payer_recent_frauds']
    data['total_recent_frauds_weighted'] = data['total_recent_frauds'] * 50
    data['total_fraud_ratio'] = data['payer_fraud_ratio'] + data['beneficiary_fraud_ratio']
    data['total_fraud_ratio_weighted'] = data['total_fraud_ratio'] * 100
    # 5. Fraud pattern indicators
    data['is_high_fraud_payer'] = (data['payer_fraud_ratio'] > 0.3).astype(int)
    data['is_high_fraud_beneficiary'] = (data['beneficiary_fraud_ratio'] > 0.3).astype(int)
    data['has_recent_fraud'] = ((data['payer_recent_frauds'] > 0) | (data['beneficiary_recent_frauds'] > 0)).astype(int)
    # --- Isolation Forest anomaly score ---
    iso_features = [
        'AMOUNT', 'payer_freq', 'beneficiary_freq', 'payer_failed_ratio',
        'payer_unique_beneficiaries', 'beneficiary_unique_payers',
        'payer_recent_frauds', 'beneficiary_recent_frauds',
        'amt_x_failed', 'freq_x_failed', 'amt_x_suspicious'
    ]
    iso_data = data[iso_features].fillna(0)
    iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    data['anomaly_score'] = iso.fit_predict(iso_data)
    data['anomaly_score'] = (data['anomaly_score'] == -1).astype(int)  # 1 if anomaly, else 0
    features = [
        'AMOUNT', 'hour', 'day_of_week', 'is_weekend', 'txn_month', 'txn_day', 'is_night', 'is_working_hour',
        'payer_freq', 'beneficiary_freq', 'amount_category',
        'TRN_STATUS', 'RESPONSE_CODE', 'TRANSACTION_TYPE', 'PAYMENT_INSTRUMENT',
        'payer_failed_ratio', 'payer_unique_beneficiaries', 'beneficiary_unique_payers',
        'payer_upi_id_length', 'payer_has_number', 'payer_has_special',
        'beneficiary_upi_id_length', 'beneficiary_has_number', 'beneficiary_has_special',
        'payer_recent_frauds', 'beneficiary_recent_frauds',
        'payer_suspicious_score', 'beneficiary_suspicious_score',
        'amt_x_failed', 'freq_x_failed', 'amt_x_suspicious',
        # --- FRAUD-FOCUSED FEATURES (HIGH PRIORITY) ---
        'payer_fraud_ratio', 'beneficiary_fraud_ratio',
        'payer_total_frauds', 'beneficiary_total_frauds',
        'payer_recent_frauds_5', 'beneficiary_recent_frauds_5',
        'payer_recent_frauds_20', 'beneficiary_recent_frauds_20',
        'payer_fraud_frequency', 'beneficiary_fraud_frequency',
        'beneficiary_recent_frauds_weighted', 'payer_recent_frauds_weighted',
        'payer_fraud_ratio_weighted', 'beneficiary_fraud_ratio_weighted',
        'payer_total_frauds_weighted', 'beneficiary_total_frauds_weighted',
        'total_recent_frauds', 'total_recent_frauds_weighted',
        'total_fraud_ratio', 'total_fraud_ratio_weighted',
        'is_high_fraud_payer', 'is_high_fraud_beneficiary', 'has_recent_fraud',
        'beneficiary_unique_payers_weighted',
        'anomaly_score'
    ]
    if 'payer_bank_code' in data and 'beneficiary_bank_code' in data:
        features += ['payer_bank_code', 'beneficiary_bank_code']
    target = 'IS_FRAUD'
    return data, features, target

def train_and_save_model():
    print("\n--- Model Training Started ---\n")
    data, features, target = load_and_preprocess_data()
    amt_99 = data['AMOUNT'].quantile(0.99)
    data = data[data['AMOUNT'] <= amt_99]
    # --- Temporal split: train on oldest 80%, test on newest 20% ---
    data = data.sort_values('TXN_TIMESTAMP')
    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]
    # SMOTE for class balancing
    if smote_installed:
        sm = SMOTE(random_state=42)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    else:
        X_train_res, y_train_res = X_train, y_train
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    models = {}
    scores = {}
    logs = []
    # 1. XGBoost
    if xgb_installed:
        print("Training XGBoost...")
        param_dist = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.01, 0.05, 0.1],
            'classifier__scale_pos_weight': [1, 3, 5]
        }
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))
        ])
        grid = RandomizedSearchCV(pipe, param_dist, n_iter=10, cv=skf, scoring='f1', n_jobs=-1, random_state=42)
        grid.fit(X_train_res, y_train_res)
        models['xgb'] = grid.best_estimator_
        y_pred = grid.predict(X_test)
        y_proba = grid.predict_proba(X_test)[:, 1]
        scores['xgb'] = f1_score(y_test, y_pred)
        logs.append(('XGBoost', grid.best_params_, classification_report(y_test, y_pred), roc_auc_score(y_test, y_proba)))
    # 2. LightGBM
    if lgbm_installed:
        print("Training LightGBM...")
        param_dist = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.01, 0.05, 0.1]
        }
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LGBMClassifier(random_state=42))
        ])
        grid = RandomizedSearchCV(pipe, param_dist, n_iter=10, cv=skf, scoring='f1', n_jobs=-1, random_state=42)
        grid.fit(X_train_res, y_train_res)
        models['lgbm'] = grid.best_estimator_
        y_pred = grid.predict(X_test)
        y_proba = grid.predict_proba(X_test)[:, 1]
        scores['lgbm'] = f1_score(y_test, y_pred)
        logs.append(('LightGBM', grid.best_params_, classification_report(y_test, y_pred), roc_auc_score(y_test, y_proba)))
    # 3. CatBoost
    if catboost_installed:
        print("Training CatBoost...")
        param_dist = {
            'classifier__iterations': [100, 200, 300],
            'classifier__depth': [3, 5, 7],
            'classifier__learning_rate': [0.01, 0.05, 0.1]
        }
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', CatBoostClassifier(random_state=42, verbose=0))
        ])
        grid = RandomizedSearchCV(pipe, param_dist, n_iter=10, cv=skf, scoring='f1', n_jobs=-1, random_state=42)
        grid.fit(X_train_res, y_train_res)
        models['catboost'] = grid.best_estimator_
        y_pred = grid.predict(X_test)
        y_proba = grid.predict_proba(X_test)[:, 1]
        scores['catboost'] = f1_score(y_test, y_pred)
        logs.append(('CatBoost', grid.best_params_, classification_report(y_test, y_pred), roc_auc_score(y_test, y_proba)))
    # 4. RandomForest (always available)
    print("Training RandomForest...")
    param_dist = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [5, 10, 15]
    }
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])
    grid = RandomizedSearchCV(pipe, param_dist, n_iter=10, cv=skf, scoring='f1', n_jobs=-1, random_state=42)
    grid.fit(X_train_res, y_train_res)
    models['rf'] = grid.best_estimator_
    y_pred = grid.predict(X_test)
    y_proba = grid.predict_proba(X_test)[:, 1]
    scores['rf'] = f1_score(y_test, y_pred)
    logs.append(('RandomForest', grid.best_params_, classification_report(y_test, y_pred), roc_auc_score(y_test, y_proba)))
    # 5. Ensemble VotingClassifier (if all advanced models available)
    if xgb_installed and lgbm_installed and catboost_installed:
        print("Training Ensemble VotingClassifier...")
        ensemble = VotingClassifier(estimators=[
            ('xgb', models['xgb'].named_steps['classifier']),
            ('lgbm', models['lgbm'].named_steps['classifier']),
            ('catboost', models['catboost'].named_steps['classifier']),
            ('rf', models['rf'].named_steps['classifier'])
        ], voting='soft')
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', ensemble)
        ])
        pipe.fit(X_train_res, y_train_res)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        scores['ensemble'] = f1_score(y_test, y_pred)
        logs.append(('Ensemble', 'VotingClassifier', classification_report(y_test, y_pred), roc_auc_score(y_test, y_proba)))
        models['ensemble'] = pipe
    # Select best model
    best_model_key = max(scores, key=scores.get)
    model = models[best_model_key]
    print(f"\nBest Model: {best_model_key}")
    # Evaluate best model
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    rocauc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {rocauc:.4f}")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)
    # Feature importances
    try:
        importances = model.named_steps['classifier'].feature_importances_
        feature_names = model.named_steps['preprocessor'].get_feature_names_out()
        feat_imp = sorted(zip(feature_names, importances), key=lambda x: -x[1])
        print("Top 10 Feature Importances:")
        for f, imp in feat_imp[:10]:
            print(f, imp)
    except Exception as e:
        print("Feature importances not available:", e)
    # ROC curve for best threshold
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    best_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[best_idx]
    print(f"Best threshold by ROC: {best_threshold}")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    with open('models/upi_fraud_best_threshold.txt', 'w') as f:
        f.write(str(best_threshold))
    # Log everything
    with open(LOG_PATH, 'w') as f:
        f.write(f"Best Model: {best_model_key}\n")
        f.write(f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1 Score: {f1:.4f}\nROC-AUC: {rocauc:.4f}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")
        f.write(f"Classification Report:\n{report}\n")
        for name, params, rep, auc in logs:
            f.write(f"\n{name} Params: {params}\n{name} ROC-AUC: {auc:.4f}\n{name} Report:\n{rep}\n")
    print(f"\nAll training metrics and details saved to {LOG_PATH}\n")
    return model, data

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        model, _ = train_and_save_model()
        return model

def get_best_threshold():
    try:
        with open('models/upi_fraud_best_threshold.txt', 'r') as f:
            return float(f.read().strip())
    except:
        return 0.3

def predict_upi_fraud(upi_id, data, model):
    payer_txns = data[data['PAYER_VPA'] == upi_id]
    beneficiary_txns = data[data['BENEFICIARY_VPA'] == upi_id]
    if len(payer_txns) == 0 and len(beneficiary_txns) == 0:
        return {"upi_id": upi_id, "error": "No transactions found for this UPI ID"}
    all_txns = pd.concat([payer_txns, beneficiary_txns])
    features_dict = {
        'AMOUNT': all_txns['AMOUNT'].mean(),
        'hour': all_txns['hour'].mode()[0],
        'day_of_week': all_txns['day_of_week'].mode()[0],
        'is_weekend': all_txns['is_weekend'].mode()[0],
        'txn_month': all_txns['txn_month'].mode()[0],
        'txn_day': all_txns['txn_day'].mode()[0],
        'is_night': all_txns['is_night'].mode()[0],
        'is_working_hour': all_txns['is_working_hour'].mode()[0],
        'payer_freq': len(payer_txns),
        'beneficiary_freq': len(beneficiary_txns),
        'amount_category': pd.cut(
            [all_txns['AMOUNT'].mean()],
            bins=[0, 500, 2000, 5000, 10000, 20000, 50000, float('inf')],
            labels=['micro', 'small', 'medium', 'large', 'xlarge', 'xxlarge', 'xxxlarge']
        )[0],
        'TRN_STATUS': all_txns['TRN_STATUS'].mode()[0],
        'RESPONSE_CODE': all_txns['RESPONSE_CODE'].mode()[0],
        'TRANSACTION_TYPE': all_txns['TRANSACTION_TYPE'].mode()[0],
        'PAYMENT_INSTRUMENT': all_txns['PAYMENT_INSTRUMENT'].mode()[0],
        'payer_failed_ratio': payer_txns['TRN_STATUS'].eq('FAILED').mean() if len(payer_txns) > 0 else 0,
        'payer_unique_beneficiaries': payer_txns['BENEFICIARY_VPA'].nunique() if len(payer_txns) > 0 else 0,
        'beneficiary_unique_payers': beneficiary_txns['PAYER_VPA'].nunique() if len(beneficiary_txns) > 0 else 0,
        'payer_recent_frauds': payer_txns['IS_FRAUD'].tail(10).sum() if len(payer_txns) > 0 else 0,
        'beneficiary_recent_frauds': beneficiary_txns['IS_FRAUD'].tail(10).sum() if len(beneficiary_txns) > 0 else 0,
        'amt_x_failed': all_txns['AMOUNT'].mean() * payer_txns['TRN_STATUS'].eq('FAILED').mean() if len(payer_txns) > 0 else 0,
        'freq_x_failed': len(payer_txns) * payer_txns['TRN_STATUS'].eq('FAILED').mean() if len(payer_txns) > 0 else 0,
        'amt_x_suspicious': all_txns['AMOUNT'].mean() * payer_txns['payer_suspicious_score'].mean() if len(payer_txns) > 0 else 0,
        # --- FRAUD-FOCUSED FEATURES ---
        'payer_fraud_ratio': payer_txns['IS_FRAUD'].mean() if len(payer_txns) > 0 else 0,
        'beneficiary_fraud_ratio': beneficiary_txns['IS_FRAUD'].mean() if len(beneficiary_txns) > 0 else 0,
        'payer_total_frauds': payer_txns['IS_FRAUD'].sum() if len(payer_txns) > 0 else 0,
        'beneficiary_total_frauds': beneficiary_txns['IS_FRAUD'].sum() if len(beneficiary_txns) > 0 else 0,
        'payer_recent_frauds_5': payer_txns['IS_FRAUD'].tail(5).sum() if len(payer_txns) > 0 else 0,
        'beneficiary_recent_frauds_5': beneficiary_txns['IS_FRAUD'].tail(5).sum() if len(beneficiary_txns) > 0 else 0,
        'payer_recent_frauds_20': payer_txns['IS_FRAUD'].tail(20).sum() if len(payer_txns) > 0 else 0,
        'beneficiary_recent_frauds_20': beneficiary_txns['IS_FRAUD'].tail(20).sum() if len(beneficiary_txns) > 0 else 0,
        'payer_fraud_frequency': payer_txns['IS_FRAUD'].tail(50).mean() if len(payer_txns) > 0 else 0,
        'beneficiary_fraud_frequency': beneficiary_txns['IS_FRAUD'].tail(50).mean() if len(beneficiary_txns) > 0 else 0,
        'total_recent_frauds': (payer_txns['IS_FRAUD'].tail(10).sum() if len(payer_txns) > 0 else 0) + 
                              (beneficiary_txns['IS_FRAUD'].tail(10).sum() if len(beneficiary_txns) > 0 else 0),
        'total_fraud_ratio': (payer_txns['IS_FRAUD'].mean() if len(payer_txns) > 0 else 0) + 
                            (beneficiary_txns['IS_FRAUD'].mean() if len(beneficiary_txns) > 0 else 0),
        'is_high_fraud_payer': 1 if (payer_txns['IS_FRAUD'].mean() > 0.3 if len(payer_txns) > 0 else 0) else 0,
        'is_high_fraud_beneficiary': 1 if (beneficiary_txns['IS_FRAUD'].mean() > 0.3 if len(beneficiary_txns) > 0 else 0) else 0,
        'has_recent_fraud': 1 if ((payer_txns['IS_FRAUD'].tail(10).sum() > 0 if len(payer_txns) > 0 else 0) or 
                                  (beneficiary_txns['IS_FRAUD'].tail(10).sum() > 0 if len(beneficiary_txns) > 0 else 0)) else 0,
    }
    payer_upi_feats = upi_id_features(upi_id)
    for k, v in payer_upi_feats.items():
        features_dict[f'payer_{k}'] = v
    if len(beneficiary_txns) > 0:
        beneficiary_upi = beneficiary_txns['BENEFICIARY_VPA'].mode()[0]
        beneficiary_upi_feats = upi_id_features(beneficiary_upi)
        for k, v in beneficiary_upi_feats.items():
            features_dict[f'beneficiary_{k}'] = v
    else:
        for k in ['upi_id_length', 'has_number', 'has_special', 'bank_code', 'suspicious_score']:
            features_dict[f'beneficiary_{k}'] = 0
    # Add anomaly score for this UPI's transactions
    if 'anomaly_score' in all_txns:
        anomaly_score = all_txns['anomaly_score'].mean()
    else:
        anomaly_score = 0
    features_dict['anomaly_score'] = anomaly_score
    # Add weighted features
    features_dict['beneficiary_recent_frauds_weighted'] = features_dict['beneficiary_recent_frauds'] * 50
    features_dict['payer_recent_frauds_weighted'] = features_dict['payer_recent_frauds'] * 50
    features_dict['payer_fraud_ratio_weighted'] = features_dict['payer_fraud_ratio'] * 100
    features_dict['beneficiary_fraud_ratio_weighted'] = features_dict['beneficiary_fraud_ratio'] * 100
    features_dict['payer_total_frauds_weighted'] = features_dict['payer_total_frauds'] * 50
    features_dict['beneficiary_total_frauds_weighted'] = features_dict['beneficiary_total_frauds'] * 50
    features_dict['total_recent_frauds_weighted'] = features_dict['total_recent_frauds'] * 50
    features_dict['total_fraud_ratio_weighted'] = features_dict['total_fraud_ratio'] * 100
    max_payers = data['beneficiary_unique_payers'].max() if 'beneficiary_unique_payers' in data else 100
    features_dict['beneficiary_unique_payers_weighted'] = (max_payers - features_dict['beneficiary_unique_payers']) * 2
    input_data = pd.DataFrame([features_dict])
    proba = model.predict_proba(input_data)[0]
    fraud_prob = proba[1] * 100
    # If anomaly_score is high, boost fraud probability
    if anomaly_score > 0.5:
        fraud_prob = min(fraud_prob + 10, 100)
    # Additional fraud probability boost based on fraud indicators
    if features_dict['has_recent_fraud'] == 1:
        fraud_prob = min(fraud_prob + 15, 100)
    if features_dict['is_high_fraud_payer'] == 1 or features_dict['is_high_fraud_beneficiary'] == 1:
        fraud_prob = min(fraud_prob + 20, 100)
    # Default risk logic
    if fraud_prob < 40:
        risk_level = "Safe"
    elif fraud_prob < 60:
        risk_level = "Low Risk"
    elif fraud_prob < 80:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"
    # Safety status: only below 40 is Safe
    if fraud_prob < 40:
        safety_status = "Safe"
    else:
        safety_status = "Not Safe"
    return {
        "upi_id": upi_id,
        "fraud_probability": float(round(fraud_prob, 2)),
        "risk_level": risk_level,
        "safety_status": safety_status,
        "safe_percentage": float(round(100 - fraud_prob, 2)),
        "total_transactions": int(len(all_txns)),
        "failed_transactions": int(len(all_txns[all_txns['TRN_STATUS'] == 'FAILED'])),
        "average_amount": float(round(all_txns['AMOUNT'].mean(), 2)),
        "payer_failed_ratio": float(round(features_dict['payer_failed_ratio'], 2)),
        "payer_unique_beneficiaries": int(features_dict['payer_unique_beneficiaries']),
        "beneficiary_unique_payers": int(features_dict['beneficiary_unique_payers']),
        "payer_recent_frauds": int(features_dict['payer_recent_frauds']),
        "beneficiary_recent_frauds": int(features_dict['beneficiary_recent_frauds']),
        "payer_fraud_ratio": float(round(features_dict['payer_fraud_ratio'], 3)),
        "beneficiary_fraud_ratio": float(round(features_dict['beneficiary_fraud_ratio'], 3)),
        "total_fraud_ratio": float(round(features_dict['total_fraud_ratio'], 3)),
        "has_recent_fraud": bool(features_dict['has_recent_fraud']),
        "is_high_fraud_payer": bool(features_dict['is_high_fraud_payer']),
        "is_high_fraud_beneficiary": bool(features_dict['is_high_fraud_beneficiary']),
        "amt_x_failed": float(round(features_dict['amt_x_failed'], 2)),
        "freq_x_failed": float(round(features_dict['freq_x_failed'], 2)),
        "amt_x_suspicious": float(round(features_dict['amt_x_suspicious'], 2))
    }

# Flask API
app = Flask(__name__)
model = load_model()
data, _, _ = load_and_preprocess_data()

@app.route('/predict-upi-fraud', methods=['POST'])
def predict():
    print("Received request on /predict-upi-fraud")  # Debug log
    req = request.get_json()
    print("Request JSON:", req)  # Debug log
    upi_id = req.get('upi_id')
    if not upi_id:
        print("Error: upi_id is required")
        return jsonify({'error': 'upi_id is required'}), 400
    result = predict_upi_fraud(upi_id, data, model)
    print("Prediction result (full):", result)  # Debug log

    # Only return required fields
    filtered_result = {
        "upi_id": result.get("upi_id"),
        "fraud_probability": result.get("fraud_probability"),
        "risk_level": result.get("risk_level"),
        "safety_status": result.get("safety_status"),
        "beneficiary_recent_frauds": result.get("beneficiary_recent_frauds"),
        "payer_recent_frauds": result.get("payer_recent_frauds"),
    }
    print("Filtered result (sent to backend):", filtered_result)
    return jsonify(filtered_result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, debug=True) 
