import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib

# Load datasets
eng_path = r'cybercrime_keyword_dataset.csv'
hing_path = r'cybercrime_keyword_dataset_hinglish.csv'

df_eng = pd.read_csv(eng_path)
df_hing = pd.read_csv(hing_path)

# Combine datasets
all_data = pd.concat([df_eng, df_hing], ignore_index=True)

# Drop rows with missing values in key columns
all_data = all_data.dropna(subset=['Keyword/Phrase', 'Category'])

# Features and labels
X = all_data['Keyword/Phrase'].astype(str)
y = all_data['Category'].astype(str)

# Encode labels if not binary
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Build pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=5000)),
    ('clf', LogisticRegression(max_iter=1000))
])

# Train model
pipeline.fit(X_train, y_train)

# Evaluate
score = pipeline.score(X_test, y_test)
print(f"Validation Accuracy: {score:.4f}")

# Save model and label encoder
joblib.dump({'model': pipeline, 'label_encoder': label_encoder}, 'scam_detector_model.pkl')
print("Model saved as scam_detector_model.pkl") 