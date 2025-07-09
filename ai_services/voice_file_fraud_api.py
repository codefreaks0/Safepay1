import os
import io
import joblib
import pandas as pd
import numpy as np
import string
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import FeatureUnion
import speech_recognition as sr
from typing import List
import traceback

# Try to import pydub and check ffmpeg
try:
    from pydub import AudioSegment
    # Explicitly set ffmpeg path for Windows
    AudioSegment.converter = r"C:\\Users\\yashs\\Downloads\\ffmpeg-2025-06-16-git-e6fb8f373e-essentials_build\\ffmpeg-2025-06-16-git-e6fb8f373e-essentials_build\\bin\\ffmpeg.exe"
    if not os.path.exists(AudioSegment.converter):
        raise ImportError('ffmpeg not found at the specified path. Please check the path.')
except Exception as e:
    print('pydub/ffmpeg error:', e)
    AudioSegment = None

MODEL_PATH = 'ai_services/voice_fraud_best_model.joblib'
VECTORIZER_PATH = 'ai_services/voice_fraud_best_vectorizer.joblib'
DATA_PATH = 'ai_services/fraud_call.file'

app = FastAPI(title="Voice File Fraud Detection API (Robust)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def clean_text(text: str) -> str:
    # Lowercase, remove punctuation, remove extra spaces
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text

def extract_extra_features(texts: List[str]) -> np.ndarray:
    # Features: message length, digit count, special char count
    features = []
    for t in texts:
        length = len(t)
        digit_count = sum(c.isdigit() for c in t)
        special_count = sum(not c.isalnum() and not c.isspace() for c in t)
        features.append([length, digit_count, special_count])
    return np.array(features)

def train_model():
    # Read file, skip bad lines
    data = []
    with open(DATA_PATH, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            label, content = parts
            data.append((label, content))
    df = pd.DataFrame(data, columns=['label', 'content'])
    df['target'] = df['label'].map({'fraud': 1, 'normal': 0})
    df['content_clean'] = df['content'].apply(clean_text)
    X = df['content_clean']
    y = df['target']
    # TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1,3), stop_words='english')
    X_tfidf = vectorizer.fit_transform(X)
    # Extra features
    X_extra = extract_extra_features(df['content_clean'].tolist())
    # Combine features
    X_combined = np.hstack([X_tfidf.toarray(), X_extra])
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)
    # Try both models
    nb = MultinomialNB()
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    nb.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    nb_acc = accuracy_score(y_test, nb.predict(X_test))
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"Naive Bayes accuracy: {nb_acc:.3f}")
    print(f"Random Forest accuracy: {rf_acc:.3f}")
    # Pick best
    if rf_acc > nb_acc:
        best_model = rf
        print("Using Random Forest")
    else:
        best_model = nb
        print("Using Naive Bayes")
    print("Classification Report:\n", classification_report(y_test, best_model.predict(X_test)))
    print("Confusion Matrix:\n", confusion_matrix(y_test, best_model.predict(X_test)))
    # Save
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    return best_model, vectorizer

def load_model():
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
            model = joblib.load(MODEL_PATH)
            vectorizer = joblib.load(VECTORIZER_PATH)
        else:
            model, vectorizer = train_model()
        return model, vectorizer
    except Exception as e:
        print('Model loading/training error:', e)
        traceback.print_exc()
        raise

model, vectorizer = load_model()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze-voice-file")
async def analyze_voice_file(file: UploadFile = File(...)):
    temp_files = []
    try:
        # Check file type
        filename = file.filename.lower()
        if not (filename.endswith('.mp3') or filename.endswith('.wav') or filename.endswith('.ogg')):
            raise HTTPException(status_code=400, detail="Only mp3, wav, or ogg files are supported.")
        # Save file temporarily
        temp_input_path = f"temp_{filename}"
        with open(temp_input_path, 'wb') as f_out:
            f_out.write(await file.read())
        temp_files.append(temp_input_path)
        # Convert to wav if needed
        temp_wav_path = temp_input_path
        if filename.endswith('.mp3') or filename.endswith('.ogg'):
            if AudioSegment is None:
                raise HTTPException(status_code=500, detail="Audio conversion failed: pydub/ffmpeg not available.")
            temp_wav_path = temp_input_path.rsplit('.', 1)[0] + '.wav'
            try:
                if filename.endswith('.mp3'):
                    audio = AudioSegment.from_mp3(temp_input_path)
                else:
                    audio = AudioSegment.from_ogg(temp_input_path)
                audio.export(temp_wav_path, format="wav")
                temp_files.append(temp_wav_path)
            except Exception as e:
                print('Audio conversion failed:', e)
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"Audio conversion failed: {e}")
        # Transcribe audio
        recognizer = sr.Recognizer()
        try:
            with sr.AudioFile(temp_wav_path) as source:
                audio = recognizer.record(source)
            transcript = recognizer.recognize_google(audio)
        except Exception as e:
            print('Transcription failed:', e)
            traceback.print_exc()
            return {"error": f"Transcription failed: {e}"}
        # Predict
        transcript_clean = clean_text(transcript)
        X_tfidf = vectorizer.transform([transcript_clean])
        X_extra = extract_extra_features([transcript_clean])
        X_combined = np.hstack([X_tfidf.toarray(), X_extra])
        try:
            pred = model.predict(X_combined)[0]
            proba = model.predict_proba(X_combined)[0][pred]
        except Exception as e:
            print('Prediction failed:', e)
            traceback.print_exc()
            return {"error": f"Prediction failed: {e}"}
        label = 'fraud' if pred == 1 else 'normal'
        return {"label": label, "probability": float(proba), "transcript": transcript}
    except HTTPException as he:
        raise he
    except Exception as e:
        print('Server error:', e)
        traceback.print_exc()
        return {"error": f"Server error: {e}"}
    finally:
        # Clean up temp files
        for path in temp_files:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                print('Temp file cleanup error:', e)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8084) 