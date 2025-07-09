from flask import Flask, request, jsonify
import joblib
import numpy as np
import speech_recognition as sr
import os
import pytesseract
from PIL import Image
import io
from video_detection import ScamVideoDetector
import tempfile
import requests

app = Flask(__name__)

# Explicitly set tesseract_cmd for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return response

# Load model and label encoder
model_data = joblib.load(os.path.join(os.path.dirname(__file__), 'scam_detector_model.pkl'))
                model = model_data['model']
                label_encoder = model_data['label_encoder']
            
# Initialize the video scam detector
                video_detector = ScamVideoDetector()

def extract_text_with_groq(image_file):
    # Use pytesseract for local OCR
    try:
        image = Image.open(image_file.stream)
        text = pytesseract.image_to_string(image)
    return text
    except Exception as e:
        print('OCR error:', e)
        return ''

@app.route('/predict-text', methods=['POST'])
def predict_text():
        data = request.get_json()
    text = data.get('text', '')
        if not text:
        return jsonify({'error': 'No text provided'}), 400
    proba = model.predict_proba([text])[0]
    label_idx = np.argmax(proba)
    label = label_encoder.inverse_transform([label_idx])[0]
    probability = float(proba[label_idx])
    return jsonify({'label': label, 'probability': probability})

@app.route('/predict-audio', methods=['POST'])
def predict_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    audio_file = request.files['audio']
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
    try:
        transcript = recognizer.recognize_google(audio)
    except Exception as e:
        return jsonify({'error': 'Speech recognition failed', 'details': str(e)}), 500
    proba = model.predict_proba([transcript])[0]
    label_idx = np.argmax(proba)
    label = label_encoder.inverse_transform([label_idx])[0]
    probability = float(proba[label_idx])
    return jsonify({'label': label, 'probability': probability, 'transcript': transcript})
    
@app.route('/ocr-extract', methods=['POST'])
def ocr_extract():
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        image_file = request.files['image']
    try:
        image = Image.open(image_file.stream)
        text = pytesseract.image_to_string(image)
        return jsonify({'text': text})
    except Exception as e:
        return jsonify({'error': 'OCR failed', 'details': str(e)}), 500

@app.route('/analyze-video', methods=['POST'])
def analyze_video():
        if 'video_file' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        video_file = request.files['video_file']
    # Save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            video_file.save(temp_video.name)
            video_path = temp_video.name
    # Run the actual video analysis
            result = video_detector.analyze_video(video_path)
    # Clean up the temp file
    os.remove(video_path)
            return jsonify(result)

@app.route('/analyze-whatsapp', methods=['POST'])
def analyze_whatsapp():
    print('Request files:', request.files)
        if 'screenshot' not in request.files:
            return jsonify({'error': 'No screenshot provided'}), 400
        image_file = request.files['screenshot']
    print('Received file:', image_file)
    print('File size:', getattr(image_file, 'content_length', 'unknown'))
    print('File type:', getattr(image_file, 'mimetype', 'unknown'))
    # Step 1: Extract text using pytesseract
    extracted_text = extract_text_with_groq(image_file)
        if not extracted_text.strip():
        return jsonify({'error': 'No text detected in image.'}), 400
    # Step 2: Analyze the extracted text using the existing text model
    proba = model.predict_proba([extracted_text])[0]
    label_idx = np.argmax(proba)
    label = label_encoder.inverse_transform([label_idx])[0]
    probability = float(proba[label_idx])
        is_scam = label.lower() == 'scam'
    reason = 'Scam keywords detected in message.' if is_scam else 'No strong indicators of scam detected.'
        return jsonify({
            'is_scam': is_scam,
            'confidence': probability,
            'reason': reason,
        'extracted_text': extracted_text
    })

@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8090) 