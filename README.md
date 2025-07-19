# SafePay - AI-Powered Payment Security Platform

SafePay is a comprehensive security platform that uses AI/ML to protect users from various types of payment fraud and scams. The platform includes multiple services for detecting QR code scams, voice/text scams, and video-based fraud.

## 🚀 Features

- **QR Code Security**: Advanced ML-based QR code scanning and risk assessment  
- **Voice/Text Scam Detection**: AI-powered detection of voice and text-based scams  
- **Video Fraud Detection**: Real-time video analysis for fraud detection  
- **UPI Fraud Prevention**: Specialized detection for UPI payment frauds  
- **Real-time Monitoring**: Continuous monitoring and alerting system  

## 🎥 Project Demo Video

[Watch the Demo on YouTube](https://youtu.be/iIcMu-H2q9s?si=1h5h4VzrtPoUri4K)

## 📑 Project Presentation (PPT)

[Download SafePay Project PPT](https://docs.google.com/presentation/d/1Yb3RwGixazAoptfjG-xXe5si-9WjMfM_/edit?usp=drive_link&ouid=112200168308218978257&rtpof=true&sd=true)

## 📸 UI Screenshots

### 💻 SafePay Screenshots

![Screenshot 1](https://raw.githubusercontent.com/codefreaks0/Safepay1/main/Images/Screenshot%202025-06-17%20095250.png)  
![Screenshot 2](https://raw.githubusercontent.com/codefreaks0/Safepay1/main/Images/Screenshot%202025-06-17%20100740.png)  
![Screenshot 3](https://raw.githubusercontent.com/codefreaks0/Safepay1/main/Images/Screenshot%202025-06-17%20101028.png)  
![Screenshot 4](https://raw.githubusercontent.com/codefreaks0/Safepay1/main/Images/Screenshot%202025-07-15%20235008.png)  
![Screenshot 5](https://raw.githubusercontent.com/codefreaks0/Safepay1/main/Images/Screenshot%202025-07-15%20235812.png)  
![Screenshot 6](https://raw.githubusercontent.com/codefreaks0/Safepay1/main/Images/Screenshot%202025-07-15%20235824.png)  
![Screenshot 7](https://raw.githubusercontent.com/codefreaks0/Safepay1/main/Images/Screenshot%202025-07-15%20235833.png)  
![Screenshot 8](https://raw.githubusercontent.com/codefreaks0/Safepay1/main/Images/Screenshot%202025-07-15%20235839.png)  
![Screenshot 9](https://raw.githubusercontent.com/codefreaks0/Safepay1/main/Images/Screenshot%202025-07-15%20235847.png)  
![Screenshot 10](https://raw.githubusercontent.com/codefreaks0/Safepay1/main/Images/Screenshot%202025-07-15%20235856.png)  
![Screenshot 11](https://raw.githubusercontent.com/codefreaks0/Safepay1/main/Images/Screenshot%202025-07-15%20235916.png)  
![Screenshot 12](https://raw.githubusercontent.com/codefreaks0/Safepay1/main/Images/Screenshot%202025-07-15%20235928.png)  
![Screenshot 13](https://raw.githubusercontent.com/codefreaks0/Safepay1/main/Images/Screenshot%202025-07-15%20235939.png)  
![Screenshot 14](https://raw.githubusercontent.com/codefreaks0/Safepay1/main/Images/Screenshot%202025-07-15%20235948.png)  
![Screenshot 15](https://raw.githubusercontent.com/codefreaks0/Safepay1/main/Images/Screenshot%202025-07-16%20000004.png)  
![Screenshot 16](https://raw.githubusercontent.com/codefreaks0/Safepay1/main/Images/Screenshot%202025-07-16%20000018.png)  
![Screenshot 17](https://raw.githubusercontent.com/codefreaks0/Safepay1/main/Images/Screenshot%202025-07-16%20000107.png)  
![Screenshot 18](https://raw.githubusercontent.com/codefreaks0/Safepay1/main/Images/Screenshot%202025-07-16%20000117.png)  
![Screenshot 19](https://raw.githubusercontent.com/codefreaks0/Safepay1/main/Images/Screenshot%202025-07-16%20000125.png)  
![Screenshot 20](https://raw.githubusercontent.com/codefreaks0/Safepay1/main/Images/Screenshot%202025-07-16%20000136.png)  
![Screenshot 21](https://raw.githubusercontent.com/codefreaks0/Safepay1/main/Images/Screenshot%202025-07-16%20000200.png)  
![Screenshot 22](https://raw.githubusercontent.com/codefreaks0/Safepay1/main/Images/Screenshot%202025-07-16%20000208.png)  
![Screenshot 23](https://raw.githubusercontent.com/codefreaks0/Safepay1/main/Images/Screenshot%202025-07-16%20000217.png)  
![Screenshot 24](https://raw.githubusercontent.com/codefreaks0/Safepay1/main/Images/Screenshot%202025-07-16%20000302.png)

---

## 🛠️ Known Issues & Future Roadmap


### ❗ Known Limitations (To Be Addressed)
- **Voice Detection Accuracy in Noisy Environments**
  - Real-time voice scam detection performance may degrade in environments with background noise, echo, or multiple speakers. Advanced noise filtering and speaker diarization are in progress.
- **QR Scanner Compatibility**
  - The current QR scanner may not work optimally on low-end mobile devices or outdated browsers due to hardware constraints or lack of camera access permissions.
- **Scam Heatmap Precision**
  - The scam location heatmap relies on IP-based geolocation which may not always reflect the user’s actual physical location accurately.
- **Limited Dataset for Regional Languages**
  - Scam messages and voice datasets are primarily in English or Hindi. Accuracy may be lower for regional dialects like Marathi, Bengali, Tamil, etc.
- **Model Bias**
  - Early ML models are trained on a limited set of labeled scams and may exhibit bias toward known patterns, missing novel fraud strategies.
- **No Offline Support Yet**
  - The platform currently requires internet connectivity for AI service calls, limiting its utility in poor-network areas.
- **Mobile Responsiveness Under Testing**
  - While the frontend is responsive, full UI testing across all screen sizes and OS/browser combinations is ongoing.

### 🔮 Upcoming Enhancements & Features (Planned)
- ✅ **UPI App Integration (Deep Linking)**
  - Direct integration with Google Pay, PhonePe, Paytm, and BHIM for seamless UPI redirection and safer in-app payments.
- 🧠 **Adaptive ML Models with Real-Time Learning**
  - Fraud detection models will adapt and retrain incrementally based on user feedback and newly flagged scam reports, enhancing system intelligence over time.
- 👆 **Biometric Verification for High-Risk Transactions**
  - Add fingerprint/face ID verification for transactions exceeding custom risk thresholds, enhancing user-level fraud mitigation.
- 📡 **Real-Time Scam Broadcasting**
  - Users in nearby regions receive push alerts when a scam is reported or confirmed, enabling proactive defense.
- 🔐 **End-to-End Encryption for Sensitive Logs**
  - Sensitive communication (voice recordings, UPI details) will be encrypted and anonymized in storage.
- 📊 **User Risk Scoring System**
  - Every user or UPI ID will have a trust/risk score based on behavior, reports, and transaction history to aid scam prevention.
- 🔎 **Explainable AI for Scam Detection**
  - Add model explainability layer so users can understand why a QR/voice/message was flagged as suspicious.
- 📱 **Native Mobile App (iOS + Android)**
  - Dedicated mobile app version under development for better performance, offline access, and biometric security.
- 🔍 **WhatsApp/Telegram Scam Link Detection**
  - Integration with messaging apps to detect suspicious links, shortened URLs, or scam groups in real-time.
- 👥 **Community Reporting & Validation System**
  - Crowdsource scam validation by allowing verified users to upvote/downvote or flag suspicious entities or numbers.

## 🏗️ Project Structure

```
Safepay/
├── frontend/         # React/TypeScript frontend application
├── backend/          # Python backend server
├── ai_services/      # AI/ML services and models
```

## 🛠️ Technology Stack

- **Frontend**: React, TypeScript, TailwindCSS
- **Backend**: Python, FastAPI
- **AI/ML**: TensorFlow, PyTorch, OpenCV,Flask,scikit-learn,pandas,joblib,SpeechRecognition,PyAudio,numpy,flask-cors
- **Database**: MongoDb
- **DevOps**: GitHub Actions

## 🚀 Getting Started

### Prerequisites

- Python 3.11
- Node.js 16+
- Docker (optional)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/codefreaks0/Safepay1
cd safepay
```

2. Install and run frontend:
```bash
cd frontend
npm install
npm run dev
```

3. Install and run backend:
```bash
cd backend
npm install
npm start
```


4. Start the services:
```bash
cd ai_services
python -m venv .venv
.venv/Scripts/activate
python -m pip install -r requirements.txt
python scam_detector_api.py
```
```bash
cd ai_services
python -m venv .venv
.venv/Scripts/activate
python -m pip install -r requirements.txt
python upi_fraud_model.py
```
```bash
cd ai_services
python -m venv .venv
.venv/Scripts/activate
python -m pip install -r requirements.txt
python voice_text_scam_service.py
```
```bash
cd ai_services
python -m venv .venv
.venv/Scripts/activate
python -m pip install -r requirements.txt
python video_detection_service.py
```

## 📘 API Documentation

### 🌐 Main Express Backend (Node.js)
**Base URL:** `http://localhost:6900/api/`

#### Authentication & User
- `POST /login` — User login
- `POST /signup` — User signup
- `POST /logout` — Logout
- `GET /profile/:userId` — Get user profile
- `PUT /profile/:userId` — Update user profile

#### Payment Methods
- `POST /api/payment-methods` — Add payment method
- `GET /api/payment-methods/:userId` — Get payment methods
- `DELETE /api/payment-methods/:methodId` — Delete payment method
- `POST /api/payment-methods/:userId/set-default/:methodId` — Set default payment method

#### Scam Reports
- `POST /api/scam-reports` — Add scam report
- `GET /api/scam-reports/:userId` — Get scam reports

#### Voice/Audio Analysis
- `POST /api/process-audio` — Analyze voice transcript or audio file for scam

#### Transaction
- `POST /api/transactions/process` — Process a new payment transaction
- `GET /api/transactions/:userId` — Get transaction history

#### WhatsApp/Message Analysis
- `POST /api/analyze-whatsapp` — Analyze WhatsApp screenshot for scam (proxies to Flask)
- `POST /api/analyze-text` — Analyze text message for scam (proxies to Flask)

#### OCR
- `POST /api/ocr-extract` — Extract text from image (proxies to Flask)

#### Video Analysis
- `POST /api/analyze-video` — Analyze video for scam (proxies to FastAPI)

#### UPI Risk
- `GET /api/upi/check/:upiId` — Dummy UPI risk analysis
- `POST /api/ai/validate-upi` — Dummy UPI validation

---

### 🐍 Flask ML Service (ai_services/scam_detector_api.py)
**Base URL:** `http://localhost:8090/`

- `POST /ocr-extract` — OCR text extraction from image
- `POST /predict-text` — Predict scam from text
- `POST /predict-audio` — Predict scam from audio file
- `POST /analyze-whatsapp` — Analyze WhatsApp screenshot for scam
- `POST /analyze-video` — Analyze video for scam (calls video detector)

---

### ⚡ FastAPI Voice/Text Scam Service (ai_services/voice_text_scam_service.py)
**Base URL:** `http://localhost:8082/`

- `GET /` — Service info
- `GET /status` — Health check
- `POST /analyze-voice` — Analyze voice transcript for scam
- `POST /analyze-text` — Analyze text message for scam
- `POST /batch-analyze-text` — Batch analyze multiple text messages

---

### ⚡ FastAPI Video Scam Service (ai_services/video_detection_service.py)
**Base URL:** `http://localhost:8083/`

- `GET /` — Service info
- `POST /analyze-video` — Analyze video for scam indicators

---

### 🏦 Flask UPI Fraud Model (ai_services/upi_fraud_model.py)
**Base URL:** (port as configured, e.g. `8091`)
- `POST /predict-upi-fraud` — Predict UPI fraud risk

---

### 🛠️ Service Ports (Standard)
| Service                | Port  |
|------------------------|-------|
| Express Backend        | 6900  |
| Flask ML Service       | 8090  |
| FastAPI Voice/Text     | 8082  |
| FastAPI Video          | 8083  |
| Flask UPI Fraud        | 8091  |

---


