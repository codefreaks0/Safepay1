# Voice Scam Detection Integration Guide

## 🎯 Overview
Your voice scam detection system is now fully integrated! Users can record voice, get transcripts, and receive real-time scam analysis using your trained model.

## 🚀 Setup Instructions

### 1. Start All Services

#### **A. Start the AI Service (Voice Scam Detection)**
```bash
cd ai_services
python voice_text_scam_service.py
```
- This starts the FastAPI service on port 8082
- You should see: "Starting Voice and Text Scam Detection API on port 8082..."

#### **B. Start the Backend (Node.js)**
```bash
cd backend
npm start
```
- This starts the Express server on port 6900
- You should see: "Node.js backend running on port 6900"

#### **C. Start the Frontend (React/Vite)**
```bash
cd frontend
npm run dev
```
- This starts the React app on port 5173
- You should see: "Local: http://localhost:5173/"

### 2. Train Your Model (First Time Only)
```bash
curl -X POST http://localhost:8082/train
```
- This trains the model with your enhanced dataset
- You should see accuracy scores for both voice and text models

## 🧪 Testing the Integration

### **Option 1: Use the Test Script**
```bash
node test_voice_integration.js
```
This will test both the AI service and backend endpoint with various scam/non-scam phrases.

### **Option 2: Manual Testing**

#### **Test AI Service Directly:**
```bash
curl -X POST http://localhost:8082/analyze-voice \
  -H "Content-Type: application/json" \
  -d '{"transcript": "Your account is suspended, verify now."}'
```

#### **Test Backend Endpoint:**
```bash
curl -X POST http://localhost:6900/api/process-audio \
  -H "Content-Type: application/json" \
  -d '{"transcript": "Your account is suspended, verify now."}'
```

### **Option 3: Frontend Testing**
1. Open your browser to `http://localhost:5173`
2. Navigate to the Voice Check page
3. Record voice or upload audio files
4. See real-time scam analysis results

## 📊 Expected Results

### **Scam Phrases (Should be flagged):**
- "Your account is suspended, verify now."
- "Congratulations! You've won a lottery prize."
- "Your KYC verification is pending."
- "Please share your OTP to continue."

### **Legitimate Phrases (Should be safe):**
- "Let's meet for lunch tomorrow."
- "Your order has been shipped."
- "Thank you for your payment."

## 🔧 Troubleshooting

### **Port Already in Use (Error 10048)**
```bash
# Find process using port 8082
netstat -ano | findstr :8082

# Kill the process
taskkill /PID <PID> /F
```

### **Model Not Training**
- Check if your dataset file exists at the specified path
- Ensure the CSV has the correct column names: `text` and `label`
- Verify the file is not corrupted

### **Frontend Not Connecting**
- Check if all services are running on correct ports
- Verify the Vite proxy configuration in `frontend/vite.config.ts`
- Check browser console for CORS errors

### **Low Model Accuracy**
- Add more training examples to your dataset
- Ensure balanced scam/non-scam examples
- Retrain the model after dataset updates

## 📁 File Structure
```
├── ai_services/
│   ├── voice_text_scam_service.py    # AI service (port 8082)
│   ├── voice_text_scam_model.py      # ML model code
│   └── models/                       # Trained model files
├── backend/
│   └── index.js                      # Express backend (port 6900)
├── frontend/
│   └── client/src/pages/voice-check.tsx  # Voice analysis UI
└── test_voice_integration.js         # Test script
```

## 🎉 Success Indicators

✅ **All services running without errors**  
✅ **Model training completes with accuracy > 50%**  
✅ **Test script passes all test cases**  
✅ **Frontend can record/upload and get analysis results**  
✅ **Scam phrases are correctly flagged**  
✅ **Legitimate phrases are marked as safe**

## 🔄 Next Steps

1. **Monitor Performance**: Check model accuracy on real-world data
2. **Improve Dataset**: Add more diverse scam/non-scam examples
3. **Fine-tune Model**: Adjust parameters for better accuracy
4. **Add Features**: Consider adding audio quality analysis
5. **Scale Up**: Deploy to production environment

## 📞 Support

If you encounter issues:
1. Check the console logs for error messages
2. Verify all services are running on correct ports
3. Test individual components using the test script
4. Ensure your dataset is properly formatted

---

**Your voice scam detection system is now ready to protect users from fraudulent calls! 🛡️** 