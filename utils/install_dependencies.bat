
@echo off
echo 🚀 SafePay Windows Setup Starting...

echo 📦 Installing Node.js dependencies...
npm install

echo 🐍 Installing Python dependencies...
pip install -r requirements_qr.txt
pip install fastapi uvicorn python-multipart
pip install opencv-python-headless
pip install scikit-learn pandas numpy
pip install librosa
pip install groq-sdk openai

echo ✅ Dependencies installed successfully!
echo Please update .env file with your credentials
pause
