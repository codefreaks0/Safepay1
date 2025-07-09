
#!/bin/bash

echo "🚀 SafePay VS Code Setup Starting..."

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
npm install

# Install Python dependencies for QR detection
echo "🐍 Installing Python QR dependencies..."
pip3 install -r requirements_qr.txt

# Install additional Python dependencies
echo "🔧 Installing additional Python packages..."
pip3 install fastapi uvicorn python-multipart
pip3 install opencv-python-headless
pip3 install scikit-learn pandas numpy
pip3 install librosa torch torchaudio torchvision
pip3 install groq-sdk openai

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please update .env file with your API keys and database URL"
fi

echo "✅ Setup complete! Please:"
echo "1. Update .env file with your credentials"
echo "2. Setup PostgreSQL database"
echo "3. Run 'npm run dev' to start the application"
