#!/bin/bash

# Start QR scanning service
python qr_scan_ml_service.py &
QR_PID=$!

# Start voice/text scam detection service
python voice_text_scam_service.py &
VOICE_PID=$!

# Start video detection service
python video_detection_service.py &
VIDEO_PID=$!

# Function to handle shutdown
function shutdown {
    echo "Shutting down AI services..."
    kill $QR_PID
    kill $VOICE_PID
    kill $VIDEO_PID
    exit 0
}

# Trap SIGTERM and SIGINT
trap shutdown SIGTERM SIGINT

# Keep the container running
while true; do
    sleep 1
done 