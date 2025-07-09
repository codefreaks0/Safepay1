#!/usr/bin/env python3
"""
Service Starter Script for SafePay Application
This script starts all services on their designated ports
"""
import os
import subprocess
import time
import sys
from manage_ports import kill_port, PORTS

def start_service(service_name: str, command: str, port: int):
    """Start a service on its designated port"""
    print(f"🚀 Starting {service_name} on port {port}...")
    
    # Kill any existing process on this port
    kill_port(port)
    
    # Set the port environment variable
    env = os.environ.copy()
    env[f"{service_name.upper()}_PORT"] = str(port)
    
    # Start the service
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"✅ {service_name} started successfully!")
        return process
    except Exception as e:
        print(f"❌ Failed to start {service_name}: {e}")
        return None

def main():
    # Kill all existing services first
    print("🧹 Cleaning up existing services...")
    for port in PORTS.values():
        kill_port(port)
    
    # Start services
    processes = []
    
    # Start Frontend
    frontend_process = start_service(
        "frontend",
        "cd client && npm run dev",
        PORTS['frontend']
    )
    if frontend_process:
        processes.append(frontend_process)
    
    # Start Backend
    backend_process = start_service(
        "backend",
        "cd server && python app.py",
        PORTS['backend']
    )
    if backend_process:
        processes.append(backend_process)
    
    # Start QR ML Service
    qr_process = start_service(
        "qr_ml",
        "python qr_scan_ml_service.py",
        PORTS['qr_ml']
    )
    if qr_process:
        processes.append(qr_process)
    
    # Start Voice/Text ML Service
    voice_process = start_service(
        "voice_ml",
        "python voice_text_scam_service.py",
        PORTS['voice_ml']
    )
    if voice_process:
        processes.append(voice_process)
    
    # Start Video ML Service
    video_process = start_service(
        "video_ml",
        "python video_detection.py",
        PORTS['video_ml']
    )
    if video_process:
        processes.append(video_process)
    
    print("\n✨ All services started! Press Ctrl+C to stop all services.")
    
    try:
        # Keep the script running and monitor processes
        while True:
            for process in processes:
                if process.poll() is not None:
                    print(f"⚠️  A service has stopped unexpectedly!")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping all services...")
        for process in processes:
            process.terminate()
        print("✅ All services stopped!")

if __name__ == "__main__":
    main() 