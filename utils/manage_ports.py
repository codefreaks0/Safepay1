#!/usr/bin/env python3
"""
Port Management Utility for SafePay Application
"""
import os
import subprocess
import time
import requests
from typing import Dict, List

# Port Configuration
PORTS = {
    'frontend': int(os.environ.get('FRONTEND_PORT', 3000)),
    'backend': int(os.environ.get('PORT', 5001)),
    'qr_ml': int(os.environ.get('ML_QR_SERVICE_PORT', 8081)),
    'voice_ml': int(os.environ.get('ML_VOICE_TEXT_SERVICE_PORT', 8082)),
    'video_ml': int(os.environ.get('ML_VIDEO_SERVICE_PORT', 8083)),
}

def check_port_status(port: int) -> bool:
    """Check if a port is in use"""
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def get_available_ports() -> Dict[str, bool]:
    """Get status of all configured ports"""
    status = {}
    for service, port in PORTS.items():
        status[service] = {
            'port': port,
            'available': check_port_status(port)
        }
    return status

def print_port_status():
    """Print current port status"""
    print("🔍 Current Port Status:")
    print("-" * 40)
    status = get_available_ports()
    
    for service, info in status.items():
        service_name = service.replace('_', ' ').title()
        port = info['port']
        available = "✅ Running" if info['available'] else "❌ Not Running"
        print(f"{service_name:<15}: Port {port:<5} - {available}")

def kill_port(port: int):
    """Kill process running on specific port"""
    try:
        result = subprocess.run(['lsof', '-t', f'-i:{port}'], 
                              capture_output=True, text=True)
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                subprocess.run(['kill', '-9', pid])
            print(f"✅ Killed processes on port {port}")
        else:
            print(f"ℹ️  No processes found on port {port}")
    except Exception as e:
        print(f"❌ Error killing port {port}: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "status":
            print_port_status()
        elif command == "kill" and len(sys.argv) > 2:
            port = int(sys.argv[2])
            kill_port(port)
        elif command == "kill-all":
            for port in PORTS.values():
                kill_port(port)
        else:
            print("Usage: python manage_ports.py [status|kill <port>|kill-all]")
    else:
        print_port_status()
