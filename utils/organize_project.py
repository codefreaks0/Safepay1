#!/usr/bin/env python3
"""
Project Organization Script
This script organizes the project files into appropriate folders
"""
import os
import shutil
from pathlib import Path

def create_folder_if_not_exists(folder):
    """Create folder if it doesn't exist"""
    if not os.path.exists(folder):
        os.makedirs(folder)

def move_file(src, dest):
    """Move file from source to destination"""
    try:
        shutil.move(src, dest)
        print(f"✅ Moved {src} to {dest}")
    except Exception as e:
        print(f"❌ Failed to move {src}: {e}")

def main():
    # Create main folders
    folders = {
        'frontend': 'frontend',
        'backend': 'backend',
        'ai_services': 'ai_services',
        'utils': 'utils'
    }
    
    for folder in folders.values():
        create_folder_if_not_exists(folder)
    
    # Frontend files
    frontend_files = [
        'client/',
        'vite.config.ts',
        'tailwind.config.ts',
        'postcss.config.js',
        'tsconfig.json',
        'tsconfig.ci.json',
        'jsconfig.json',
        'package.json',
        'package-lock.json',
        'index.html',
        'theme.json'
    ]
    
    # Backend files
    backend_files = [
        'server/',
        'models/',
        'migrations/',
        'shared/',
        'upi_fraud_detection_service.py',
        'upi_fraud_detection_model.py',
        'fraud_map.py'
    ]
    
    # AI/ML services
    ai_files = [
        'qr_scan_ml_service.py',
        'voice_text_scam_service.py',
        'video_detection.py',
        'qr_risk_model.joblib',
        'voice_text_scam_model.py',
        'qr_risk_detection_model.py',
        'qr_dataset.csv',
        'scam_data.csv',
        'updated_scam_data.csv',
        'qr_data.json',
        'model.joblib',
        'scam_model.pkl',
        'train_video_model.py',
        'train_and_run_qr_model.py',
        'quick_train.py',
        'quick_test.py',
        'measure_qr_model_accuracy.py',
        'initialize_models.py',
        'extract_audio_for_training.py',
        'download_training_data.py',
        'copy_video_detection.py',
        'apply_voice_focus.py',
        'verify_detection.py',
        'test_video_detection.py',
        'test_upi_model.py',
        'test_optimized_qr.py'
    ]
    
    # Utility files
    utils_files = [
        'manage_ports.py',
        'start_services.py',
        '.port_config',
        'install_dependencies.bat',
        'setup.sh',
        'Dockerfile',
        'docker-compose.yml',
        'organize_project.py'
    ]
    
    # Move frontend files
    print("\n📦 Moving frontend files...")
    for file in frontend_files:
        if os.path.exists(file):
            move_file(file, os.path.join(folders['frontend'], os.path.basename(file)))
    
    # Move backend files
    print("\n📦 Moving backend files...")
    for file in backend_files:
        if os.path.exists(file):
            move_file(file, os.path.join(folders['backend'], os.path.basename(file)))
    
    # Move AI/ML files
    print("\n📦 Moving AI/ML files...")
    for file in ai_files:
        if os.path.exists(file):
            move_file(file, os.path.join(folders['ai_services'], os.path.basename(file)))
    
    # Move utility files
    print("\n📦 Moving utility files...")
    for file in utils_files:
        if os.path.exists(file):
            move_file(file, os.path.join(folders['utils'], os.path.basename(file)))
    
    print("\n✨ Project organization complete!")
    print("\nNew project structure:")
    print("""
Safepay/
├── frontend/         # Frontend application files
├── backend/          # Backend server files
├── ai_services/      # AI/ML services and models
└── utils/           # Utility scripts and configs
    """)

if __name__ == "__main__":
    main() 