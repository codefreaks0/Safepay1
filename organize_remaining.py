#!/usr/bin/env python3
"""
Script to organize remaining project files
"""
import os
import shutil

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
        'utils': 'utils',
        'docs': 'docs',
        'config': 'config',
        'data': 'data'
    }
    
    for folder in folders.values():
        create_folder_if_not_exists(folder)
    
    # Frontend files
    frontend_files = [
        'drizzle.config.ts',
        'generated-icon.png',
        'node_modules/',
        '.prettierrc',
        '.eslintrc.js'
    ]
    
    # Backend files
    backend_files = [
        'requirements_qr.txt',
        'pyproject.toml',
        'pyrightconfig.json',
        '.pylintrc',
        'qr_service.log'
    ]
    
    # AI/ML services
    ai_files = [
        'start_qr_service.py',
        'start_qr_ml_service.py',
        'start_optimized_qr_service.py',
        'start_enhanced_qr_service.py',
        'scam_model.py',
        'run_optimized_qr_service.py',
        'qr_scam_service.py',
        'optimized_qr_scanner.py',
        'optimized_qr_risk_service.py',
        'enhanced_qr_integration.py',
        'mydata.zip'
    ]
    
    # Utility files
    utils_files = [
        'uv.lock',
        '.dockerignore',
        '.editorconfig',
        '.gitignore',
        'cache/',
        'prediction_cache/'
    ]
    
    # Documentation files
    docs_files = [
        'VSCODE_DEVELOPMENT.md',
        'SECURITY.md',
        'README.md',
        'PULL_REQUEST_TEMPLATE.md',
        'PORT_CONFIGURATION.md',
        'LICENSE',
        'DEPLOYMENT_GUIDE.md',
        'CONTRIBUTING.md'
    ]
    
    # Configuration files
    config_files = [
        '.replit',
        '.replit.deploy',
        '.vscode/',
        '.streamlit/',
        '.github/',
        '.config/'
    ]
    
    # Data files
    data_files = [
        'attached_assets/'
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
    
    # Move documentation files
    print("\n📦 Moving documentation files...")
    for file in docs_files:
        if os.path.exists(file):
            move_file(file, os.path.join(folders['docs'], os.path.basename(file)))
    
    # Move configuration files
    print("\n📦 Moving configuration files...")
    for file in config_files:
        if os.path.exists(file):
            move_file(file, os.path.join(folders['config'], os.path.basename(file)))
    
    # Move data files
    print("\n📦 Moving data files...")
    for file in data_files:
        if os.path.exists(file):
            move_file(file, os.path.join(folders['data'], os.path.basename(file)))
    
    print("\n✨ Project organization complete!")
    print("\nNew project structure:")
    print("""
Safepay/
├── frontend/         # Frontend application files
├── backend/          # Backend server files
├── ai_services/      # AI/ML services and models
├── utils/           # Utility scripts and configs
├── docs/            # Documentation files
├── config/          # Configuration files
└── data/            # Data and assets
    """)

if __name__ == "__main__":
    main() 