#!/usr/bin/env python3
"""Test script to verify all imports work correctly"""

print("Testing imports...")

try:
    from flask import Flask
    print("✅ Flask imported successfully")
except ImportError as e:
    print(f"❌ Flask import failed: {e}")

try:
    from flask_cors import CORS
    print("✅ flask_cors imported successfully")
except ImportError as e:
    print(f"❌ flask_cors import failed: {e}")

try:
    import pytesseract
    print("✅ pytesseract imported successfully")
    # Test if tesseract is available
    try:
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract version: {version}")
    except Exception as e:
        print(f"⚠️ Tesseract not found: {e}")
except ImportError as e:
    print(f"❌ pytesseract import failed: {e}")

try:
    from PIL import Image
    print("✅ PIL imported successfully")
except ImportError as e:
    print(f"❌ PIL import failed: {e}")

try:
    import joblib
    print("✅ joblib imported successfully")
except ImportError as e:
    print(f"❌ joblib import failed: {e}")

try:
    import numpy as np
    print("✅ numpy imported successfully")
except ImportError as e:
    print(f"❌ numpy import failed: {e}")

try:
    import speech_recognition as sr
    print("✅ speech_recognition imported successfully")
except ImportError as e:
    print(f"❌ speech_recognition import failed: {e}")

print("\nImport test completed!") 