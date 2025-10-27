#!/usr/bin/env python3
"""
Setup script for AI-Powered Parking Detection System
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("🔧 Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def check_yolo_model():
    """Check if YOLO model needs to be downloaded"""
    print("🤖 Checking YOLOv8 model...")
    
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        print("✅ YOLOv8 model ready!")
        return True
    except Exception as e:
        print(f"⚠️ Model download may be needed: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = ["assets", "logs", "outputs"]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"📁 Directory exists: {directory}")

def main():
    """Main setup function"""
    print("🚀 Setting up AI-Powered Parking Detection System")
    print("=" * 50)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during package installation")
        return False
    
    # Check YOLO model
    check_yolo_model()
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run the application: streamlit run app.py")
    print("2. Open your browser to the provided URL")
    print("3. Upload a video file to start analysis")
    print("\n💡 Tips:")
    print("- Place sample videos in the 'assets/' directory")
    print("- Adjust confidence threshold in the sidebar")
    print("- Use the sample image feature for testing")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
