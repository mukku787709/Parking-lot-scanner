# 🚀 Quick Start Guide

## AI-Powered Parking Detection System

### ⚡ Quick Setup (Windows)

1. **Double-click `run.bat`** - This will automatically:
   - Install all required packages
   - Download the YOLOv11 model
   - Start the Streamlit application

2. **Open your browser** to the provided URL (usually `http://localhost:8501`)

3. **Upload a video** or try the sample image feature

### 🐍 Manual Setup (All Platforms)

1. **Install Python packages:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   streamlit run app.py
   ```

3. **Open your browser** to the provided URL

### 🎯 Testing the System

#### Option 1: Sample Image Analysis
- Click "Analyze Sample Image" button
- See instant results with color-coded parking zones

#### Option 2: Video Upload
- Upload a video file (MP4, AVI, MOV, MKV)
- Click "Start Analysis"
- Watch real-time processing with live statistics

#### Option 3: Demo Script
- Run `python demo.py` for a command-line demonstration
- See sample analysis without the web interface

### ⚙️ Configuration Options

- **Confidence Threshold**: Adjust detection sensitivity (0.1-1.0)
- **Number of Zones**: Set parking zones (4-12)
- **Skip Frames**: Speed up processing by skipping frames
- **Show Original**: Display both original and processed video

### 📊 Features Overview

- ✅ **Real-time Detection**: YOLOv11-powered vehicle detection
- ✅ **Smart Zoning**: Automatic parking zone creation
- ✅ **Color Coding**: Red (occupied) / Green (free) visualization
- ✅ **Live Analytics**: Real-time occupancy statistics
- ✅ **Interactive Dashboard**: Beautiful Streamlit interface
- ✅ **CPU Optimized**: Runs smoothly without GPU

### 🔧 Troubleshooting

**Model Download Issues:**
- The system will automatically download YOLOv11 on first run
- Ensure internet connection for initial setup

**Performance Tips:**
- Use smaller video files for faster processing
- Increase "Skip Frames" setting for better performance
- Lower confidence threshold for more detections

**Browser Issues:**
- Use Chrome, Firefox, or Edge for best compatibility
- Clear browser cache if interface doesn't load properly

### 📁 Project Structure

```
parking-system/
├── app.py                 # Main Streamlit application
├── parking_detector.py    # Core detection logic
├── utils.py              # Utility functions
├── demo.py               # Command-line demo
├── setup.py              # Setup script
├── run.bat               # Windows quick start
├── requirements.txt       # Dependencies
├── README.md             # Full documentation
├── assets/               # Sample videos directory
└── outputs/              # Generated results
```

### 🎓 Learning Outcomes

This project demonstrates:
- **Computer Vision**: Real-time object detection
- **Web Development**: Interactive dashboards
- **Data Visualization**: Live analytics and charts
- **AI Integration**: Deep learning model deployment

---

**Ready to detect parking spaces? Let's go! 🅿️**
