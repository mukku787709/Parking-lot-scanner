@echo off
echo 🚀 AI-Powered Parking Detection System Setup
echo ============================================

echo.
echo 📦 Installing Python packages...
pip install -r requirements.txt

echo.
echo 🤖 Downloading YOLOv8 model...
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

echo.
echo 📁 Creating directories...
if not exist "assets" mkdir assets
if not exist "logs" mkdir logs
if not exist "outputs" mkdir outputs

echo.
echo ✅ Setup completed!
echo.
echo 🚀 Starting the application...
streamlit run app.py

pause
