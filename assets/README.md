# Sample Video Instructions

## 📹 Adding Sample Videos

To test the parking detection system, you can add sample videos to this directory.

### Recommended Video Types:
- **MP4** files (preferred)
- **AVI** files
- **MOV** files
- **MKV** files

### Video Requirements:
- **Resolution**: Any resolution (system will resize automatically)
- **Duration**: 10-60 seconds recommended for testing
- **Content**: Videos showing parking lots with vehicles
- **File Size**: Under 100MB for optimal performance

### Sample Video Sources:
1. **Traffic Cameras**: Public traffic camera feeds
2. **Parking Lot Cameras**: Security camera footage
3. **Drone Footage**: Aerial views of parking areas
4. **Simulated Videos**: Computer-generated parking scenarios

### Usage:
1. Place your video files in this `assets/` directory
2. Run the Streamlit app: `streamlit run app.py`
3. Use the file uploader to select your video
4. Click "Start Analysis" to begin processing

### Tips for Best Results:
- Ensure good lighting in the video
- Avoid extremely fast camera movements
- Include clear views of parking spaces
- Videos with multiple vehicles work best for testing

---

**Note**: The system will automatically create parking zones based on the video dimensions and detect vehicles using YOLOv11.
