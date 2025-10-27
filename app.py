import streamlit as st
import cv2
import numpy as np
import time
import plotly.express as px
import plotly.graph_objects as go
from parking_detector import ParkingDetector
from utils import (
    resize_frame, save_uploaded_file, cleanup_temp_file, 
    get_video_info, create_sample_parking_image, 
    format_time, calculate_fps, create_occupancy_chart
)

# Page configuration
st.set_page_config(
    page_title="🅿️ AI-Powered Parking Detection System",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .status-occupied {
        color: #FF6B6B;
        font-weight: bold;
    }
    
    .status-free {
        color: #4ECDC4;
        font-weight: bold;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🅿️ AI-Powered Parking Detection System</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    
    # Model settings
    st.sidebar.subheader("🔧 Detection Settings")
    confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
    num_zones = st.sidebar.slider("Number of Parking Zones", 4, 12, 6, 2)
    
    # Processing options
    st.sidebar.subheader("📹 Processing Options")
    show_original = st.sidebar.checkbox("Show Original Video", False)
    skip_frames = st.sidebar.slider("Skip Frames (for faster processing)", 1, 10, 2)
    
    # Initialize detector
    if 'detector' not in st.session_state:
        with st.spinner("Loading YOLOv8 model..."):
            st.session_state.detector = ParkingDetector(confidence=confidence)
    
    # Update detector settings
    st.session_state.detector.confidence = confidence
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📹 Video Analysis", "📊 Analytics", "ℹ️ About"])
    
    with tab1:
        st.subheader("🎥 Upload and Analyze Video")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a video file", 
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file to analyze parking occupancy"
        )
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if uploaded_file is not None:
                # Save uploaded file
                video_path = save_uploaded_file(uploaded_file)
                
                # Get video info
                width, height, fps = get_video_info(video_path)
                
                st.success(f"✅ Video loaded successfully!")
                st.info(f"📐 Resolution: {width}x{height} | 🎬 FPS: {fps:.1f}")
                
                # Process video
                if st.button("🚀 Start Analysis", type="primary"):
                    process_video(video_path, num_zones, skip_frames, show_original)
                
                # Cleanup
                cleanup_temp_file(video_path)
                
            else:
                # Show sample image
                st.info("👆 Upload a video file to get started, or try the sample below")
                
                # Create and display sample parking image
                sample_img = create_sample_parking_image()
                st.image(sample_img, caption="Sample Parking Lot Layout", use_column_width=True)
                
                if st.button("🎯 Analyze Sample Image", type="secondary"):
                    analyze_sample_image(num_zones)
        
        with col2:
            st.subheader("📈 Real-time Stats")
            
            # Placeholder for real-time metrics
            if 'current_stats' in st.session_state:
                stats = st.session_state.current_stats
                
                st.metric("Total Zones", stats.get('total_zones', 0))
                st.metric("Occupied", stats.get('occupied_zones', 0), 
                        delta=f"{stats.get('occupancy_rate', 0):.1f}%")
                st.metric("Free Spaces", stats.get('free_zones', 0))
                
                # Occupancy gauge
                occupancy_rate = stats.get('occupancy_rate', 0)
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = occupancy_rate,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Occupancy Rate (%)"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("📊 Parking Analytics Dashboard")
        
        if 'occupancy_history' in st.session_state and st.session_state.occupancy_history:
            # Create time series chart
            df = st.session_state.occupancy_history
            
            fig = px.line(df, x='timestamp', y='occupancy_rate', 
                         title='Occupancy Rate Over Time',
                         labels={'occupancy_rate': 'Occupancy Rate (%)', 'timestamp': 'Time'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Zone-wise analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Zone Status")
                for i, zone_data in enumerate(st.session_state.detector.zone_occupancy.values()):
                    status = "🟥 Occupied" if zone_data['occupied'] else "🟩 Free"
                    st.write(f"Zone {i+1}: {status}")
            
            with col2:
                # Occupancy pie chart
                if st.session_state.current_stats:
                    chart_data = create_occupancy_chart(st.session_state.current_stats)
                    if chart_data:
                        fig = px.pie(values=chart_data['values'], names=chart_data['labels'],
                                   title="Parking Space Distribution",
                                   color_discrete_sequence=chart_data['colors'])
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📹 Process a video to see analytics data")
    
    with tab3:
        st.subheader("ℹ️ About This System")
        
        st.markdown("""
        ### 🎯 Features
        - ✅ **Real-time vehicle detection** using YOLOv8
        - ✅ **Automatic parking-slot occupancy analysis**
        - ✅ **Color-coded visualization** (🔵 Occupied | 🟢 Free)
        - ✅ **CPU-friendly** — runs smoothly without GPU
        - ✅ **Interactive dashboard** built with Streamlit
        
        ### 🧠 Tech Stack
        - 🔹 **YOLOv8 (Ultralytics)** for object detection
        - 🔹 **OpenCV + NumPy** for image processing
        - 🔹 **Streamlit** for live analytics and visualization
        - 🔹 **Python** for integration and logic
        
        ### 🚀 Future Enhancements
        - 📷 Real-time webcam integration
        - ☁️ Cloud deployment via Render / Streamlit Cloud
        - 📊 Historical parking analytics
        - 🔔 Occupancy alerts and notifications
        
        ### 🎓 Learning Outcomes
        This project demonstrates:
        - Integration of deep learning models with web applications
        - Real-time computer vision processing
        - Interactive data visualization
        - Modern Python web development
        
        ---
        **Built with ❤️ using Python, YOLOv8, OpenCV, and Streamlit**
        """)

def process_video(video_path: str, num_zones: int, skip_frames: int, show_original: bool):
    """Process uploaded video for parking detection"""
    
    cap = cv2.VideoCapture(video_path)
    detector = st.session_state.detector
    
    # Initialize session state
    if 'occupancy_history' not in st.session_state:
        st.session_state.occupancy_history = []
    
    # Create placeholders for video display
    col1, col2 = st.columns(2)
    
    with col1:
        video_placeholder = st.empty()
        st.subheader("🎥 Processed Video")
    
    with col2:
        if show_original:
            original_placeholder = st.empty()
            st.subheader("📹 Original Video")
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames for faster processing
            if frame_count % skip_frames != 0:
                frame_count += 1
                continue
            
            # Define parking zones on first frame
            if frame_count == 0:
                detector.define_parking_zones(frame, num_zones)
            
            # Process frame
            annotated_frame, stats = detector.process_video_frame(frame)
            
            # Update session state
            st.session_state.current_stats = stats
            
            # Add to history
            st.session_state.occupancy_history.append({
                'timestamp': time.time(),
                'occupancy_rate': stats.get('occupancy_rate', 0),
                'occupied_zones': stats.get('occupied_zones', 0),
                'free_zones': stats.get('free_zones', 0)
            })
            
            # Resize frames for display
            display_frame = resize_frame(annotated_frame, 600)
            if show_original:
                original_display = resize_frame(frame, 600)
            
            # Convert BGR to RGB for Streamlit
            display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            if show_original:
                original_display_rgb = cv2.cvtColor(original_display, cv2.COLOR_BGR2RGB)
            
            # Display frames
            video_placeholder.image(display_frame_rgb, use_column_width=True)
            if show_original:
                original_placeholder.image(original_display_rgb, use_column_width=True)
            
            # Update progress
            fps = calculate_fps(start_time, frame_count)
            progress = min(frame_count / 100, 1.0)  # Approximate progress
            progress_bar.progress(progress)
            status_text.text(f"Processing... FPS: {fps:.1f} | Frame: {frame_count}")
            
            frame_count += 1
            
            # Add small delay to prevent overwhelming the UI
            time.sleep(0.1)
    
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
    
    finally:
        cap.release()
        progress_bar.progress(1.0)
        status_text.text("✅ Processing complete!")
        
        # Show final statistics
        if st.session_state.current_stats:
            st.success("🎉 Analysis completed successfully!")
            stats = st.session_state.current_stats
            st.metric("Final Occupancy Rate", f"{stats['occupancy_rate']:.1f}%")

def analyze_sample_image(num_zones: int):
    """Analyze the sample parking image"""
    
    detector = st.session_state.detector
    sample_img = create_sample_parking_image()
    
    # Process sample image
    annotated_frame, stats = detector.process_video_frame(sample_img)
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), 
                caption="Sample Analysis Results", use_column_width=True)
    
    with col2:
        st.subheader("📊 Sample Analysis Results")
        st.metric("Total Zones", stats.get('total_zones', 0))
        st.metric("Occupied Zones", stats.get('occupied_zones', 0))
        st.metric("Free Zones", stats.get('free_zones', 0))
        st.metric("Occupancy Rate", f"{stats.get('occupancy_rate', 0):.1f}%")
        
        # Show zone details
        st.subheader("Zone Details")
        for i, zone_data in enumerate(detector.zone_occupancy.values()):
            status = "🟥 Occupied" if zone_data['occupied'] else "🟩 Free"
            st.write(f"Zone {i+1}: {status}")

if __name__ == "__main__":
    main()
