import cv2
import numpy as np
import streamlit as st
from typing import Tuple, Optional
import tempfile
import os
import time

def resize_frame(frame: np.ndarray, max_width: int = 800) -> np.ndarray:
    """
    Resize frame while maintaining aspect ratio
    
    Args:
        frame: Input frame
        max_width: Maximum width for resizing
        
    Returns:
        Resized frame
    """
    height, width = frame.shape[:2]
    
    if width <= max_width:
        return frame
    
    # Calculate new dimensions
    ratio = max_width / width
    new_width = max_width
    new_height = int(height * ratio)
    
    return cv2.resize(frame, (new_width, new_height))

def save_uploaded_file(uploaded_file) -> str:
    """
    Save uploaded file to temporary location
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Path to saved file
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

def cleanup_temp_file(file_path: str):
    """
    Clean up temporary file
    
    Args:
        file_path: Path to file to delete
    """
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
    except Exception as e:
        st.warning(f"Could not delete temporary file: {e}")

def get_video_info(video_path: str) -> Tuple[int, int, float]:
    """
    Get video information
    
    Args:
        video_path: Path to video file
        
    Returns:
        Tuple of (width, height, fps)
    """
    cap = cv2.VideoCapture(video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    cap.release()
    
    return width, height, fps

def create_sample_parking_image() -> np.ndarray:
    """
    Create a sample parking lot image for demonstration
    
    Returns:
        Sample parking lot image
    """
    # Create a simple parking lot visualization
    img = np.ones((400, 800, 3), dtype=np.uint8) * 50  # Dark gray background
    
    # Draw parking spaces
    spaces = [
        (50, 50, 150, 150),   # Space 1
        (200, 50, 300, 150),  # Space 2
        (350, 50, 450, 150),  # Space 3
        (500, 50, 600, 150),  # Space 4
        (650, 50, 750, 150),  # Space 5
        (50, 200, 150, 300),  # Space 6
        (200, 200, 300, 300), # Space 7
        (350, 200, 450, 300), # Space 8
        (500, 200, 600, 300), # Space 9
        (650, 200, 750, 300), # Space 10
    ]
    
    # Draw parking spaces
    for i, (x1, y1, x2, y2) in enumerate(spaces):
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(img, f"P{i+1}", (x1+10, y1+30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add some sample vehicles
    vehicle_positions = [
        (70, 70, 130, 130),   # Car in space 1
        (220, 220, 280, 280), # Car in space 7
        (370, 70, 430, 130),  # Car in space 3
    ]
    
    for x1, y1, x2, y2 in vehicle_positions:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), -1)  # Red vehicles
        cv2.putText(img, "CAR", (x1+5, y1+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img

def format_time(seconds: float) -> str:
    """
    Format time in seconds to MM:SS format
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def calculate_fps(start_time: float, frame_count: int) -> float:
    """
    Calculate current FPS
    
    Args:
        start_time: Start time of processing
        frame_count: Number of frames processed
        
    Returns:
        Current FPS
    """
    elapsed_time = time.time() - start_time
    return frame_count / elapsed_time if elapsed_time > 0 else 0

def create_occupancy_chart(occupancy_data: dict) -> dict:
    """
    Create data for occupancy visualization
    
    Args:
        occupancy_data: Occupancy statistics
        
    Returns:
        Chart data dictionary
    """
    if not occupancy_data:
        return {}
    
    return {
        'labels': ['Occupied', 'Free'],
        'values': [occupancy_data['occupied_zones'], occupancy_data['free_zones']],
        'colors': ['#FF6B6B', '#4ECDC4']
    }
