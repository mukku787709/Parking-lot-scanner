#!/usr/bin/env python3
"""
Demo script for AI-Powered Parking Detection System
This script demonstrates the core functionality without Streamlit
"""

import cv2
import numpy as np
from parking_detector import ParkingDetector
from utils import create_sample_parking_image, resize_frame
import time

def demo_parking_detection():
    """Demonstrate parking detection on sample image"""
    
    print("🚀 AI-Powered Parking Detection System Demo")
    print("=" * 50)
    
    # Initialize detector
    print("🤖 Loading YOLOv8 model...")
    detector = ParkingDetector(confidence=0.5)
    
    # Create sample parking image
    print("📸 Creating sample parking lot image...")
    sample_img = create_sample_parking_image()
    
    # Process the image
    print("🔍 Analyzing parking occupancy...")
    start_time = time.time()
    
    annotated_frame, stats = detector.process_video_frame(sample_img)
    
    processing_time = time.time() - start_time
    
    # Display results
    print("\n📊 Analysis Results:")
    print(f"⏱️  Processing Time: {processing_time:.2f} seconds")
    print(f"🅿️  Total Zones: {stats.get('total_zones', 0)}")
    print(f"🟥 Occupied Zones: {stats.get('occupied_zones', 0)}")
    print(f"🟩 Free Zones: {stats.get('free_zones', 0)}")
    print(f"📈 Occupancy Rate: {stats.get('occupancy_rate', 0):.1f}%")
    
    # Show zone details
    print("\n📍 Zone Details:")
    for i, zone_data in enumerate(detector.zone_occupancy.values()):
        status = "🟥 Occupied" if zone_data['occupied'] else "🟩 Free"
        print(f"   Zone {i+1}: {status}")
    
    # Save results
    print("\n💾 Saving results...")
    cv2.imwrite("outputs/sample_analysis.jpg", annotated_frame)
    print("✅ Results saved to 'outputs/sample_analysis.jpg'")
    
    # Display the image (if running in environment that supports it)
    try:
        cv2.imshow("Parking Detection Results", resize_frame(annotated_frame, 800))
        print("\n🖼️  Press any key to close the image window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("🖼️  Image display not available in this environment")
    
    print("\n🎉 Demo completed successfully!")
    print("\n💡 To run the full Streamlit app:")
    print("   streamlit run app.py")

def demo_video_processing():
    """Demonstrate video processing capabilities"""
    
    print("\n🎥 Video Processing Demo")
    print("=" * 30)
    
    # Create a simple video simulation
    print("📹 Creating simulated video frames...")
    
    detector = ParkingDetector(confidence=0.5)
    
    # Simulate processing multiple frames
    for frame_num in range(5):
        # Create slightly different parking scenarios
        sample_img = create_sample_parking_image()
        
        # Add some variation to simulate different frames
        if frame_num % 2 == 0:
            # Add noise to simulate different lighting
            noise = np.random.randint(0, 50, sample_img.shape, dtype=np.uint8)
            sample_img = cv2.add(sample_img, noise)
        
        # Process frame
        annotated_frame, stats = detector.process_video_frame(sample_img)
        
        print(f"Frame {frame_num + 1}: Occupancy Rate = {stats.get('occupancy_rate', 0):.1f}%")
        
        # Save frame
        cv2.imwrite(f"outputs/frame_{frame_num + 1}.jpg", annotated_frame)
    
    print("✅ Video simulation completed!")
    print("📁 Check the 'outputs/' directory for processed frames")

if __name__ == "__main__":
    # Create outputs directory
    import os
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    
    # Run demos
    demo_parking_detection()
    demo_video_processing()
    
    print("\n🚀 Ready to run the full application!")
    print("   Use 'streamlit run app.py' to start the web interface")
