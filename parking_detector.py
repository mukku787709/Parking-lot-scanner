import cv2
import numpy as np
from ultralytics import YOLO
import streamlit as st
from typing import List, Tuple, Dict
import time

class ParkingDetector:
    """
    AI-Powered Parking Space Detection System using YOLOv8
    """
    
    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.5):
        """
        Initialize the parking detector
        
        Args:
            model_path: Path to YOLO model
            confidence: Detection confidence threshold
        """
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        
        # Parking zones (will be defined based on video analysis)
        self.parking_zones = []
        self.zone_occupancy = {}
        
    def detect_vehicles(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect vehicles in the frame using YOLOv8
        
        Args:
            frame: Input video frame
            
        Returns:
            List of detected vehicles with bounding boxes
        """
        results = self.model(frame, conf=self.confidence, verbose=False)
        vehicles = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Check if detected object is a vehicle
                    if int(box.cls[0]) in self.vehicle_classes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0])
                        
                        vehicles.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence,
                            'class': int(box.cls[0]),
                            'class_name': self.model.names[int(box.cls[0])]
                        })
        
        return vehicles
    
    def define_parking_zones(self, frame: np.ndarray, num_zones: int = 6) -> List[Tuple]:
        """
        Automatically define parking zones based on frame dimensions
        
        Args:
            frame: Input frame
            num_zones: Number of parking zones to create
            
        Returns:
            List of parking zone coordinates
        """
        height, width = frame.shape[:2]
        
        # Create grid-based parking zones
        zones = []
        rows = 2
        cols = num_zones // rows
        
        zone_width = width // cols
        zone_height = height // rows
        
        for row in range(rows):
            for col in range(cols):
                x1 = col * zone_width
                y1 = row * zone_height
                x2 = x1 + zone_width
                y2 = y1 + zone_height
                
                zones.append((x1, y1, x2, y2))
        
        self.parking_zones = zones
        return zones
    
    def analyze_parking_occupancy(self, frame: np.ndarray, vehicles: List[Dict]) -> Dict:
        """
        Analyze parking zone occupancy based on detected vehicles
        
        Args:
            frame: Input frame
            vehicles: List of detected vehicles
            
        Returns:
            Dictionary with occupancy status for each zone
        """
        if not self.parking_zones:
            self.define_parking_zones(frame)
        
        occupancy = {}
        
        for i, zone in enumerate(self.parking_zones):
            zone_x1, zone_y1, zone_x2, zone_y2 = zone
            occupied = False
            
            # Check if any vehicle overlaps with this parking zone
            for vehicle in vehicles:
                v_x1, v_y1, v_x2, v_y2 = vehicle['bbox']
                
                # Calculate overlap
                overlap_x1 = max(zone_x1, v_x1)
                overlap_y1 = max(zone_y1, v_y1)
                overlap_x2 = min(zone_x2, v_x2)
                overlap_y2 = min(zone_y2, v_y2)
                
                if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                    # Calculate overlap area
                    overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                    zone_area = (zone_x2 - zone_x1) * (zone_y2 - zone_y1)
                    
                    # If overlap is significant (>30% of zone), consider occupied
                    if overlap_area / zone_area > 0.3:
                        occupied = True
                        break
            
            occupancy[i] = {
                'occupied': occupied,
                'zone': zone,
                'confidence': 0.9 if occupied else 0.1
            }
        
        self.zone_occupancy = occupancy
        return occupancy
    
    def visualize_detection(self, frame: np.ndarray, vehicles: List[Dict], occupancy: Dict) -> np.ndarray:
        """
        Visualize detection results with color-coded parking zones
        
        Args:
            frame: Input frame
            vehicles: List of detected vehicles
            occupancy: Parking zone occupancy status
            
        Returns:
            Annotated frame with visualizations
        """
        annotated_frame = frame.copy()
        
        # Draw parking zones with color coding
        for zone_id, zone_info in occupancy.items():
            zone = zone_info['zone']
            occupied = zone_info['occupied']
            
            # Color: Blue for occupied, Green for free (no red for vacant spaces)
            color = (255, 0, 0) if occupied else (0, 255, 0)  # BGR format: Blue for occupied, Green for free
            thickness = 3
            
            # Draw zone rectangle
            cv2.rectangle(annotated_frame, 
                         (zone[0], zone[1]), 
                         (zone[2], zone[3]), 
                         color, thickness)
            
            # Add zone label
            label = f"Zone {zone_id + 1}: {'🔵 Occupied' if occupied else '🟢 Free'}"
            cv2.putText(annotated_frame, label, 
                       (zone[0], zone[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw vehicle bounding boxes
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle['bbox']
            confidence = vehicle['confidence']
            class_name = vehicle['class_name']
            
            # Draw vehicle bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Add vehicle label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(annotated_frame, label, 
                       (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return annotated_frame
    
    def get_occupancy_stats(self) -> Dict:
        """
        Get parking occupancy statistics
        
        Returns:
            Dictionary with occupancy statistics
        """
        if not self.zone_occupancy:
            return {}
        
        total_zones = len(self.zone_occupancy)
        occupied_zones = sum(1 for zone in self.zone_occupancy.values() if zone['occupied'])
        free_zones = total_zones - occupied_zones
        
        occupancy_rate = (occupied_zones / total_zones) * 100 if total_zones > 0 else 0
        
        return {
            'total_zones': total_zones,
            'occupied_zones': occupied_zones,
            'free_zones': free_zones,
            'occupancy_rate': occupancy_rate
        }
    
    def process_video_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Process a single video frame for parking detection
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (annotated_frame, occupancy_stats)
        """
        # Detect vehicles
        vehicles = self.detect_vehicles(frame)
        
        # Analyze parking occupancy
        occupancy = self.analyze_parking_occupancy(frame, vehicles)
        
        # Visualize results
        annotated_frame = self.visualize_detection(frame, vehicles, occupancy)
        
        # Get statistics
        stats = self.get_occupancy_stats()
        
        return annotated_frame, stats
