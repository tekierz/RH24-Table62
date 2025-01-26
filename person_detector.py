import cv2 # type: ignore
import time
import numpy as np # type: ignore
from dataclasses import dataclass
from typing import Optional, Tuple, List
import logging

@dataclass
class DetectedPerson:
    """Data class to store information about a detected person"""
    box: List[int]
    priority_score: float
    foreground_score: float
    facing_forward: bool
    in_center: bool
    image: Optional[np.ndarray] = None

class PersonDetector:
    """Handles person detection, tracking, and trigger logic for YOLOv11"""
    
    def __init__(self, model, confidence_threshold=0.45, center_region_scale=0.33, debug=False):
        """Initialize detector with optional debug mode"""
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.center_region_scale = center_region_scale
        self.debug = debug
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)
        
        # Initialize face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Enhanced tracking state
        self.person_present = False
        self.current_person_id = None
        self.person_count = 0
        self.person_image = None
        self.consecutive_empty_frames = 0
        self.consecutive_frames_threshold = 90  # 3 seconds at 30fps
        self.last_person_exit_time = 0
        self.exit_cooldown = 3  # 3 second cooldown after person leaves
        self.last_roast_time = 0
        self.roast_cooldown = 10
        
        # Debug visualization
        self.show_detection_info = True
        
    def adjust_confidence(self, delta: float) -> None:
        """Adjust detection confidence threshold"""
        self.confidence_threshold = max(0.1, min(0.9, self.confidence_threshold + delta))
        self.model.conf = self.confidence_threshold
        
    def adjust_center_region(self, delta: float) -> None:
        """Adjust size of center detection region"""
        self.center_region_scale = max(0.1, min(0.9, self.center_region_scale + delta))
    
    def should_trigger_roast(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Check if conditions are met to trigger a new roast"""
        current_time = time.time()
        
        if (current_time - self.last_person_exit_time >= self.exit_cooldown and
            current_time - self.last_roast_time >= self.roast_cooldown and 
            self.person_image is not None):
            
            captured_image = self.person_image.copy()
            self.last_roast_time = current_time
            return True, captured_image
            
        return False, None
    
    def process_frame(self, frame: np.ndarray) -> Tuple[List[DetectedPerson], str]:
        """Process a frame and return detected people and status message"""
        frame_height, frame_width = frame.shape[:2]
        center_x = frame_width // 2
        center_y = frame_height // 2

        # Store original frame for display/capture
        display_frame = frame.copy()
        
        # Convert BGR to RGB only for YOLO detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Define center region
        center_region_width = int(frame_width * self.center_region_scale)
        center_region_height = int(frame_height * self.center_region_scale)
        
        # Draw center region if debug enabled
        if self.show_detection_info:
            cv2.rectangle(frame,
                         (center_x - center_region_width//2, center_y - center_region_height//2),
                         (center_x + center_region_width//2, center_y + center_region_height//2),
                         (0, 255, 0), 2)
        
        # Run YOLOv11 detection
        results = self.model(frame, verbose=False)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        detected_people = []
        person_still_in_center = False
        
        # Process detections
        for result in results[0].boxes.data:
            if int(result[5]) == 0:  # class 0 is person
                confidence = float(result[4])
                if confidence > self.confidence_threshold:
                    box = result[:4].int().tolist()
                    person = self._process_detection(frame, gray, box, 
                                                  center_x, center_y,
                                                  center_region_width, 
                                                  center_region_height)
                    detected_people.append(person)
                    
                    if self.show_detection_info:
                        self._draw_detection_info(frame, person, box)
                    
                    if (person.in_center and 
                        person.facing_forward and 
                        person.foreground_score >= 70):
                        person_still_in_center = True
        
        # Sort by priority score
        detected_people.sort(key=lambda x: x.priority_score, reverse=True)
        
        # Update tracking state and get status
        status = self._update_tracking_state(detected_people, person_still_in_center, frame)
        
        return detected_people, status
    
    def _process_detection(self, frame, gray, box, center_x, center_y, 
                         center_region_width, center_region_height) -> DetectedPerson:
        """Process individual detection and calculate metrics"""
        # Calculate metrics
        person_height = box[3] - box[1]
        height_ratio = person_height / frame.shape[0]
        person_center_x = (box[0] + box[2]) // 2
        person_center_y = (box[1] + box[3]) // 2
        
        # Check position
        in_center_x = abs(person_center_x - center_x) < (center_region_width * 0.6)
        in_center_y = abs(person_center_y - center_y) < (center_region_height * 0.6)
        
        # Calculate foreground score
        foreground_score = min(100, int((height_ratio / 0.12) * 100))
        
        # Check for face
        person_roi = gray[box[1]:box[3], box[0]:box[2]]
        faces = self.face_cascade.detectMultiScale(person_roi, 1.3, 3)
        facing_forward = len(faces) > 0
        
        # Calculate priority score
        priority_score = (
            (foreground_score * 0.4) +
            ((in_center_x and in_center_y) * 30) +
            (facing_forward * 30)
        )
        
        return DetectedPerson(
            box=box,
            priority_score=priority_score,
            foreground_score=foreground_score,
            facing_forward=facing_forward,
            in_center=in_center_x and in_center_y
        )
    
    def _update_tracking_state(self, detected_people: List[DetectedPerson], 
                             person_still_in_center: bool, 
                             frame: np.ndarray) -> str:
        current_time = time.time()
        
        if not detected_people:
            self.consecutive_empty_frames += 1
            if self.consecutive_empty_frames >= self.consecutive_frames_threshold:
                if self.person_present:
                    print("Person has left the frame")
                    self.last_person_exit_time = current_time
                    self.person_present = False
                    self.current_person_id = None
                    # Don't clear person_image here - keep it for the roast
        else:
            top_person = detected_people[0]
            
            # Debug print for detection criteria
            if self.debug:
                self.logger.debug(
                    f"Detection: in_center={top_person.in_center}, "
                    f"facing_forward={top_person.facing_forward}, "
                    f"foreground_score={top_person.foreground_score}"
                )
            
            if (top_person.in_center and 
                top_person.facing_forward and 
                top_person.foreground_score >= 70):
                
                if not self.person_present:
                    self.logger.info("New person detected!")  # Keep this as INFO level
                    self.person_count += 1
                    self.current_person_id = self.person_count
                    self.person_present = True
                
                # Always update the person image when criteria are met
                if self.debug:
                    self.logger.debug("Capturing new person image")
                self.person_image = frame.copy()
                self.consecutive_empty_frames = 0
            else:
                # Only increment empty frames if person doesn't meet criteria
                self.consecutive_empty_frames += 1
        
        return self._generate_status_message(person_still_in_center, current_time)
    
    def _generate_status_message(self, person_still_in_center: bool, current_time: float) -> str:
        """Generate detailed status message including timers"""
        status = f"Person #{self.current_person_id} | " if self.current_person_id else "Ready | "
        
        if self.person_present:
            if person_still_in_center:
                status += "In Frame"
            else:
                status += "Leaving Frame"
        else:
            exit_time_remaining = max(0, self.exit_cooldown - 
                                    (current_time - self.last_person_exit_time))
            if exit_time_remaining > 0:
                status += f"Exit Timer: {exit_time_remaining:.1f}s"
            else:
                status += "Ready for New Person"
        
        return status

    def _draw_detection_info(self, frame, person: DetectedPerson, box: List[int]) -> None:
        """Draw detection information on frame"""
        # Draw bounding box
        color = (0, 255, 0) if person.in_center else (255, 165, 0)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        
        # Draw detection info if enabled
        if self.show_detection_info:
            y_offset = box[1] - 10
            cv2.putText(frame, f"Score: {person.priority_score:.1f}", 
                      (box[0], y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            y_offset -= 20
            cv2.putText(frame, f"Foreground: {person.foreground_score:.0f}%", 
                      (box[0], y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            y_offset -= 20
            cv2.putText(frame, f"Face: {'Yes' if person.facing_forward else 'No'}", 
                      (box[0], y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)