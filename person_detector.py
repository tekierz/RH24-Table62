import cv2
import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List

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
    """Handles person detection, tracking, and trigger logic"""
    
    def __init__(self, model, confidence_threshold=0.45, center_region_scale=0.33):
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.center_region_scale = center_region_scale
        
        # Initialize face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Tracking state
        self.person_present = False
        self.current_person_id = None
        self.person_count = 0
        self.person_image = None
        self.consecutive_empty_frames = 0
        self.consecutive_frames_threshold = 30
        self.last_roast_time = 0
        self.roast_cooldown = 10
        
    def adjust_confidence(self, delta: float) -> None:
        """Adjust detection confidence threshold"""
        self.confidence_threshold = max(0.1, min(0.9, self.confidence_threshold + delta))
        self.model.conf = self.confidence_threshold
        
    def adjust_center_region(self, delta: float) -> None:
        """Adjust size of center detection region"""
        self.center_region_scale = max(0.1, min(0.9, self.center_region_scale + delta))
    
    def should_trigger_roast(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Check if we should trigger a new roast"""
        current_time = time.time()
        if (current_time - self.last_roast_time >= self.roast_cooldown and 
            self.person_image is not None):
            self.last_roast_time = current_time
            return True, self.person_image
        return False, None
    
    def process_frame(self, frame: np.ndarray) -> Tuple[List[DetectedPerson], str]:
        """
        Process a frame and return detected people and status message
        """
        frame_height, frame_width = frame.shape[:2]
        center_x = frame_width // 2
        center_y = frame_height // 2
        
        # Define center region
        center_region_width = int(frame_width * self.center_region_scale)
        center_region_height = int(frame_height * self.center_region_scale)
        
        # Run detection
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
                    
                    if (self.person_present and person.in_center and 
                        person.facing_forward and person.foreground_score >= 70):
                        person_still_in_center = True
        
        # Update tracking state
        status_message = self._update_tracking_state(detected_people, 
                                                   person_still_in_center, 
                                                   frame)
        
        return detected_people, status_message
    
    def _process_detection(self, frame, gray, box, center_x, center_y, 
                          center_region_width, center_region_height) -> DetectedPerson:
        """Process a single detection and return a DetectedPerson object"""
        person_height = box[3] - box[1]
        height_ratio = person_height / frame.shape[0]
        person_center_x = (box[0] + box[2]) // 2
        person_center_y = (box[1] + box[3]) // 2
        
        in_center_x = abs(person_center_x - center_x) < (center_region_width // 2)
        in_center_y = abs(person_center_y - center_y) < (center_region_height // 2)
        
        foreground_score = min(100, int((height_ratio / 0.15) * 100))
        
        person_roi = gray[box[1]:box[3], box[0]:box[2]]
        faces = self.face_cascade.detectMultiScale(person_roi, 1.1, 4)
        facing_forward = len(faces) > 0
        
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
        """Update tracking state and return status message"""
        if not detected_people or not person_still_in_center:
            self.consecutive_empty_frames += 1
        else:
            top_person = detected_people[0]
            if (top_person.in_center and 
                top_person.facing_forward and 
                top_person.foreground_score >= 70):
                
                if not self.person_present:
                    self.person_count += 1
                    self.current_person_id = self.person_count
                    self.person_present = True
                    self.person_image = frame.copy()
            
            self.consecutive_empty_frames = 0
        
        # Reset tracking when person leaves
        if self.consecutive_empty_frames >= self.consecutive_frames_threshold:
            if self.person_present:
                self.current_person_id = None
                self.person_present = False
                self.person_image = None
                self.last_roast_time = 0
            self.consecutive_empty_frames = 0
        
        # Generate status message
        return self._generate_status_message(person_still_in_center)
    
    def _generate_status_message(self, person_still_in_center: bool) -> str:
        """Generate status message based on current state"""
        status = f"Current: #{self.current_person_id} | " if self.current_person_id else ""
        
        if self.person_present:
            if person_still_in_center:
                status += "Person still in frame - waiting for exit"
            else:
                status += "Roasted - waiting for complete exit"
        else:
            status += "Ready for new person"
            
        return status