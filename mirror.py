import cv2
import numpy as np
import openai
import pyttsx3
import os
import base64
import time
from datetime import datetime
from dotenv import load_dotenv

class RoastingMirror:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize OpenAI client with API key from .env
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Initialize text-to-speech engine
        self.tts_engine = pyttsx3.init()
        
        # Initialize camera
        self.camera = cv2.VideoCapture(0)
        
        # Load pre-trained person detection model (HOG + Linear SVM)
        self.person_detector = cv2.HOGDescriptor()
        self.person_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Cooldown timer to prevent constant roasting
        self.last_roast_time = 0
        self.roast_cooldown = 10  # seconds
        
    def detect_person(self, frame):
        """Detect if a person is in the frame"""
        boxes, weights = self.person_detector.detectMultiScale(frame, winStride=(8, 8))
        return len(boxes) > 0
    
    def save_frame(self, frame):
        """Save the current frame as a temporary image"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"temp_frame_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        return filename
    
    def generate_roast(self, image_path):
        """Generate a roast using GPT-4 Vision"""
        try:
            with open(image_path, "rb") as image_file:
                response = self.client.chat.completions.create(
                    model="gpt-4-vision-preview",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a witty roast comedian. Keep roasts funny but not cruel."
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Roast the person in this image with a short, funny comment."},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(image_file.read()).decode()}"}}
                            ]
                        }
                    ],
                    max_tokens=100
                )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating roast: {str(e)}"
    
    def speak_roast(self, roast_text):
        """Convert text to speech and play it"""
        self.tts_engine.say(roast_text)
        self.tts_engine.runAndWait()
    
    def cleanup(self, image_path):
        """Clean up temporary files"""
        if os.path.exists(image_path):
            os.remove(image_path)
    
    def run(self):
        """Main application loop"""
        print("Starting Miragé - The Roasting Smart Mirror")
        
        while True:
            # Capture frame from camera
            ret, frame = self.camera.read()
            if not ret:
                print("Error capturing frame")
                break
            
            # Show mirror image (flipped horizontally)
            mirror_frame = cv2.flip(frame, 1)
            cv2.imshow('Miragé', mirror_frame)
            
            # Check if it's time for a new roast
            current_time = time.time()
            if current_time - self.last_roast_time >= self.roast_cooldown:
                # Check for person in frame
                if self.detect_person(frame):
                    # Save frame and generate roast
                    image_path = self.save_frame(frame)
                    roast = self.generate_roast(image_path)
                    
                    # Speak the roast
                    self.speak_roast(roast)
                    
                    # Clean up
                    self.cleanup(image_path)
                    
                    # Update last roast time
                    self.last_roast_time = current_time
            
            # Check for quit command (q key)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Check for OpenAI API key in .env
    if not os.getenv('OPENAI_API_KEY'):
        print("Please set your OpenAI API key in the .env file")
        exit(1)
    
    # Create and run the mirror
    mirror = RoastingMirror()
    mirror.run()