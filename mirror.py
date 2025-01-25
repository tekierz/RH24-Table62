import cv2
import numpy as np
import openai
import pyttsx3
import os
import base64
import time
from datetime import datetime
from dotenv import load_dotenv
import sys

class RoastingMirror:
    """
    A smart mirror application that detects people and provides AI-generated fashion critiques.
    
    This class implements a computer vision system that uses a webcam to detect people,
    captures their image, and generates both text and audio feedback about their appearance
    using OpenAI's GPT-4 Vision and text-to-speech capabilities.
    
    Attributes:
        client (openai.OpenAI): OpenAI client instance for API interactions
        tts_engine (pyttsx3.Engine): Text-to-speech engine for audio output
        camera (cv2.VideoCapture): Webcam capture device
        person_detector (cv2.HOGDescriptor): HOG-based person detection model
        last_roast_time (float): Timestamp of the last generated roast
        roast_cooldown (int): Minimum time (seconds) between roasts
    """

    def __init__(self):
        """
        Initialize the RoastingMirror with all necessary components.
        
        Sets up OpenAI client, text-to-speech engine, camera, and person detection model.
        Creates required directories and initializes cooldown timing system.
        """
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
        
        # Create sounds directory if it doesn't exist
        if not os.path.exists("sounds"):
            os.makedirs("sounds")
        
    def detect_person(self, frame):
        """
        Detect if any persons are present in the given frame.
        
        Uses HOG (Histogram of Oriented Gradients) descriptor with SVM classifier
        to detect human figures in the image. Draws bounding boxes around detected
        persons and displays count.
        
        Args:
            frame (numpy.ndarray): Input image frame from camera
            
        Returns:
            bool: True if at least one person is detected, False otherwise
        """
        boxes, weights = self.person_detector.detectMultiScale(frame, winStride=(8, 8))
        
        # Add visual feedback by drawing rectangles around detected people
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Add text indicator for person detection
        if len(boxes) > 0:
            cv2.putText(frame, f"Persons detected: {len(boxes)}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                        (0, 255, 0), 2)
        
        return len(boxes) > 0
    
    def save_frame(self, frame):
        """
        Save the current camera frame as a temporary image file.
        
        Args:
            frame (numpy.ndarray): Input image frame to save
            
        Returns:
            str: Path to the saved temporary image file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"temp_frame_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        return filename
    
    def generate_and_play_audio(self, text):
        """
        Generate and play audio for the given text using OpenAI's text-to-speech.
        
        Converts the roast text into audio with specific voice characteristics
        (British accent, dramatic flair) and plays it through the system's
        default audio player.
        
        Args:
            text (str): The text to convert to speech
            
        Raises:
            Exception: If audio generation or playback fails
        """
        try:
            # Generate audio with specific voice characteristics
            completion = self.client.chat.completions.create(
                model="gpt-4o-audio-preview",
                modalities=["text", "audio"],
                audio={"voice": "fable", "format": "mp3"},
                messages=[
                    {
                        "role": "system",
                        "content": """You are a brutally honest fashion reality TV judge with an exaggerated British accent. 
                        Think Simon Cowell meets RuPaul. Speak with dramatic flair, elongate your vowels, 
                        and be absolutely theatrical in your delivery. Use British phrases like 'darling', 
                        'sweetie', and 'oh my word'. Make dramatic pauses for effect."""
                    },
                    {
                        "role": "user",
                        "content": text,
                    }
                ],
            )
            
            # Save and play the audio
            speech_file_path = f"./sounds/roast_{int(time.time())}.mp3"
            mp3_bytes = base64.b64decode(completion.choices[0].message.audio.data)
            with open(speech_file_path, "wb") as f:
                f.write(mp3_bytes)
            
            # Play the audio using system default player
            if sys.platform == "darwin":  # macOS
                os.system(f"afplay {speech_file_path}")
            elif sys.platform == "win32":  # Windows
                os.system(f"start {speech_file_path}")
            else:  # Linux
                os.system(f"xdg-open {speech_file_path}")
            
            # Clean up the audio file after playing
            time.sleep(5)  # Wait for audio to finish playing
            os.remove(speech_file_path)
            
        except Exception as e:
            print(f"Error generating or playing audio: {str(e)}")
    
    def generate_roast(self, image_path):
        """
        Generate a fashion critique using GPT-4 Vision API.
        
        Analyzes the provided image and generates a contextual fashion critique,
        taking into account specific markers like lanyard colors to determine
        the tone of the feedback.
        
        Args:
            image_path (str): Path to the image file to analyze
            
        Returns:
            str: Generated fashion critique text
            
        Raises:
            Exception: If roast generation fails
        """
        try:
            with open(image_path, "rb") as image_file:
                print("\n🎭 Generating fashion critique...\n")  # Added visual indicator
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": """You are a brutally honest fashion reality TV judge. 
                            Focus your roasts on fashion choices, styling, and presentation. 
                            Use phrases like 'darling', 'honey', and British fashion terminology. 
                            Keep it dramatic but not cruel. If the person is wearing a black lanyard, roast them more harshly. If the person is wearing a yellow, pink or green lanyard, compliment them on physical appearance and personality. Be gushing and dramatic in your delivery."""
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Judge this person's fashion choices and appearance like you're on a reality TV show. If the person is wearing a black lanyard, roast them more harshly. If the person is wearing a yellow, pink or green lanyard, compliment them on physical appearance and personality. Be gushing and dramatic in your delivery."},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(image_file.read()).decode()}"}}
                            ]
                        }
                    ],
                    max_tokens=100
                )
                roast_text = response.choices[0].message.content
                
                # Print the roast with some formatting
                print("\n📺 Fashion Judge Says:")
                print("=" * 50)
                print(roast_text)
                print("=" * 50 + "\n")
                
                # Generate and play audio for the roast
                self.generate_and_play_audio(roast_text)
                return roast_text
        except Exception as e:
            error_message = f"Error generating roast: {str(e)}"
            print(f"\n❌ {error_message}\n")  # Added error printing
            return error_message
    
    def speak_roast(self, roast_text):
        """
        Convert the roast text to speech and play it.
        
        Args:
            roast_text (str): The text to be spoken
        """
        self.tts_engine.say(roast_text)
        self.tts_engine.runAndWait()
    
    def cleanup(self, image_path):
        """
        Remove temporary files created during the roast generation process.
        
        Args:
            image_path (str): Path to the temporary image file to remove
        """
        if os.path.exists(image_path):
            os.remove(image_path)
    
    def run(self):
        """
        Main application loop for the RoastingMirror.
        
        Continuously captures frames from the camera, detects persons,
        and generates roasts when appropriate. Handles the following:
        - Camera frame capture and mirror display
        - Person detection
        - Cooldown timer management
        - Roast generation and playback
        - User interface elements
        - Cleanup on exit
        
        The loop continues until the user presses 'q' to quit.
        """
        print("Starting Miragé - The Roasting Smart Mirror")
        print("Press 'q' to quit")
        
        while True:
            # Capture frame from camera
            ret, frame = self.camera.read()
            if not ret:
                print("Error capturing frame")
                break
            
            # Show mirror image (flipped horizontally)
            mirror_frame = cv2.flip(frame, 1)
            
            # Check for person in frame
            person_detected = self.detect_person(mirror_frame)
            
            # Check if it's time for a new roast
            current_time = time.time()
            if current_time - self.last_roast_time >= self.roast_cooldown:
                if person_detected:
                    print("Person detected - generating roast...")
                    # Save frame and generate roast
                    image_path = self.save_frame(frame)
                    try:
                        roast = self.generate_roast(image_path)
                        print(f"Generated roast: {roast}")
                        
                        # Speak the roast
                        self.speak_roast(roast)
                        
                        # Clean up
                        self.cleanup(image_path)
                        
                        # Update last roast time
                        self.last_roast_time = current_time
                    except Exception as e:
                        print(f"Error during roast generation: {e}")
            
            # Add cooldown timer indicator
            time_remaining = max(0, self.roast_cooldown - (current_time - self.last_roast_time))
            cv2.putText(mirror_frame, f"Next roast in: {int(time_remaining)}s", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                        (0, 255, 0), 2)
            
            cv2.imshow('Miragé', mirror_frame)
            
            # Check for quit command (q key)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    """
    Main entry point for the RoastingMirror application.
    
    Performs the following setup steps:
    1. Loads environment variables from .env file
    2. Verifies OpenAI API key presence and validity
    3. Initializes and runs the RoastingMirror instance
    
    Environment Variables Required:
        OPENAI_API_KEY: Valid OpenAI API key for GPT-4 Vision and audio services
    """
    # Load environment variables
    load_dotenv()
    
    # Debug: Print the API key (first few characters)
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print(f"API key found: {api_key[:7]}...")
    else:
        print("No API key found in .env file")
        print(f"Current working directory: {os.getcwd()}")
        print(f".env file exists: {os.path.exists('.env')}")
    
    # Check for OpenAI API key in .env
    # if not os.getenv('OPENAI_API_KEY'):
    #     print("Please set your OpenAI API key in the .env file")
    #     exit(1)
    
    # Create and run the mirror
    mirror = RoastingMirror()
    mirror.run()    