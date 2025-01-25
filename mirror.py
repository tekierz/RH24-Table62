import cv2
import numpy as np
import openai
import pyttsx3
import os
import base64
import time
import threading
import queue
from datetime import datetime
from dotenv import load_dotenv
import sys

class RoastingMirror:
    """
    A smart mirror application that detects people using a MobileNet-SSD DNN 
    and provides AI-generated fashion critiques via the OpenAI API.
    
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
        Initialize the RoastingMirror with all necessary components:
         - OpenAI client
         - Text-to-speech engine
         - Camera initialization
         - DNN-based person detection model
         - Queues and threading logic for asynchronous roast generation
        """
        # Load environment variables
        load_dotenv()
        
        # Initialize OpenAI client using .env file
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Initialize TTS engine (pyttsx3)
        self.tts_engine = pyttsx3.init()
        
        # Initialize camera
        self.camera = cv2.VideoCapture(0)
        
        # ----------------------------------------------------------------------------
        # DNN-based person detection setup
        # ----------------------------------------------------------------------------
        model_proto = "MobileNetSSD_deploy.prototxt"
        model_weights = "MobileNetSSD_deploy.caffemodel"
        
        # Check if model files exist
        if not (os.path.exists(model_proto) and os.path.exists(model_weights)):
            print("\nDownloading required model files...")
            # URLs for the model files
            proto_url = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.prototxt"
            weights_url = "https://drive.google.com/uc?export=download&id=0B3gersZ2cHIxRm5PMWRoTkdHdHc"
            
            try:
                import urllib.request
                import gdown
                
                # Download prototxt file
                urllib.request.urlretrieve(proto_url, model_proto)
                print(f"Downloaded {model_proto}")
                
                # Download caffemodel file using gdown (handles Google Drive links)
                gdown.download(weights_url, model_weights, quiet=False)
                print(f"Downloaded {model_weights}")
            except Exception as e:
                print(f"\nError downloading model files: {str(e)}")
                print("\nPlease manually download the following files and place them in the project directory:")
                print(f"1. {model_proto} from: {proto_url}")
                print(f"2. {model_weights} from: {weights_url}")
                sys.exit(1)
        
        # Load the DNN
        try:
            self.net = cv2.dnn.readNetFromCaffe(model_proto, model_weights)
            print("\nSuccessfully loaded MobileNet-SSD model!")
        except Exception as e:
            print(f"\nError loading model: {str(e)}")
            sys.exit(1)
        
        # Class indices for this model; 15 is 'person' in MobileNetSSD
        self.PERSON_CLASS_ID = 15
        
        # Roast cooldown settings
        self.last_roast_time = 0
        self.roast_cooldown = 10  # in seconds
        
        # Directory for saving audio files
        if not os.path.exists("sounds"):
            os.makedirs("sounds")
        
        # Threading / concurrency management
        self.roast_queue = queue.Queue()    # to receive roasted text
        self.roast_in_progress = False      # lock for background tasks
        self.roast_thread = None            # reference to background thread

    def detect_person(self, frame):
        """
        Detect if any person is present in the frame using the MobileNet-SSD DNN.
        
        Args:
            frame (numpy.ndarray): The input image from camera
        
        Returns:
            bool: True if at least one person is detected, False otherwise
        """
        # Prepare the frame for DNN: resize, convert to blob, etc.
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        
        # Perform the forward pass
        detections = self.net.forward()
        
        person_detected = False
        
        # Typically the output shape is [1, 1, N, 7], where N is number of detections
        # Each detection has [batchId, classId, confidence, left, top, right, bottom]
        h, w = frame.shape[:2]
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]  # Confidence of detection
            class_id = int(detections[0, 0, i, 1])
            
            if class_id == self.PERSON_CLASS_ID and confidence > 0.4:
                # Found a person
                person_detected = True
                box_left = int(detections[0, 0, i, 3] * w)
                box_top = int(detections[0, 0, i, 4] * h)
                box_right = int(detections[0, 0, i, 5] * w)
                box_bottom = int(detections[0, 0, i, 6] * h)
                
                # Draw bounding box
                cv2.rectangle(frame, (box_left, box_top), (box_right, box_bottom), (0, 255, 0), 2)
        
        # You can add text overlay if person_detected is True
        if person_detected:
            cv2.putText(frame, "Person Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2)
        
        return person_detected

    def generate_and_play_audio(self, text):
        """
        Generate and play audio for roast text using OpenAI's text-to-speech (audio preview).
        
        Args:
            text (str): The roast text to be converted to speech
        """
        try:
            completion = self.client.chat.completions.create(
                model="gpt-4o-audio-preview",
                modalities=["text", "audio"],
                audio={"voice": "fable", "format": "mp3"},
                messages=[
                    {
                        "role": "system",
                        "content": """You are a brutally honest fashion reality TV judge 
                            with a dramatic British accent. Be theatrical and flamboyant!"""
                    },
                    {
                        "role": "user",
                        "content": text,
                    }
                ],
            )
            
            # Decode and save as MP3
            speech_file_path = f"./sounds/roast_{int(time.time())}.mp3"
            mp3_bytes = base64.b64decode(completion.choices[0].message.audio.data)
            
            with open(speech_file_path, "wb") as f:
                f.write(mp3_bytes)
            
            # Play audio in a non-blocking fashion
            if sys.platform == "darwin":  # macOS
                os.system(f"afplay {speech_file_path} &")
            elif sys.platform == "win32":  # Windows
                os.system(f"start {speech_file_path}")
            else:  # Linux
                os.system(f"xdg-open {speech_file_path} &")
            
        except Exception as e:
            print(f"Error generating or playing audio: {str(e)}")
    
    def _roast_worker(self, image_data):
        """
        Background worker function to handle GPT-based roast generation 
        and then call the audio generation method.
        
        Args:
            image_data (bytes): base64-encoded, in-memory representation of the frame
        """
        try:
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
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
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
            print(f"\n❌ {error_message}\n")
        finally:
            self.roast_in_progress = False

    def _start_roast_generation(self, frame):
        """
        Start a separate thread to handle roast generation so the main loop doesn't freeze.
        
        Args:
            frame (numpy.ndarray): The current camera frame
        """
        ret, encoded_image = cv2.imencode(".jpg", frame)
        if not ret:
            print("Failed to encode image for roast generation.")
            return
        
        image_data = base64.b64encode(encoded_image).decode('utf-8')
        
        self.roast_thread = threading.Thread(
            target=self._roast_worker, 
            args=(image_data,)
        )
        self.roast_thread.daemon = True
        self.roast_thread.start()
    
    def speak_roast(self, roast_text):
        """
        Use pyttsx3 to speak the roast text. This can provide immediate TTS feedback 
        while the audio preview is being fetched or as a fallback.
        
        Args:
            roast_text (str): The text to read out loud
        """
        self.tts_engine.say(roast_text)
        self.tts_engine.runAndWait()
    
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
        print("Starting Miragé - The Roasting Smart Mirror (DNN Edition)")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = self.camera.read()
            if not ret:
                print("Camera frame capture failed.")
                break
            
            # Flip horizontally for a mirror effect
            mirror_frame = cv2.flip(frame, 1)
            
            # Detect person
            person_detected = self.detect_person(mirror_frame)
            
            # Check cooldown and roast availability
            current_time = time.time()
            if (person_detected 
                and current_time - self.last_roast_time >= self.roast_cooldown
                and not self.roast_in_progress):
                
                print("Person detected! Generating roast asynchronously...")
                self._start_roast_generation(frame)  # send the unflipped frame
                self.last_roast_time = current_time
            
            # Check if any new roasts are returned by the background thread
            while not self.roast_queue.empty():
                roast_text = self.roast_queue.get()
                print(f"Roast text from background: {roast_text}")
                # Optional: speak it aloud with pyttsx3
                self.speak_roast(roast_text)
            
            # Display cooldown timer
            time_remaining = max(0, self.roast_cooldown - (current_time - self.last_roast_time))
            cv2.putText(mirror_frame, f"Next roast in: {int(time_remaining)}s", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show the resulting frame
            cv2.imshow("Miragé (DNN)", mirror_frame)
            
            # Break out if user hits 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup on exit
        self.camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    """
    Entry point for the RoastingMirror application using MobileNet-SSD for person detection.
    """
    load_dotenv()
    
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print(f"API key found: {api_key[:7]}...")
    else:
        print("No API key found in .env file, please add it.")
        print(f"Current directory: {os.getcwd()}")
        print(f".env file exists: {os.path.exists('.env')}")
    
    mirror = RoastingMirror()
    mirror.run()    