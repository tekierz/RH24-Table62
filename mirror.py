import cv2 # type: ignore
import numpy as np # type: ignore
import openai
import os
import base64
import time
import threading
import queue
import torch # type: ignore
from datetime import datetime
from dotenv import load_dotenv
import sys
import warnings
import shutil
import asyncio
from discord_bot import bot, send_image
from prompt_manager import PromptManager
import subprocess
import argparse
from openai import OpenAI
import requests

class RoastingMirror:
    """
    A smart mirror application that detects people using YOLOv5-tiny
    and provides AI-generated fashion critiques via the OpenAI API.
    
    This class implements a computer vision system that uses a webcam to detect people,
    captures their image, and generates both text and audio feedback about their appearance
    using OpenAI's GPT-4 Vision and text-to-speech capabilities.
    
    Attributes:
        client (openai.OpenAI): OpenAI client instance for API interactions
        tts_engine (pyttsx3.Engine): Text-to-speech engine for audio output
        camera (cv2.VideoCapture): Webcam capture device
        model (torch.hub.load): YOLOv5 model for person detection
        last_roast_time (float): Timestamp of the last generated roast
        roast_cooldown (int): Minimum time (seconds) between roasts
    """

    def __init__(self, use_lambda=False, horizontal_mode=False):
        """
        Initialize the RoastingMirror with all necessary components
        
        Args:
            use_lambda (bool): Whether to use Lambda Labs API instead of OpenAI
            horizontal_mode (bool): Whether to use horizontal (landscape) orientation
        """
        print("[Init] Starting initialization...")
        
        # Store API choice
        self.use_lambda = use_lambda
        
        # Store orientation mode
        self.horizontal_mode = horizontal_mode
        
        # Initialize OpenAI or Lambda Labs client
        if self.use_lambda:
            self.client = OpenAI(
                api_key=os.getenv('LAMBDA_API_KEY'),
                base_url="https://api.lambdalabs.com/v1"
            )
            self.vision_model = "llama3.2-11b-vision-instruct"
        else:
            self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            self.vision_model = "gpt-4o-mini"  # Using shorter model name
        
        # Initialize prompt manager and current prompt style
        self.prompt_manager = PromptManager()
        self.current_prompt_style = 1  # Default to kind, child-friendly style
        
        # Start Discord bot in a separate thread
        print("[Discord] Creating Discord bot thread...")
        self.discord_thread = threading.Thread(target=self._run_discord_bot)
        self.discord_thread.daemon = True
        self.discord_thread.start()
        print("[Discord] Discord thread started")
        
        # Wait a moment for Discord bot to initialize
        time.sleep(2)
        print("[Discord] Waited for initialization")
        
        # Load environment variables
        load_dotenv()
        
        # Suppress CUDA deprecation warnings since we're on CPU anyway
        warnings.filterwarnings('ignore', category=FutureWarning)
        
        # Define local model directory
        self.model_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize camera
        self.camera = cv2.VideoCapture(0)
        
        # Load YOLOv5 model
        try:
            # On Mac, we'll always use CPU
            self.device = 'cpu'
            
            # Check if model already exists locally
            model_path = os.path.join(self.model_dir, 'yolov5s.pt')
            try:
                print("\nLoading YOLOv5 model from local storage...")
                self.model = torch.hub.load('ultralytics/yolov5:v6.0', 'custom', 
                                          path=model_path, force_reload=True)
            except Exception as local_error:
                print(f"\nFailed to load local model: {str(local_error)}")
                print("\nDownloading fresh YOLOv5 model...")
                self.model = torch.hub.load('ultralytics/yolov5:v6.0', 'yolov5s', 
                                          pretrained=True, force_reload=True)
                # Save model for future use
                torch.save(self.model.state_dict(), model_path)
            
            self.model.to(self.device)
            print(f"Successfully loaded YOLOv5 model on CPU!")
        except Exception as e:
            print(f"\nError loading YOLOv5 model: {str(e)}")
            sys.exit(1)
        
        # Roast cooldown and tracking settings
        self.last_roast_time = 0
        self.roast_cooldown = 10  # in seconds
        self.person_present = False
        self.frame_center = None
        self.consecutive_empty_frames = 0
        self.consecutive_frames_threshold = 30  # about 1 second at 30 fps
        
        # Add audio management attributes
        self.current_audio_process = None
        self.audio_lock = threading.Lock()
        
        # Directory for saving audio files
        if not os.path.exists("sounds"):
            os.makedirs("sounds")
        # Create a temporary directory for active audio
        self.active_audio_dir = os.path.join("sounds", "active")
        if os.path.exists(self.active_audio_dir):
            shutil.rmtree(self.active_audio_dir)
        os.makedirs(self.active_audio_dir)
        
        # Threading / concurrency management
        self.roast_queue = queue.Queue()
        self.roast_in_progress = False
        self.roast_thread = None
        
        # YOLO detection parameters
        self.confidence_threshold = 0.45  # Default confidence threshold
        self.center_region_scale = 0.33  # Default center region size (1/3 of frame)
        
        # Set model parameters
        self.model.conf = self.confidence_threshold  # confidence threshold
        self.model.classes = [0]  # only detect people (class 0 in COCO dataset)
        
        # Add roast completion tracking
        self.roast_completed = True  # Track if current roast has finished
        self.skip_current_roast = False  # Flag to skip current roast

    def _run_discord_bot(self):
        """
        Run the Discord bot in a separate thread
        """
        try:
            TOKEN = os.getenv('DISCORD_TOKEN')
            if not TOKEN:
                print("Error: DISCORD_TOKEN not found in .env file")
            print("[Discord] Attempting to start Discord bot...")
            print(f"[Discord] Using token: {TOKEN[:5]}...{TOKEN[-5:]}")  # Show first/last 5 chars safely
            print(f"[Discord] Bot object status: {bot}")
            asyncio.run(bot.start(TOKEN))
        except Exception as e:
            print(f"[Discord] Error starting Discord bot: {str(e)}")
            print(f"[Discord] Full error details: {repr(e)}")

    def adjust_confidence(self, delta):
        """
        Adjust the confidence threshold for YOLO detection
        
        Args:
            delta (float): Amount to adjust confidence by (positive or negative)
        """
        self.confidence_threshold = max(0.1, min(0.9, self.confidence_threshold + delta))
        self.model.conf = self.confidence_threshold
        print(f"\nConfidence threshold adjusted to: {self.confidence_threshold:.2f}")

    def adjust_center_region(self, delta):
        """
        Adjust the size of the center detection region
        
        Args:
            delta (float): Amount to adjust region scale by (positive or negative)
        """
        self.center_region_scale = max(0.1, min(0.9, self.center_region_scale + delta))
        print(f"\nCenter region scale adjusted to: {self.center_region_scale:.2f}")

    def detect_person(self, frame):
        """
        Detect if any person is present in the frame using YOLOv5.
        Also tracks if the person is in the center and if they've left the frame.
        Checks if the person is in the foreground based on bounding box size.
        
        Args:
            frame (numpy.ndarray): The input image from camera
        
        Returns:
            bool: True if a new person is detected in the center foreground, False otherwise
        """
        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]
        center_x = frame_width // 2
        center_y = frame_height // 2
        
        # Define center region using scale parameter
        center_region_width = int(frame_width * self.center_region_scale)
        center_region_height = int(frame_height * self.center_region_scale)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform detection without autocast since we're on CPU
        results = self.model(rgb_frame)
        
        person_in_center = False
        any_person = False
        
        # Process detections
        detections = results.xyxy[0].cpu().numpy()
        
        # Minimum size threshold for foreground detection (percentage of frame)
        min_size_threshold = 0.15  # Person must occupy at least 15% of frame height
        
        for detection in detections:
            if detection[5] == 0:  # class 0 is person in COCO dataset
                any_person = True
                confidence = detection[4]
                if confidence > self.model.conf:
                    # Draw bounding box
                    box = detection[:4].astype(int)
                    
                    # Calculate person height relative to frame height
                    person_height = box[3] - box[1]
                    height_ratio = person_height / frame_height
                    
                    # Calculate person center
                    person_center_x = (box[0] + box[2]) // 2
                    person_center_y = (box[1] + box[3]) // 2
                    
                    # Check if person is in center region and foreground
                    in_center_x = abs(person_center_x - center_x) < (center_region_width // 2)
                    in_center_y = abs(person_center_y - center_y) < (center_region_height // 2)
                    in_foreground = height_ratio > min_size_threshold
                    
                    if in_center_x and in_center_y and in_foreground:
                        person_in_center = True
                        cv2.rectangle(frame, 
                                    (box[0], box[1]), 
                                    (box[2], box[3]), 
                                    (0, 255, 0), 2)
                        # Add foreground indicator
                        cv2.putText(frame, f"Foreground: {height_ratio:.2f}", 
                                  (box[0], box[1] - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        cv2.rectangle(frame, 
                                    (box[0], box[1]), 
                                    (box[2], box[3]), 
                                    (255, 165, 0), 2)  # Orange for non-center/background people
        
        # Draw center region
        cv2.rectangle(frame,
                     (center_x - center_region_width // 2, center_y - center_region_height // 2),
                     (center_x + center_region_width // 2, center_y + center_region_height // 2),
                     (255, 255, 255), 1)
        
        # Update tracking logic
        if not any_person:
            self.consecutive_empty_frames += 1
        else:
            self.consecutive_empty_frames = 0
        
        # Reset person_present if the frame has been empty for enough consecutive frames
        if self.consecutive_empty_frames >= self.consecutive_frames_threshold:
            self.person_present = False
            self.consecutive_empty_frames = 0
        
        # Determine if this is a new person to roast
        should_roast = person_in_center and not self.person_present
        if person_in_center:
            self.person_present = True
        
        # Add status overlay
        status_text = "Ready for new person" if not self.person_present else "Waiting for person to leave"
        cv2.putText(frame, status_text, (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return should_roast

    def _clear_audio(self):
        """
        Stop current audio playback and clear the buffer
        """
        with self.audio_lock:
            if self.current_audio_process:
                if sys.platform == "darwin":  # macOS
                    os.system("pkill afplay")
                elif sys.platform == "win32":  # Windows
                    os.system("taskkill /F /IM wmplayer.exe >NUL 2>&1")
                else:  # Linux
                    os.system("pkill -f xdg-open")
                self.current_audio_process = None
            
            # Clear the active audio directory
            for file in os.listdir(self.active_audio_dir):
                os.remove(os.path.join(self.active_audio_dir, file))

    def generate_and_play_audio(self, text):
        """
        Generate and play audio for roast text using OpenAI's text-to-speech (audio preview).
        
        Args:
            text (str): The roast text to be converted to speech
        """
        try:
            # Clear any existing audio first
            self._clear_audio()

            with self.audio_lock:
                # Note: Currently the API only supports MP3 format
                completion = self.client.chat.completions.create(
                    model="gpt-4o-mini-audio-preview",
                    modalities=["text", "audio"],
                    audio={
                        "voice": "fable",
                        # "format": "mp3",  # Currently only MP3 is supported
                        "format": "pcm16", # Future support for streaming PCM
                        # "sample_rate": 24000  # Future support for sample rate
                    },
                    messages=[
                        {
                            "role": "system",
                            "content": """You are a brutally honest Americanfashion reality TV judge 
                                with a over dramatic fake British accent. Be theatrical and flamboyant!"""
                        },
                        {
                            "role": "user",
                            "content": text,
                        }
                    ],
                )
                
                # Decode PCM16 data and convert to WAV format
                pcm_bytes = base64.b64decode(completion.choices[0].message.audio.data)
                
                # Save to active audio directory instead
                speech_file_path = os.path.join(self.active_audio_dir, f"roast_{int(time.time())}.wav")
                
                import wave
                with wave.open(speech_file_path, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono audio
                    wav_file.setsampwidth(2)  # 2 bytes per sample for PCM16
                    wav_file.setframerate(24000)  # Sample rate
                    wav_file.writeframes(pcm_bytes)
                
                # Play audio with process tracking
                if sys.platform == "darwin":  # macOS
                    self.current_audio_process = subprocess.Popen(['afplay', speech_file_path])
                elif sys.platform == "win32":  # Windows
                    self.current_audio_process = subprocess.Popen(['start', speech_file_path], shell=True)
                else:  # Linux
                    self.current_audio_process = subprocess.Popen(['xdg-open', speech_file_path])
            
        except Exception as e:
            print(f"Error generating or playing audio: {str(e)}")
    
    async def _roast_worker(self, image_data):
        """
        Background worker function to handle GPT-based roast generation 
        and then call the audio generation method.
        
        Args:
            image_data (bytes): base64-encoded, in-memory representation of the frame
        """
        try:
            self.roast_completed = False
            print("\n🎭 Generating fashion critique...\n")
            
            # Get appropriate prompts based on current style
            system_prompt = getattr(
                self.prompt_manager, 
                f'get_vision_system_prompt_{self.current_prompt_style}'
            )()
            user_prompt = getattr(
                self.prompt_manager, 
                f'get_vision_user_prompt_{self.current_prompt_style}'
            )()
            
            # Save debug image
            try:
                debug_image_path = f"debug_capture_{int(time.time())}.jpg"
                print(f"[Debug] Attempting to save debug image to: {debug_image_path}")
                image_bytes = base64.b64decode(image_data)
                with open(debug_image_path, "wb") as f:
                    f.write(image_bytes)
                print(f"[Debug] Successfully saved debug image")
            except Exception as img_error:
                print(f"[Debug] Failed to save debug image: {str(img_error)}")

            # Create API request based on provider
            if self.use_lambda:
                lambda_api_url = os.getenv('LAMBDA_API_URL')
                if not lambda_api_url:
                    raise ValueError("LAMBDA_API_URL not found in environment variables")
                
                print(f"\n[Debug] Using Lambda API URL: {lambda_api_url}")
                
                # Format the messages for Lambda's vision model
                messages = [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_prompt
                            },
                            {
                                "type": "image",
                                "image": {
                                    "data": image_data
                                }
                            }
                        ]
                    }
                ]

                # Make Lambda API request
                try:
                    response = requests.post(
                        f"{lambda_api_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {os.getenv('LAMBDA_API_KEY')}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "messages": messages,
                            "model": self.vision_model,
                            "max_tokens": 500,
                            "temperature": 0.8
                        }
                    )
                    
                    print(f"\n[Debug] Lambda API Response Status: {response.status_code}")
                    print(f"[Debug] Lambda API Response Headers: {response.headers}")
                    
                    # Check if the request was successful
                    response.raise_for_status()
                    
                    # Parse the JSON response
                    response_data = response.json()
                    print(f"[Debug] Lambda API Response Data: {response_data}")
                    
                    if response_data is None:
                        raise ValueError("Received empty response from Lambda API")
                        
                    if 'choices' not in response_data:
                        raise ValueError(f"Missing 'choices' in response: {response_data}")
                        
                    if not response_data['choices']:
                        raise ValueError("Empty choices array in response")
                        
                    roast_text = response_data['choices'][0]['message']['content']
                    
                except requests.exceptions.RequestException as e:
                    raise ValueError(f"Lambda API request failed: {str(e)}")
                except (KeyError, IndexError) as e:
                    raise ValueError(f"Failed to parse Lambda API response: {str(e)}")
                
            else:
                # Original OpenAI implementation
                response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}", "detail": "high"}}
                        ]
                    }
                ],
                max_tokens=500,
                temperature=0.6

            )
            
            roast_text = response.choices[0].message.content
            
            # Print the roast with some formatting
            print("\n📺 Fashion Judge Says:")
            print("=" * 50)
            print(roast_text)
            print("=" * 50 + "\n")

            # Send to Discord and get upload status
            upload_success = await send_image("•☽────✧˖°˖☆˖°˖✧────☾•" "\n" + roast_text + "\n" + "⬇️ ⬇️ ⬇️", debug_image_path)

            # Delete the debug image if upload was successful
            if upload_success and os.path.exists(debug_image_path):
                try:
                    os.remove(debug_image_path)
                    print(f"[Debug] Successfully deleted debug image: {debug_image_path}")
                except Exception as del_error:
                    print(f"[Debug] Failed to delete debug image: {str(del_error)}")
            
            # Generate and play audio for the roast
            self.generate_and_play_audio(roast_text)
            
            # Wait for audio playback to complete or skip
            while self.current_audio_process and self.current_audio_process.poll() is None:
                if self.skip_current_roast:
                    self._clear_audio()
                    self.skip_current_roast = False
                    break
                time.sleep(0.1)
            
            self.roast_completed = True  # Mark as completed when done
            return roast_text
        except Exception as e:
            error_message = f"Error generating roast: {str(e)}"
            print(f"\n❌ {error_message}\n")
            self.roast_completed = True  # Mark as completed even on error
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
        
        # Create and run the async function in the background
        async def run_async_worker():
            await self._roast_worker(image_data)
        
        self.roast_thread = threading.Thread(
            target=lambda: asyncio.run(run_async_worker()),
            args=()
        )
        self.roast_thread.daemon = True
        self.roast_thread.start()
    
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
        print("Starting Miragé - The Roasting Smart Mirror (YOLOv5 Edition)")
        print("Press 'q' to quit")
        print(f"\nOrientation: {'Horizontal' if self.horizontal_mode else 'Vertical'}")
        print("\nDetection Controls:")
        print("[ and ] - Adjust confidence threshold (currently: {:.2f})".format(self.confidence_threshold))
        print("- and + - Adjust center region size (currently: {:.2f})".format(self.center_region_scale))
        print("\nStyle Controls:")
        print("1-5 to switch between different critic styles:")
        print("1: Kind & Child-Friendly")
        print("2: Professional & Balanced")
        print("3: Weather-Aware")
        print("4: Ultra-Critical Expert")
        print("5: Savage Roast Master")
        print("\nManual Control:")
        print("SPACE - Force trigger next roast")
        print("BACKSPACE - Skip current roast")
        
        while True:
            ret, frame = self.camera.read()
            if not ret:
                print("Camera frame capture failed.")
                break
            
            # Only rotate if we're in vertical mode
            if not self.horizontal_mode:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            
            # Flip horizontally for mirror effect
            mirror_frame = cv2.flip(frame, 1)
            
            # Detect person (store result but don't use it directly)
            _ = self.detect_person(mirror_frame)
            
            # Display current critic style
            style_names = {
                1: "Kind & Child-Friendly",
                2: "Professional & Balanced",
                3: "Weather-Aware",
                4: "Ultra-Critical Expert",
                5: "Savage Roast Master"
            }
            cv2.putText(mirror_frame, f"Style: {style_names[self.current_prompt_style]}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display detection parameters
            cv2.putText(mirror_frame, f"Conf: {self.confidence_threshold:.2f}", 
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(mirror_frame, f"Region: {self.center_region_scale:.2f}", 
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Check cooldown and roast availability
            current_time = time.time()
            if (current_time - self.last_roast_time >= self.roast_cooldown
                and not self.roast_in_progress
                and self.roast_completed):
                
                print("Person detected! Generating roast asynchronously...")
                self._start_roast_generation(frame)  # send the unflipped frame
                self.last_roast_time = current_time
            
            # Check if any new roasts are returned by the background thread
            while not self.roast_queue.empty():
                roast_text = self.roast_queue.get()
                print(f"Roast text from background: {roast_text}")
            
            # Display cooldown timer
            time_remaining = max(0, self.roast_cooldown - (current_time - self.last_roast_time))
            cv2.putText(mirror_frame, f"Next roast in: {int(time_remaining)}s", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show the resulting frame
            cv2.imshow("Miragé (YOLOv5)", mirror_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space bar press
                if (current_time - self.last_roast_time >= self.roast_cooldown 
                    and not self.roast_in_progress 
                    and self.roast_completed):  # Only trigger if previous roast completed
                    print("Manual trigger activated! Generating roast...")
                    self._start_roast_generation(frame)
                    self.last_roast_time = current_time
                else:
                    if not self.roast_completed:
                        print("Please wait for current roast to complete")
                    elif not current_time - self.last_roast_time >= self.roast_cooldown:
                        print("Please wait for cooldown to finish")
            elif key == 8:  # Backspace key
                if not self.roast_completed:
                    print("Skipping current roast...")
                    self.skip_current_roast = True
                    self.roast_completed = True  # Mark as completed when skipping
            elif key == ord('['):
                self.adjust_confidence(-0.05)
            elif key == ord(']'):
                self.adjust_confidence(0.05)
            elif key == ord('-'):
                self.adjust_center_region(-0.05)
            elif key == ord('=') or key == ord('+'):  # Both - and = keys work
                self.adjust_center_region(0.05)
            elif ord('1') <= key <= ord('5'):
                self.current_prompt_style = key - ord('0')
                print(f"\nSwitched to style: {style_names[self.current_prompt_style]}")
            elif key == ord('c'):
                self._clear_audio()
                print("\nCleared audio playback")
        
            # Check if current roast is completed
            if self.roast_completed:
                print("Current roast completed. Waiting for next roast.")
        
        # Cleanup on exit
        self._clear_audio()  # Clear audio before closing
        self.camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Add argument parser
    parser = argparse.ArgumentParser(description='Miragé - The Roasting Smart Mirror')
    parser.add_argument('--lambda', action='store_true', help='Use Lambda Labs API instead of OpenAI')
    parser.add_argument('--hori', action='store_true', help='Use horizontal (landscape) orientation')
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Verify all required environment variables
    required_vars = {
        'OPENAI_API_KEY': 'OpenAI API key',
        'DISCORD_TOKEN': 'Discord bot token',
        'DISCORD_CHANNEL_ID': 'Discord channel ID'
    }
    
    # Add Lambda Labs API key requirement if --lambda is used
    if getattr(args, 'lambda'):
        required_vars['LAMBDA_API_KEY'] = 'Lambda Labs API key'
    
    missing_vars = []
    for var, name in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(name)
    
    if missing_vars:
        print("Missing required environment variables:")
        for var in missing_vars:
            print(f"- {var}")
        print(f"\nPlease add them to your .env file in: {os.getcwd()}")
        sys.exit(1)
    
    print("Starting Miragé with integrated Discord bot...")
    mirror = RoastingMirror(
        use_lambda=getattr(args, 'lambda'),
        horizontal_mode=getattr(args, 'hori')
    )
    mirror.run()    