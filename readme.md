# Miragé - The Smart Mirror

A Python-based smart mirror application that combines computer vision, artificial intelligence, and natural language processing to create an interactive experience. The system detects people in real-time, analyzes their appearance using a combination of YOLOv5, Llama 3.2 (via Lambda Labs), and OpenAI's GPT-4o-Mini, and delivers witty responses through speech synthesis, while also providing useful information like weather updates. 

## Core Technologies

### AI and Machine Learning
- **OpenAI GPT-4 Vision**: Powers image analysis and generates contextual responses
- **OpenCV**: Handles real-time computer vision and person detection
- **YOLOv5**: Provides real-time person detection with high accuracy
- **PyTorch**: Supports deep learning models for advanced computer vision tasks

### Integration & Services
- **Discord Bot**: Remote control, monitoring, and image sharing capabilities
- **OpenWeather API**: Real-time weather information and condition-aware fashion advice
- **Text-to-Speech**: Voice output using platform-specific engines

## Features

### Person Detection
- Real-time person tracking using YOLOv5
- Adjustable confidence threshold (using '[' and ']' keys)
- Configurable center detection region (using '-' and '+' keys)
- Automatic person presence tracking with cooldown timer

### Fashion Analysis Styles
Five different critic personalities available (switch using number keys 1-5):
1. Kind & Child-Friendly: Supportive and encouraging feedback
2. Professional & Balanced: Constructive professional styling advice
3. Weather-Aware: Context-aware fashion suggestions based on current weather
4. Ultra-Critical Expert: High-standard professional critique
5. Savage Roast Master: Dramatic and theatrical fashion commentary

### Controls
- **SPACE**: Force trigger next roast
- **BACKSPACE**: Skip current roast
- **Q**: Quit application
- **C**: Clear audio playback
- **1-5**: Switch between critic styles
- **[ ]**: Adjust detection confidence
- **- +**: Adjust center region size

### Discord Integration
- Automatic sharing of fashion critiques with images
- Remote monitoring capabilities
- Dedicated channel for mirror interactions

### Weather Integration
- Real-time weather condition monitoring
- Weather-aware fashion advice
- Temperature and condition-specific recommendations

## Project Structure
- `mirror.py`: Main application entry point and core logic
- `discord_bot.py`: Discord integration for remote monitoring
- `weather_service.py`: OpenWeather API integration
- `prompt_manager.py`: AI prompt management and personality systems

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is open-sourced under the MIT License - as per the rules of the MIT Reality Hack rules. For more information, please check out the [MIT Reality Hack Rules](https://mitrealityhack.com/rules/).

## Acknowledgments
- OpenAI for GPT-4 Vision API
- OpenCV community
- Discord.py developers
- OpenWeather API team
- Ultralytics for YOLOv5
- Lambda Labs for Lambda API + Free GPU credits

## Requirements

### Hardware
- Webcam or camera input device (we are using the Qualcomm RB3 Gen 2 Developer Kit (Vision Kit)
- Display monitor
- Microphone (optional, for voice commands)
- Speakers or audio output device
- This project is built to run on lightweight edge compute devices, such as the Qualcomm RB3 Gen 2 Developer Kit (Vision Kit), Raspberry Pi, Mac Mini, and more. 

### Software
- Python (Tested working on version 3.11)
- Operating system: Windows 10+, macOS 10.15+, or Linux (Ubuntu 20.04+ recommended)

### API Keys
- OpenAI API key (for GPT-4 Vision)
- Discord Bot Token (optional, for remote control)
- OpenWeather API key (for weather services)

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/mirage-smart-mirror.git
cd mirage-smart-mirror
```

### Step 2: Set Up Python Environment

#### Windows
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

#### macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables
1. Copy the `.env_template` file to `.env`
2. Fill in your API keys and configuration:
   ```
   OPENAI_API_KEY=your_api_key_here
   DISCORD_TOKEN=your_discord_token_here
   DISCORD_CHANNEL_ID=your_discord_channel_id_here
   OPEN_WEATHER_API_KEY=your_open_weather_api_key_here
   WEATHER_LAT=your_latitude_here
   WEATHER_LON=your_longitude_here
   ```

### Platform-Specific Setup

#### Windows
- Install the appropriate PyTorch version from pytorch.org

#### macOS
- Install Homebrew (if not already installed):
  ```bash
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  ```
- Install required system dependencies:
  ```bash
  brew install python cmake pkg-config
  ```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install -y python3-dev python3-pip
sudo apt-get install -y libopencv-dev python3-opencv
sudo apt-get install -y portaudio19-dev python3-pyaudio
```

## Running the Application

1. Activate the virtual environment (if not already activated)
2. Start the main application:
   ```bash
   python mirror.py
   ```