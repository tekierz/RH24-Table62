import os
import requests
from dotenv import load_dotenv
import json

class WeatherService:
    """
    Service to fetch current weather conditions from OpenWeatherMap API
    """
    def __init__(self):
        # Load environment variables
        load_dotenv()
        self.api_key = os.getenv('OPEN_WEATHER_API_KEY')
        self.lat = os.getenv('WEATHER_LAT')
        self.lon = os.getenv('WEATHER_LON')
        # Update to use the correct base URL for One Call API 3.0
        self.base_url = "https://api.openweathermap.org/data/3.0/onecall"
        
        # Print initialization details
        print(f"[Weather] Initializing WeatherService")
        print(f"[Weather] Using coordinates: LAT={self.lat}, LON={self.lon}")
        print(f"[Weather] API Key present: {bool(self.api_key)}")

    def get_current_weather(self):
        """
        Fetch current weather conditions for the configured location
        Returns formatted weather string or None if request fails
        """
        try:
            # Make API request with correct endpoint (removing /overview)
            params = {
                'lat': self.lat,
                'lon': self.lon,
                'appid': self.api_key,
                'units': 'metric',  # Use metric units
                'exclude': 'minutely,hourly,daily,alerts'  # Only get current weather
            }
            
            print(f"[Weather] Making API request to: {self.base_url}")
            print(f"[Weather] Using params: {json.dumps({k:v if k != 'appid' else '***' for k,v in params.items()}, indent=2)}")
            
            response = requests.get(self.base_url, params=params)
            print(f"[Weather] Response status code: {response.status_code}")
            
            response.raise_for_status()
            
            # Extract weather data from response
            data = response.json()
            
            # Format weather overview from current conditions
            if 'current' in data:
                current = data['current']
                weather_desc = current['weather'][0]['description'] if current.get('weather') else 'unknown conditions'
                temp = current.get('temp', 'N/A')
                feels_like = current.get('feels_like', 'N/A')
                humidity = current.get('humidity', 'N/A')
                
                weather_overview = (
                    f"Current weather: {weather_desc} with temperature of {temp}°C "
                    f"(feels like {feels_like}°C). Humidity: {humidity}%"
                )
            else:
                weather_overview = 'Weather data unavailable'
                
            print(f"[Weather] Weather overview: {weather_overview}")
            return weather_overview
            
        except Exception as e:
            print(f"[Weather] Error fetching weather data: {str(e)}")
            print(f"[Weather] Response content: {response.text if 'response' in locals() else 'No response'}")
            return None

    def test_connection(self):
        """
        Test method to verify API connection and data retrieval
        """
        print("\n=== Testing WeatherService Connection ===")
        print(f"Base URL: {self.base_url}")
        print(f"Coordinates: LAT={self.lat}, LON={self.lon}")
        print(f"API Key present: {bool(self.api_key)}")
        
        weather = self.get_current_weather()
        
        if weather:
            print("\n✅ Weather service test successful!")
            print(f"Current weather: {weather}")
        else:
            print("\n❌ Weather service test failed!")
        print("=====================================\n")
        
        return weather is not None
