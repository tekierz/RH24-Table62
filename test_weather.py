from weather_service import WeatherService

def main():
    # Create weather service instance
    weather_service = WeatherService()
    
    # Test the connection with more detailed error handling
    try:
        weather_service.test_connection()
    except Exception as e:
        print("\n❌ Weather service test failed!")
        print("Error details:")
        print("1. Verify you have subscribed to the 'One Call by Call' plan at:")
        print("   https://openweathermap.org/price")
        print("2. Ensure your API key is activated (can take up to 2 hours)")
        print("3. Verify your API key is correctly set in the configuration")
        print("\nAPI Response:", str(e))
    else:
        print("\n✅ Weather service test successful!")

if __name__ == "__main__":
    main()