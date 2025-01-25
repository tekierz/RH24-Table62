import os
import discord
from discord.ext import commands
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Enable all intents
intents = discord.Intents.all()

# Create bot instance
bot = discord.Client(intents=intents)

# Load token from environment variable
TOKEN = os.getenv('DISCORD_TOKEN')
if not TOKEN:
    print("Error: DISCORD_TOKEN not found in .env file")
    exit(1)

def send_image(channel_id, message, image_path):
    """
    Send an image to a specific channel
    
    :param channel_id: Discord channel ID to send image to as an integar

    :param message: string to send

    :param image_path: local path to image file
    """
    # Get the channel
    channel = bot.get_channel(channel_id)
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return
    
    # Async function to send image
    async def send_image_to_channel():
        with open(image_path, 'rb') as image_file:
            picture = discord.File(image_file)
            if message and message != "":
                await channel.send(message)
                
            await channel.send(file=picture)
    
    # Run the async function
    bot.loop.create_task(send_image_to_channel())

@bot.event
async def on_ready():
    print(f'Discord bot logged in as {bot.user.name}')

# Remove the main() function and if __name__ == '__main__' block