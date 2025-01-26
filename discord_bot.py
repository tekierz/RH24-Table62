import os
import discord # type: ignore
from discord.ext import commands # type: ignore
from dotenv import load_dotenv
import asyncio

# Load environment variables from .env file
load_dotenv()

# Enable all intents
intents = discord.Intents.all()

# Create bot instance
bot = discord.Client(intents=intents)

async def send_image(message, image_path):
    """
    Send an image to a specific channel

    :param message: string to send

    :param image_path: local path to image file



    """
    # Get the channel
    CHANNEL_ID = os.getenv('DISCORD_CHANNEL_ID')
    channel = bot.get_channel(int(CHANNEL_ID))
    
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
