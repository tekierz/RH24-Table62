import os
import discord
from discord.ext import commands

# Read token from token.txt
def load_token():
    try:
        with open('token.txt', 'r') as token_file:
            return token_file.read().strip()
    except FileNotFoundError:
        print("Error: token.txt file not found")
        exit(1)

# Enable all intents
intents = discord.Intents.all()

# Create bot instance
bot = discord.Client(intents=intents)

# Load token from file
TOKEN = load_token()

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
            if message and message not "":
                await channel.send(message)
                
            await channel.send(file=picture)
    
    # Run the async function
    bot.loop.create_task(send_image_to_channel())

@bot.event
async def on_ready():
    print(f'Bot is logged in as {bot.user.name}')
    # Example: Send image to a specific channel when bot is ready
    # Replace CHANNEL_ID with the actual channel ID you want to send to

def main():
    bot.run(TOKEN)

if __name__ == '__main__':
    main()