#!/usr/bin/env python3
"""
Enhanced script to get your Telegram chat ID with continuous checking
"""

import asyncio
import time
from telegram import Bot

async def get_chat_id_continuous():
    bot_token = "8251794018:AAHpVakPEvJyzTa8YVpXa1jXuqvjb0wtimY"
    bot = Bot(token=bot_token)
    
    print("🤖 Bot is running...")
    print("📱 Go to Telegram and send a message to @ClarityFlow_bot")
    print("⏳ Checking for messages every 2 seconds...")
    print("   (Press Ctrl+C to stop)")
    
    try:
        while True:
            try:
                # Get updates
                updates = await bot.get_updates()
                
                if updates:
                    for update in updates:
                        if update.message:
                            chat_id = update.message.chat_id
                            username = update.message.from_user.username or "Unknown"
                            first_name = update.message.from_user.first_name or "Unknown"
                            
                            print(f"\n✅ Found message from: {first_name} (@{username})")
                            print(f"📋 Your chat ID: {chat_id}")
                            print(f"\n🔧 Add this to your config.yaml:")
                            print(f"   chat_id: \"{chat_id}\"")
                            
                            # Update the config file automatically
                            await update_config_file(chat_id)
                            return chat_id
                
                print(".", end="", flush=True)
                await asyncio.sleep(2)
                
            except Exception as e:
                print(f"\n❌ Error: {e}")
                await asyncio.sleep(5)
                
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopped by user")
        return None

async def update_config_file(chat_id):
    """Update the config.yaml file with the chat ID"""
    try:
        # Read current config
        with open('config.yaml', 'r') as f:
            content = f.read()
        
        # Replace the chat_id line
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'chat_id:' in line and 'YOUR_CHAT_ID_HERE' in line:
                lines[i] = f'  chat_id: "{chat_id}"                 # Your Telegram chat ID'
                break
        
        # Write updated config
        with open('config.yaml', 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"✅ Config file updated with chat ID: {chat_id}")
        
    except Exception as e:
        print(f"❌ Error updating config: {e}")

if __name__ == "__main__":
    chat_id = asyncio.run(get_chat_id_continuous())
