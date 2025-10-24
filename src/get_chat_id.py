#!/usr/bin/env python3
"""
Simple script to get your Telegram chat ID.
Run this script and then send a message to your bot @ClarityFlow_bot
"""

import asyncio
from telegram import Bot

async def get_chat_id():
    bot_token = "8251794018:AAHpVakPEvJyzTa8YVpXa1jXuqvjb0wtimY"
    bot = Bot(token=bot_token)
    
    print("🤖 Bot is running...")
    print("📱 Go to Telegram and send a message to @ClarityFlow_bot")
    print("⏳ Waiting for your message...")
    
    try:
        # Get updates
        updates = await bot.get_updates()
        
        if updates:
            for update in updates:
                if update.message:
                    chat_id = update.message.chat_id
                    print(f"\n✅ Found your chat ID: {chat_id}")
                    print(f"📋 Add this to your config.yaml:")
                    print(f"   chat_id: \"{chat_id}\"")
                    return chat_id
        else:
            print("❌ No messages found. Please send a message to @ClarityFlow_bot first.")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    chat_id = asyncio.run(get_chat_id())
