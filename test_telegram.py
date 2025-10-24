#!/usr/bin/env python3
"""
Test script to send a sample alert to your Telegram bot
"""

import asyncio
from telegram import Bot

async def send_test_alert():
    bot_token = "8251794018:AAHpVakPEvJyzTa8YVpXa1jXuqvjb0wtimY"
    bot = Bot(token=bot_token)
    
    # You'll need to replace this with your actual chat ID
    chat_id = "YOUR_CHAT_ID_HERE"  # Replace with your chat ID
    
    test_message = """
🟢 *ORDER FLOW ALERT* 🟢

🎯 *Signal:* 🟢 *LONG*
🔥 *Confidence:* 85%
💰 *Symbol:* `BTCUSDT`
📈 *Price:* $108,234.20
🟢 *Session:* Active

📊 *Order Flow Metrics:*
🚀 CVD: `+0.3456`
⬆️ Imbalance: `+0.1234`
📊 VWAP: `$108,200.00`
🏢 Absorption: `0.4567`

🧠 *Analysis:*
• CVD rising (+0.346) - buying pressure
• Strong bid imbalance (+0.123)
• Price 0.32% above VWAP
• High absorption (0.457) - institutional activity
• High confidence signal

⏰ *Time:* 2025-10-22 19:02:45 UTC
🤖 *OrderFlow Engine*
    """.strip()
    
    try:
        await bot.send_message(
            chat_id=chat_id,
            text=test_message,
            parse_mode='Markdown',
            disable_web_page_preview=True
        )
        print("✅ Test alert sent successfully!")
    except Exception as e:
        print(f"❌ Error sending test alert: {e}")

if __name__ == "__main__":
    print("⚠️  Remember to replace YOUR_CHAT_ID_HERE with your actual chat ID")
    print("Run: python3 get_chat_id.py to get your chat ID first")
    # asyncio.run(send_test_alert())
