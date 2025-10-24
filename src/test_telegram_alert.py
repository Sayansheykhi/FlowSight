#!/usr/bin/env python3
"""
Send a test alert to your Telegram bot
"""

import asyncio
import yaml
from telegram import Bot

async def send_test_alert():
    # Load config to get chat_id
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        bot_token = config['telegram']['token']
        chat_id = config['telegram']['chat_id']
        
        if not chat_id or chat_id == "":
            print("❌ Chat ID not found in config.yaml")
            print("   Please run: python3 get_chat_id_enhanced.py")
            return
        
        bot = Bot(token=bot_token)
        
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

⏰ *Time:* 2025-10-22 19:08:00 UTC
🤖 *OrderFlow Engine*
        """.strip()
        
        await bot.send_message(
            chat_id=chat_id,
            text=test_message,
            parse_mode='Markdown',
            disable_web_page_preview=True
        )
        print("✅ Test alert sent successfully!")
        print(f"📱 Check your Telegram chat with @ClarityFlow_bot")
        
    except Exception as e:
        print(f"❌ Error sending test alert: {e}")

if __name__ == "__main__":
    asyncio.run(send_test_alert())
