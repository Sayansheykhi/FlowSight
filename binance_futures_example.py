#!/usr/bin/env python3
"""
Example usage of Binance Futures WebSocket connector.
Demonstrates streaming trades and depth data.
"""

import asyncio
import json
import logging
from core.feeds import BinanceFuturesWS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def process_trade_data(queue: asyncio.Queue):
    """Process trade data from queue."""
    while True:
        try:
            data = await asyncio.wait_for(queue.get(), timeout=1.0)
            if data["type"] == "trade":
                print(f"Trade: {data['symbol']} - {data['side']} {data['qty']} @ {data['price']}")
        except asyncio.TimeoutError:
            continue
        except Exception as e:
            logger.error(f"Error processing trade data: {e}")

async def process_depth_data(queue: asyncio.Queue):
    """Process depth data from queue."""
    while True:
        try:
            data = await asyncio.wait_for(queue.get(), timeout=1.0)
            if data["type"] == "depth":
                best_bid = data["bid_levels"][0] if data["bid_levels"] else None
                best_ask = data["ask_levels"][0] if data["ask_levels"] else None
                if best_bid and best_ask:
                    spread = best_ask["price"] - best_bid["price"]
                    print(f"Depth: {data['symbol']} - Spread: {spread:.4f} "
                          f"(Bid: {best_bid['price']}, Ask: {best_ask['price']})")
        except asyncio.TimeoutError:
            continue
        except Exception as e:
            logger.error(f"Error processing depth data: {e}")

async def main():
    """Main example function."""
    print("Binance Futures WebSocket Example")
    print("=" * 40)
    
    # Create queues
    trade_queue = asyncio.Queue()
    depth_queue = asyncio.Queue()
    
    # Create WebSocket connector
    ws_connector = BinanceFuturesWS()
    
    # Start the connector
    ws_connector.start()
    
    # Create tasks
    tasks = [
        # Data processing tasks
        asyncio.create_task(process_trade_data(trade_queue)),
        asyncio.create_task(process_depth_data(depth_queue)),
        
        # WebSocket streaming tasks
        asyncio.create_task(ws_connector.stream_trades("BTCUSDT", trade_queue)),
        asyncio.create_task(ws_connector.stream_depth("BTCUSDT", depth_queue)),
        
        # Ping task
        asyncio.create_task(ws_connector.start_ping_task())
    ]
    
    try:
        print("Starting data streams for BTCUSDT...")
        print("Press Ctrl+C to stop...")
        
        # Run for 30 seconds
        await asyncio.sleep(30)
        
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        # Stop the connector
        ws_connector.stop()
        
        # Cancel all tasks
        for task in tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        print("Example completed!")

if __name__ == "__main__":
    asyncio.run(main())
