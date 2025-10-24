#!/usr/bin/env python3
"""
Example usage of Order Flow Engine.
Demonstrates basic functionality and signal generation.
"""

import asyncio
import json
from datetime import datetime
from core.feeds import BinanceFeed, DataFeedManager
from core.metrics import OrderFlowCalculator
from core.signals import SignalGenerator
from core.utils import format_signal_for_json

async def example_callback(data):
    """Example callback for processing data."""
    print(f"Received data: {type(data).__name__} for {getattr(data, 'symbol', 'unknown')}")

async def main():
    """Example main function."""
    print("Order Flow Engine Example")
    print("=" * 40)
    
    # Initialize components
    feed_manager = DataFeedManager()
    calculator = OrderFlowCalculator()
    signal_generator = SignalGenerator({
        'cvd_threshold': 0.3,
        'imbalance_threshold': 0.2,
        'confidence_threshold': 0.6
    })
    
    # Setup data feed
    symbols = ['BTC', 'ETH']
    binance_feed = BinanceFeed(symbols, example_callback)
    feed_manager.add_feed(binance_feed)
    
    # Add calculator callback
    async def data_processor(data):
        """Process incoming data."""
        if hasattr(data, 'symbol'):
            if hasattr(data, 'price'):  # Trade
                calculator.add_trade(data)
            elif hasattr(data, 'bids'):  # Order book
                calculator.add_orderbook(data)
            
            # Generate signal
            signal = signal_generator.generate_signal(data.symbol)
            
            # Print signal
            if signal.signal_type.value != 'WAIT':
                signal_json = format_signal_for_json(signal)
                print(f"\n🚨 Signal Generated:")
                print(json.dumps(signal_json, indent=2))
    
    feed_manager.add_callback(data_processor)
    
    print(f"Starting data feeds for symbols: {symbols}")
    print("Press Ctrl+C to stop...")
    
    try:
        # Start feeds
        feed_task = asyncio.create_task(feed_manager.start())
        
        # Run for 30 seconds
        await asyncio.sleep(30)
        
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        await feed_manager.stop()
        print("Example completed!")

if __name__ == "__main__":
    asyncio.run(main())
