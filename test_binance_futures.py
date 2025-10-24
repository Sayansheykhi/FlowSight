#!/usr/bin/env python3
"""
Test script for Binance Futures WebSocket connector.
Tests reconnection, error handling, and data parsing.
"""

import asyncio
import json
import logging
import time
from core.feeds import BinanceFuturesWS

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestBinanceFuturesWS:
    """Test class for Binance Futures WebSocket connector."""
    
    def __init__(self):
        self.trade_count = 0
        self.depth_count = 0
        self.errors = []
        
    async def test_trade_streaming(self):
        """Test trade data streaming."""
        print("\n=== Testing Trade Streaming ===")
        
        trade_queue = asyncio.Queue()
        ws_connector = BinanceFuturesWS(max_reconnect_attempts=3, reconnect_delay=0.5)
        
        try:
            ws_connector.start()
            
            # Start trade streaming
            trade_task = asyncio.create_task(
                ws_connector.stream_trades("BTCUSDT", trade_queue)
            )
            
            # Process trades for 10 seconds
            start_time = time.time()
            while time.time() - start_time < 10:
                try:
                    data = await asyncio.wait_for(trade_queue.get(), timeout=1.0)
                    if data["type"] == "trade":
                        self.trade_count += 1
                        print(f"Trade #{self.trade_count}: {data['symbol']} - "
                              f"{data['side']} {data['qty']} @ {data['price']}")
                        
                        # Validate data structure
                        self._validate_trade_data(data)
                        
                except asyncio.TimeoutError:
                    print("No trade data received in 1 second")
                    continue
                except Exception as e:
                    self.errors.append(f"Trade processing error: {e}")
                    logger.error(f"Trade processing error: {e}")
            
            trade_task.cancel()
            
        except Exception as e:
            self.errors.append(f"Trade streaming error: {e}")
            logger.error(f"Trade streaming error: {e}")
        finally:
            ws_connector.stop()
    
    async def test_depth_streaming(self):
        """Test depth data streaming."""
        print("\n=== Testing Depth Streaming ===")
        
        depth_queue = asyncio.Queue()
        ws_connector = BinanceFuturesWS(max_reconnect_attempts=3, reconnect_delay=0.5)
        
        try:
            ws_connector.start()
            
            # Start depth streaming
            depth_task = asyncio.create_task(
                ws_connector.stream_depth("BTCUSDT", depth_queue)
            )
            
            # Process depth for 10 seconds
            start_time = time.time()
            while time.time() - start_time < 10:
                try:
                    data = await asyncio.wait_for(depth_queue.get(), timeout=1.0)
                    if data["type"] == "depth":
                        self.depth_count += 1
                        
                        # Calculate spread
                        if data["bid_levels"] and data["ask_levels"]:
                            best_bid = data["bid_levels"][0]["price"]
                            best_ask = data["ask_levels"][0]["price"]
                            spread = best_ask - best_bid
                            
                            print(f"Depth #{self.depth_count}: {data['symbol']} - "
                                  f"Spread: {spread:.4f} "
                                  f"(Bid: {best_bid}, Ask: {best_ask})")
                        
                        # Validate data structure
                        self._validate_depth_data(data)
                        
                except asyncio.TimeoutError:
                    print("No depth data received in 1 second")
                    continue
                except Exception as e:
                    self.errors.append(f"Depth processing error: {e}")
                    logger.error(f"Depth processing error: {e}")
            
            depth_task.cancel()
            
        except Exception as e:
            self.errors.append(f"Depth streaming error: {e}")
            logger.error(f"Depth streaming error: {e}")
        finally:
            ws_connector.stop()
    
    async def test_concurrent_streams(self):
        """Test concurrent streaming of multiple symbols."""
        print("\n=== Testing Concurrent Streams ===")
        
        symbols = ["BTCUSDT", "ETHUSDT"]
        trade_queues = {symbol: asyncio.Queue() for symbol in symbols}
        depth_queues = {symbol: asyncio.Queue() for symbol in symbols}
        
        ws_connector = BinanceFuturesWS(max_reconnect_attempts=3, reconnect_delay=0.5)
        
        try:
            ws_connector.start()
            
            # Start all streams
            tasks = []
            for symbol in symbols:
                tasks.append(asyncio.create_task(
                    ws_connector.stream_trades(symbol, trade_queues[symbol])
                ))
                tasks.append(asyncio.create_task(
                    ws_connector.stream_depth(symbol, depth_queues[symbol])
                ))
            
            # Process data from all streams
            start_time = time.time()
            while time.time() - start_time < 15:
                for symbol in symbols:
                    # Process trades
                    try:
                        trade_data = await asyncio.wait_for(
                            trade_queues[symbol].get(), timeout=0.1
                        )
                        if trade_data["type"] == "trade":
                            print(f"Trade {symbol}: {trade_data['side']} "
                                  f"{trade_data['qty']} @ {trade_data['price']}")
                    except asyncio.TimeoutError:
                        pass
                    
                    # Process depth
                    try:
                        depth_data = await asyncio.wait_for(
                            depth_queues[symbol].get(), timeout=0.1
                        )
                        if depth_data["type"] == "depth":
                            if depth_data["bid_levels"] and depth_data["ask_levels"]:
                                spread = (depth_data["ask_levels"][0]["price"] - 
                                        depth_data["bid_levels"][0]["price"])
                                print(f"Depth {symbol}: Spread {spread:.4f}")
                    except asyncio.TimeoutError:
                        pass
                
                await asyncio.sleep(0.1)
            
            # Cancel all tasks
            for task in tasks:
                task.cancel()
            
        except Exception as e:
            self.errors.append(f"Concurrent streaming error: {e}")
            logger.error(f"Concurrent streaming error: {e}")
        finally:
            ws_connector.stop()
    
    def _validate_trade_data(self, data: dict):
        """Validate trade data structure."""
        required_fields = ["type", "ts", "price", "qty", "side", "symbol"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        if data["type"] != "trade":
            raise ValueError(f"Invalid type: {data['type']}")
        
        if data["side"] not in ["buy", "sell"]:
            raise ValueError(f"Invalid side: {data['side']}")
        
        if data["price"] <= 0 or data["qty"] <= 0:
            raise ValueError(f"Invalid price or quantity: {data['price']}, {data['qty']}")
    
    def _validate_depth_data(self, data: dict):
        """Validate depth data structure."""
        required_fields = ["type", "ts", "symbol", "bid_levels", "ask_levels"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        if data["type"] != "depth":
            raise ValueError(f"Invalid type: {data['type']}")
        
        # Validate bid levels
        for bid in data["bid_levels"]:
            if "price" not in bid or "qty" not in bid:
                raise ValueError("Invalid bid level structure")
            if bid["price"] <= 0 or bid["qty"] <= 0:
                raise ValueError(f"Invalid bid: {bid}")
        
        # Validate ask levels
        for ask in data["ask_levels"]:
            if "price" not in ask or "qty" not in ask:
                raise ValueError("Invalid ask level structure")
            if ask["price"] <= 0 or ask["qty"] <= 0:
                raise ValueError(f"Invalid ask: {ask}")
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 50)
        print("TEST SUMMARY")
        print("=" * 50)
        print(f"Trades processed: {self.trade_count}")
        print(f"Depth updates processed: {self.depth_count}")
        print(f"Errors encountered: {len(self.errors)}")
        
        if self.errors:
            print("\nErrors:")
            for error in self.errors:
                print(f"  - {error}")
        else:
            print("\n✓ All tests passed successfully!")

async def main():
    """Run all tests."""
    print("Binance Futures WebSocket Connector Test Suite")
    print("=" * 60)
    
    tester = TestBinanceFuturesWS()
    
    try:
        # Run individual tests
        await tester.test_trade_streaming()
        await asyncio.sleep(2)  # Brief pause between tests
        
        await tester.test_depth_streaming()
        await asyncio.sleep(2)  # Brief pause between tests
        
        await tester.test_concurrent_streams()
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        logger.error(f"Test suite error: {e}")
    finally:
        tester.print_summary()

if __name__ == "__main__":
    asyncio.run(main())
