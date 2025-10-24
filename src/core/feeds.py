"""
Live data feeds for order flow analysis.
Handles WebSocket connections to exchanges and data normalization.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime
import websockets
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Represents a single trade."""
    timestamp: datetime
    price: float
    size: float
    side: str  # 'buy' or 'sell'
    symbol: str

@dataclass
class OrderBookLevel:
    """Represents a single level in the order book."""
    price: float
    size: float
    side: str  # 'bid' or 'ask'

@dataclass
class OrderBookSnapshot:
    """Complete order book snapshot."""
    timestamp: datetime
    symbol: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]

class BinanceFeed:
    """Binance WebSocket feed implementation."""
    
    def __init__(self, symbols: List[str], callback: Callable[[Any], None]):
        self.symbols = symbols
        self.callback = callback
        self.websocket = None
        self.running = False
        
    async def connect(self):
        """Connect to Binance WebSocket streams."""
        streams = []
        for symbol in self.symbols:
            symbol_lower = symbol.lower()
            streams.extend([
                f"{symbol_lower}@trade",
                f"{symbol_lower}@depth20@100ms"
            ])
        
        stream_names = "/".join(streams)
        uri = f"wss://stream.binance.com:9443/stream?streams={stream_names}"
        
        try:
            self.websocket = await websockets.connect(uri)
            self.running = True
            logger.info(f"Connected to Binance WebSocket for {self.symbols}")
            
            async for message in self.websocket:
                if not self.running:
                    break
                    
                data = json.loads(message)
                await self._process_message(data)
                
        except Exception as e:
            logger.error(f"Binance WebSocket error: {e}")
            raise
    
    async def _process_message(self, data: Dict):
        """Process incoming WebSocket message."""
        try:
            stream_data = data.get('data', {})
            stream_name = data.get('stream', '')
            
            if '@trade' in stream_name:
                trade = self._parse_trade(stream_data)
                if trade:
                    await self.callback(trade)
                    
            elif '@depth' in stream_name:
                orderbook = self._parse_orderbook(stream_data)
                if orderbook:
                    await self.callback(orderbook)
                    
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def _parse_trade(self, data: Dict) -> Optional[Trade]:
        """Parse trade data from Binance."""
        try:
            symbol = data.get('s', '').replace('USDT', '')
            return Trade(
                timestamp=datetime.fromtimestamp(data['T'] / 1000),
                price=float(data['p']),
                size=float(data['q']),
                side='buy' if data['m'] else 'sell',  # m=True means buyer is maker (sell)
                symbol=symbol
            )
        except Exception as e:
            logger.error(f"Error parsing trade: {e}")
            return None
    
    def _parse_orderbook(self, data: Dict) -> Optional[OrderBookSnapshot]:
        """Parse order book data from Binance."""
        try:
            symbol = data.get('s', '').replace('USDT', '')
            bids = [OrderBookLevel(float(b[0]), float(b[1]), 'bid') 
                   for b in data.get('b', [])]
            asks = [OrderBookLevel(float(a[0]), float(a[1]), 'ask') 
                   for a in data.get('a', [])]
            
            return OrderBookSnapshot(
                timestamp=datetime.fromtimestamp(data['E'] / 1000),
                symbol=symbol,
                bids=bids,
                asks=asks
            )
        except Exception as e:
            logger.error(f"Error parsing orderbook: {e}")
            return None
    
    async def disconnect(self):
        """Disconnect from WebSocket."""
        self.running = False
        if self.websocket:
            await self.websocket.close()

class CoinbaseFuturesWS:
    """Robust async Coinbase Advanced Trade WebSocket connector."""
    
    def __init__(self, max_reconnect_attempts: int = 10, reconnect_delay: float = 1.0):
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.ws_url = "wss://advanced-trade-ws.coinbase.com"
        self.websocket = None
        self.running = False
        self.last_ping = 0
        self.ping_interval = 20  # seconds
        self.reconnect_count = 0
        
    async def stream_trades(self, symbol: str, queue: asyncio.Queue) -> None:
        """
        Stream trade data for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC-USD')
            queue: asyncio.Queue to push parsed trade data
        """
        while self.running:
            try:
                logger.info(f"Connecting to Coinbase Advanced Trade trades stream: {symbol}")
                
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=10
                ) as websocket:
                    self.websocket = websocket
                    self.reconnect_count = 0
                    
                    # Subscribe to trades channel
                    subscribe_message = {
                        "type": "subscribe",
                        "product_ids": [symbol],
                        "channels": ["ticker"]
                    }
                    
                    await websocket.send(json.dumps(subscribe_message))
                    logger.info(f"Connected to Coinbase Advanced Trade trades stream: {symbol}")
                    
                    async for message in websocket:
                        if not self.running:
                            break
                            
                        try:
                            data = json.loads(message)
                            parsed_trade = self._parse_trade_data(data)
                            
                            if parsed_trade:
                                await queue.put(parsed_trade)
                                
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error in Coinbase trades stream: {e}")
                        except Exception as e:
                            logger.error(f"Error processing Coinbase trade message: {e}")
                            
            except websockets.exceptions.ConnectionClosed:
                logger.warning(f"Coinbase trades stream connection closed for {symbol}")
            except Exception as e:
                logger.error(f"Coinbase trades stream error for {symbol}: {e}")
            
            if self.running:
                await self._handle_reconnect()
    
    async def stream_depth(self, symbol: str, queue: asyncio.Queue) -> None:
        """
        Stream order book depth data for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC-USD')
            queue: asyncio.Queue to push parsed depth data
        """
        while self.running:
            try:
                logger.info(f"Connecting to Coinbase Advanced Trade depth stream: {symbol}")
                
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=10
                ) as websocket:
                    self.websocket = websocket
                    self.reconnect_count = 0
                    
                    # Subscribe to level2 channel
                    subscribe_message = {
                        "type": "subscribe",
                        "product_ids": [symbol],
                        "channels": ["level2"]
                    }
                    
                    await websocket.send(json.dumps(subscribe_message))
                    logger.info(f"Connected to Coinbase Advanced Trade depth stream: {symbol}")
                    
                    async for message in websocket:
                        if not self.running:
                            break
                            
                        try:
                            data = json.loads(message)
                            parsed_depth = self._parse_depth_data(data)
                            
                            if parsed_depth:
                                await queue.put(parsed_depth)
                                
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error in Coinbase depth stream: {e}")
                        except Exception as e:
                            logger.error(f"Error processing Coinbase depth message: {e}")
                            
            except websockets.exceptions.ConnectionClosed:
                logger.warning(f"Coinbase depth stream connection closed for {symbol}")
            except Exception as e:
                logger.error(f"Coinbase depth stream error for {symbol}: {e}")
            
            if self.running:
                await self._handle_reconnect()
    
    def _parse_trade_data(self, data: Dict) -> Optional[Trade]:
        """Parse trade data from Coinbase Advanced Trade."""
        try:
            if data.get('type') != 'ticker':
                return None
                
            product_id = data.get('product_id', '')
            if not product_id:
                return None
                
            # Convert Coinbase symbol format to our format
            symbol = product_id.replace('-USD', '').lower() + 'usdt'
            
            # Extract trade information from ticker
            price = float(data.get('price', 0))
            size = float(data.get('last_size', 0))
            
            if price <= 0 or size <= 0:
                return None
            
            # Determine side from ticker data
            # Coinbase doesn't provide explicit buy/sell in ticker, so we'll use a heuristic
            side = 'buy'  # Default to buy, could be enhanced with more sophisticated logic
            
            return Trade(
                timestamp=datetime.fromisoformat(data['time'].replace('Z', '+00:00')),
                price=price,
                size=size,
                side=side,
                symbol=symbol
            )
            
        except Exception as e:
            logger.error(f"Error parsing Coinbase trade data: {e}")
            return None
    
    def _parse_depth_data(self, data: Dict) -> Optional[OrderBookSnapshot]:
        """Parse order book depth data from Coinbase Advanced Trade."""
        try:
            if data.get('type') != 'l2update':
                return None
                
            product_id = data.get('product_id', '')
            if not product_id:
                return None
                
            # Convert Coinbase symbol format to our format
            symbol = product_id.replace('-USD', '').lower() + 'usdt'
            
            bids = []
            asks = []
            
            # Parse changes
            changes = data.get('changes', [])
            for change in changes:
                side, price_str, size_str = change
                price = float(price_str)
                size = float(size_str)
                
                if side == 'buy' and size > 0:
                    bids.append(OrderBookLevel(price, size, 'bid'))
                elif side == 'sell' and size > 0:
                    asks.append(OrderBookLevel(price, size, 'ask'))
            
            if not bids and not asks:
                return None
            
            return OrderBookSnapshot(
                timestamp=datetime.fromisoformat(data['time'].replace('Z', '+00:00')),
                symbol=symbol,
                bids=bids,
                asks=asks
            )
            
        except Exception as e:
            logger.error(f"Error parsing Coinbase depth data: {e}")
            return None
    
    async def _handle_reconnect(self):
        """Handle reconnection logic."""
        if self.reconnect_count >= self.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached for Coinbase WebSocket")
            self.running = False
            return
        
        self.reconnect_count += 1
        delay = self.reconnect_delay * (2 ** min(self.reconnect_count, 5))  # Exponential backoff
        logger.info(f"Reconnecting to Coinbase WebSocket in {delay} seconds (attempt {self.reconnect_count})")
        await asyncio.sleep(delay)
    
    def start(self):
        """Start the WebSocket connection."""
        self.running = True
    
    def stop(self):
        """Stop the WebSocket connection."""
        self.running = False
        if self.websocket:
            asyncio.create_task(self.websocket.close())

class DataFeedManager:
    """Manages multiple data feeds and aggregates data."""
    
    def __init__(self):
        self.feeds: List[Any] = []
        self.trade_buffer: List[Trade] = []
        self.orderbook_buffer: List[OrderBookSnapshot] = []
        self.callbacks: List[Callable] = []
        
    def add_feed(self, feed):
        """Add a data feed."""
        self.feeds.append(feed)
        
    def add_callback(self, callback: Callable):
        """Add a callback for processed data."""
        self.callbacks.append(callback)
        
    async def start(self):
        """Start all feeds."""
        tasks = []
        for feed in self.feeds:
            task = asyncio.create_task(feed.connect())
            tasks.append(task)
        
        await asyncio.gather(*tasks)
    
    async def stop(self):
        """Stop all feeds."""
        for feed in self.feeds:
            await feed.disconnect()
    
    async def _process_data(self, data):
        """Process incoming data and notify callbacks."""
        if isinstance(data, Trade):
            self.trade_buffer.append(data)
            # Keep only last 1000 trades
            if len(self.trade_buffer) > 1000:
                self.trade_buffer = self.trade_buffer[-1000:]
                
        elif isinstance(data, OrderBookSnapshot):
            self.orderbook_buffer.append(data)
            # Keep only last 100 snapshots
            if len(self.orderbook_buffer) > 100:
                self.orderbook_buffer = self.orderbook_buffer[-100:]
        
        # Notify all callbacks
        for callback in self.callbacks:
            try:
                await callback(data)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Trade]:
        """Get recent trades for a symbol."""
        return [t for t in self.trade_buffer if t.symbol == symbol][-limit:]
    
    def get_latest_orderbook(self, symbol: str) -> Optional[OrderBookSnapshot]:
        """Get latest order book for a symbol."""
        for ob in reversed(self.orderbook_buffer):
            if ob.symbol == symbol:
                return ob
        return None

class BinanceFuturesWS:
    """Robust async Binance Futures WebSocket connector."""
    
    def __init__(self, max_reconnect_attempts: int = 10, reconnect_delay: float = 1.0):
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.ws_url = "wss://fstream.binance.com/ws/"
        self.ws_combined_url = "wss://fstream.binance.com/stream"
        self.websocket = None
        self.running = False
        self.last_ping = 0
        self.ping_interval = 20  # seconds
        self.reconnect_count = 0
        
    async def stream_trades(self, symbol: str, queue: asyncio.Queue) -> None:
        """
        Stream trade data for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            queue: asyncio.Queue to push parsed trade data
        """
        stream_name = f"{symbol.lower()}@trade"
        uri = f"{self.ws_url}{stream_name}"
        
        while self.running:
            try:
                logger.info(f"Connecting to Binance Futures trades stream: {stream_name}")
                
                async with websockets.connect(
                    uri,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=10
                ) as websocket:
                    self.websocket = websocket
                    self.reconnect_count = 0
                    
                    logger.info(f"Connected to Binance Futures trades stream: {stream_name}")
                    
                    async for message in websocket:
                        if not self.running:
                            break
                            
                        try:
                            data = json.loads(message)
                            parsed_trade = self._parse_trade_data(data)
                            
                            if parsed_trade:
                                await queue.put(parsed_trade)
                                
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error in trades stream: {e}")
                        except Exception as e:
                            logger.error(f"Error processing trade message: {e}")
                            
            except websockets.exceptions.ConnectionClosed:
                logger.warning(f"Trades stream connection closed for {symbol}")
            except Exception as e:
                logger.error(f"Trades stream error for {symbol}: {e}")
            
            if self.running:
                await self._handle_reconnect()
    
    async def stream_depth(self, symbol: str, queue: asyncio.Queue) -> None:
        """
        Stream order book depth data for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            queue: asyncio.Queue to push parsed depth data
        """
        stream_name = f"{symbol.lower()}@depth10"
        uri = f"{self.ws_url}{stream_name}"
        
        while self.running:
            try:
                logger.info(f"Connecting to Binance Futures depth stream: {stream_name}")
                
                async with websockets.connect(
                    uri,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=10
                ) as websocket:
                    self.websocket = websocket
                    self.reconnect_count = 0
                    
                    logger.info(f"Connected to Binance Futures depth stream: {stream_name}")
                    
                    async for message in websocket:
                        if not self.running:
                            break
                            
                        try:
                            data = json.loads(message)
                            parsed_depth = self._parse_depth_data(data)
                            
                            if parsed_depth:
                                await queue.put(parsed_depth)
                                
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error in depth stream: {e}")
                        except Exception as e:
                            logger.error(f"Error processing depth message: {e}")
                            
            except websockets.exceptions.ConnectionClosed:
                logger.warning(f"Depth stream connection closed for {symbol}")
            except Exception as e:
                logger.error(f"Depth stream error for {symbol}: {e}")
            
            if self.running:
                await self._handle_reconnect()
    
    def _parse_trade_data(self, data: Dict) -> Optional[Dict]:
        """Parse trade data from Binance Futures WebSocket."""
        try:
            return {
                "type": "trade",
                "ts": data.get("T", int(time.time() * 1000)),
                "price": float(data.get("p", 0)),
                "qty": float(data.get("q", 0)),
                "side": "sell" if data.get("m", False) else "buy",  # m=True means buyer is maker (sell)
                "symbol": data.get("s", ""),
                "trade_id": data.get("t", 0),
                "is_buyer_maker": data.get("m", False)
            }
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing trade data: {e}")
            return None
    
    def _parse_depth_data(self, data: Dict) -> Optional[Dict]:
        """Parse order book depth data from Binance Futures WebSocket."""
        try:
            bids = []
            asks = []
            
            # Parse bid levels
            for bid in data.get("b", []):
                if len(bid) >= 2:
                    bids.append({
                        "price": float(bid[0]),
                        "qty": float(bid[1])
                    })
            
            # Parse ask levels
            for ask in data.get("a", []):
                if len(ask) >= 2:
                    asks.append({
                        "price": float(ask[0]),
                        "qty": float(ask[1])
                    })
            
            return {
                "type": "depth",
                "ts": data.get("E", int(time.time() * 1000)),
                "symbol": data.get("s", ""),
                "bid_levels": bids,
                "ask_levels": asks,
                "first_update_id": data.get("U", 0),
                "final_update_id": data.get("u", 0)
            }
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing depth data: {e}")
            return None
    
    async def _handle_reconnect(self) -> None:
        """Handle reconnection logic with exponential backoff."""
        if self.reconnect_count >= self.max_reconnect_attempts:
            logger.error(f"Max reconnection attempts ({self.max_reconnect_attempts}) reached")
            self.running = False
            return
        
        self.reconnect_count += 1
        delay = min(self.reconnect_delay * (2 ** (self.reconnect_count - 1)), 60)
        
        logger.info(f"Reconnecting in {delay:.1f} seconds (attempt {self.reconnect_count}/{self.max_reconnect_attempts})")
        await asyncio.sleep(delay)
    
    async def _send_ping(self) -> None:
        """Send ping to keep connection alive."""
        if self.websocket and not self.websocket.closed:
            try:
                await self.websocket.ping()
                self.last_ping = time.time()
                logger.debug("Sent ping to Binance Futures WebSocket")
            except Exception as e:
                logger.error(f"Error sending ping: {e}")
    
    async def start_ping_task(self) -> None:
        """Start periodic ping task."""
        while self.running:
            try:
                await asyncio.sleep(self.ping_interval)
                if time.time() - self.last_ping > self.ping_interval:
                    await self._send_ping()
            except Exception as e:
                logger.error(f"Error in ping task: {e}")
    
    def start(self) -> None:
        """Start the WebSocket connector."""
        self.running = True
        logger.info("Binance Futures WebSocket connector started")
    
    def stop(self) -> None:
        """Stop the WebSocket connector."""
        self.running = False
        if self.websocket and not self.websocket.closed:
            asyncio.create_task(self.websocket.close())
        logger.info("Binance Futures WebSocket connector stopped")

class CoinbaseDerivativesWS:
    """Coinbase Derivatives Platform (CDP) WebSocket connector for futures."""
    
    def __init__(self, max_reconnect_attempts: int = 10, reconnect_delay: float = 1.0):
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.ws_url = "wss://advanced-trade-ws.coinbase.com"
        self.websocket = None
        self.running = False
        self.last_ping = 0
        self.ping_interval = 20
        self.reconnect_count = 0
        
    async def stream_trades(self, symbol: str, queue: asyncio.Queue) -> None:
        """
        Stream trade data for a Coinbase futures symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC-USD-PERP')
            queue: asyncio.Queue to push parsed trade data
        """
        while self.running:
            try:
                logger.info(f"Connecting to Coinbase Derivatives trades stream: {symbol}")
                
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=10
                ) as websocket:
                    self.websocket = websocket
                    self.reconnect_count = 0
                    
                    # Subscribe to trades channel for futures
                    subscribe_message = {
                        "type": "subscribe",
                        "product_ids": [symbol],
                        "channels": ["ticker", "level2"]
                    }
                    
                    await websocket.send(json.dumps(subscribe_message))
                    logger.info(f"Connected to Coinbase Derivatives trades stream: {symbol}")
                    
                    async for message in websocket:
                        if not self.running:
                            break
                            
                        try:
                            data = json.loads(message)
                            
                            if data.get("type") == "ticker":
                                trade = self._parse_trade(data)
                                if trade:
                                    await queue.put(trade)
                                    
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error: {e}")
                        except Exception as e:
                            logger.error(f"Error processing message: {e}")
                            
            except Exception as e:
                logger.error(f"Coinbase Derivatives WebSocket error: {e}")
                self.reconnect_count += 1
                
                if self.reconnect_count >= self.max_reconnect_attempts:
                    logger.error(f"Max reconnection attempts reached for {symbol}")
                    break
                    
                logger.info(f"Reconnecting in {self.reconnect_delay} seconds...")
                await asyncio.sleep(self.reconnect_delay)
                self.reconnect_delay = min(self.reconnect_delay * 2, 60)
    
    async def stream_depth(self, symbol: str, queue: asyncio.Queue) -> None:
        """
        Stream order book depth data for a Coinbase futures symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC-USD-PERP')
            queue: asyncio.Queue to push parsed depth data
        """
        while self.running:
            try:
                logger.info(f"Connecting to Coinbase Derivatives depth stream: {symbol}")
                
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=10
                ) as websocket:
                    self.websocket = websocket
                    self.reconnect_count = 0
                    
                    # Subscribe to level2 channel for order book
                    subscribe_message = {
                        "type": "subscribe",
                        "product_ids": [symbol],
                        "channels": ["level2"]
                    }
                    
                    await websocket.send(json.dumps(subscribe_message))
                    logger.info(f"Connected to Coinbase Derivatives depth stream: {symbol}")
                    
                    async for message in websocket:
                        if not self.running:
                            break
                            
                        try:
                            data = json.loads(message)
                            
                            if data.get("type") == "l2update":
                                depth = self._parse_depth(data)
                                if depth:
                                    await queue.put(depth)
                                    
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error: {e}")
                        except Exception as e:
                            logger.error(f"Error processing message: {e}")
                            
            except Exception as e:
                logger.error(f"Coinbase Derivatives WebSocket error: {e}")
                self.reconnect_count += 1
                
                if self.reconnect_count >= self.max_reconnect_attempts:
                    logger.error(f"Max reconnection attempts reached for {symbol}")
                    break
                    
                logger.info(f"Reconnecting in {self.reconnect_delay} seconds...")
                await asyncio.sleep(self.reconnect_delay)
                self.reconnect_delay = min(self.reconnect_delay * 2, 60)
    
    def _parse_trade(self, data: Dict) -> Optional[Trade]:
        """Parse trade data from Coinbase Derivatives."""
        try:
            symbol = data.get('product_id', '').replace('-PERP', '')
            price = float(data.get('price', 0))
            size = float(data.get('last_size', 0))
            
            # Determine side based on trade data
            side = 'buy' if data.get('side') == 'buy' else 'sell'
            
            return Trade(
                timestamp=datetime.now(),
                price=price,
                size=size,
                side=side,
                symbol=symbol
            )
        except Exception as e:
            logger.error(f"Error parsing Coinbase Derivatives trade: {e}")
            return None
    
    def _parse_depth(self, data: Dict) -> Optional[Dict]:
        """Parse order book depth data from Coinbase Derivatives."""
        try:
            symbol = data.get('product_id', '').replace('-PERP', '')
            
            # Parse bid and ask levels
            bid_levels = []
            ask_levels = []
            
            changes = data.get('changes', [])
            for change in changes:
                side, price, size = change
                level = {'price': float(price), 'quantity': float(size)}
                
                if side == 'buy':
                    bid_levels.append(level)
                else:
                    ask_levels.append(level)
            
            return {
                'symbol': symbol,
                'bid_levels': bid_levels,
                'ask_levels': ask_levels,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error parsing Coinbase Derivatives depth: {e}")
            return None
    
    def start(self) -> None:
        """Start the WebSocket connector."""
        self.running = True
        logger.info("Coinbase Derivatives WebSocket connector started")
    
    def stop(self) -> None:
        """Stop the WebSocket connector."""
        self.running = False
        if self.websocket and not self.websocket.closed:
            asyncio.create_task(self.websocket.close())
        logger.info("Coinbase Derivatives WebSocket connector stopped")

class CoinbaseDerivativesREST:
    """Coinbase Derivatives Platform (CDP) REST API connector for futures."""
    
    def __init__(self, api_base_url: str = "https://api.cdp.coinbase.com", 
                 max_reconnect_attempts: int = 10, reconnect_delay: float = 1.0):
        self.api_base_url = api_base_url
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.running = False
        self.reconnect_count = 0
        self.session = None
        
    async def stream_trades(self, symbol: str, queue: asyncio.Queue) -> None:
        """
        Stream trade data for a Coinbase derivatives symbol using REST API polling.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC-USD-PERP')
            queue: asyncio.Queue to push parsed trade data
        """
        import aiohttp
        
        while self.running:
            try:
                logger.info(f"Starting Coinbase Derivatives REST polling for trades: {symbol}")
                
                async with aiohttp.ClientSession() as session:
                    self.session = session
                    self.reconnect_count = 0
                    
                    while self.running:
                        try:
                            # Get recent trades from Coinbase Derivatives
                            url = f"{self.api_base_url}/products/{symbol}/trades"
                            
                            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                                if response.status == 200:
                                    data = await response.json()
                                    
                                    if isinstance(data, list) and data:
                                        # Process the most recent trade
                                        latest_trade = data[-1]
                                        trade = self._parse_trade(latest_trade, symbol)
                                        if trade:
                                            await queue.put(trade)
                                    
                                elif response.status == 404:
                                    logger.warning(f"Symbol {symbol} not found on Coinbase Derivatives")
                                    break
                                else:
                                    logger.warning(f"HTTP {response.status} for {symbol} trades")
                            
                            # Poll every 1 second
                            await asyncio.sleep(1.0)
                            
                        except asyncio.TimeoutError:
                            logger.warning(f"Timeout fetching trades for {symbol}")
                        except Exception as e:
                            logger.error(f"Error fetching trades for {symbol}: {e}")
                            await asyncio.sleep(self.reconnect_delay)
                            
            except Exception as e:
                logger.error(f"Coinbase Derivatives REST error: {e}")
                self.reconnect_count += 1
                
                if self.reconnect_count >= self.max_reconnect_attempts:
                    logger.error(f"Max reconnection attempts reached for {symbol}")
                    break
                    
                logger.info(f"Reconnecting in {self.reconnect_delay} seconds...")
                await asyncio.sleep(self.reconnect_delay)
                self.reconnect_delay = min(self.reconnect_delay * 2, 60)
    
    async def stream_depth(self, symbol: str, queue: asyncio.Queue) -> None:
        """
        Stream order book depth data for a Coinbase derivatives symbol using REST API polling.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC-USD-PERP')
            queue: asyncio.Queue to push parsed depth data
        """
        import aiohttp
        
        while self.running:
            try:
                logger.info(f"Starting Coinbase Derivatives REST polling for depth: {symbol}")
                
                async with aiohttp.ClientSession() as session:
                    self.session = session
                    self.reconnect_count = 0
                    
                    while self.running:
                        try:
                            # Get order book from Coinbase Derivatives
                            url = f"{self.api_base_url}/products/{symbol}/book"
                            
                            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                                if response.status == 200:
                                    data = await response.json()
                                    
                                    depth = self._parse_depth(data, symbol)
                                    if depth:
                                        await queue.put(depth)
                                    
                                elif response.status == 404:
                                    logger.warning(f"Symbol {symbol} not found on Coinbase Derivatives")
                                    break
                                else:
                                    logger.warning(f"HTTP {response.status} for {symbol} book")
                            
                            # Poll every 2 seconds for order book
                            await asyncio.sleep(2.0)
                            
                        except asyncio.TimeoutError:
                            logger.warning(f"Timeout fetching depth for {symbol}")
                        except Exception as e:
                            logger.error(f"Error fetching depth for {symbol}: {e}")
                            await asyncio.sleep(self.reconnect_delay)
                            
            except Exception as e:
                logger.error(f"Coinbase Derivatives REST error: {e}")
                self.reconnect_count += 1
                
                if self.reconnect_count >= self.max_reconnect_attempts:
                    logger.error(f"Max reconnection attempts reached for {symbol}")
                    break
                    
                logger.info(f"Reconnecting in {self.reconnect_delay} seconds...")
                await asyncio.sleep(self.reconnect_delay)
                self.reconnect_delay = min(self.reconnect_delay * 2, 60)
    
    def _parse_trade(self, data: Dict, symbol: str) -> Optional[Trade]:
        """Parse trade data from Coinbase Derivatives REST API."""
        try:
            # Extract symbol without -PERP suffix
            clean_symbol = symbol.replace('-PERP', '')
            
            # Parse trade data (format may vary based on actual API response)
            price = float(data.get('price', 0))
            size = float(data.get('size', 0))
            
            # Determine side based on trade data
            side = 'buy' if data.get('side') == 'buy' else 'sell'
            
            return Trade(
                timestamp=datetime.now(),
                price=price,
                size=size,
                side=side,
                symbol=clean_symbol
            )
        except Exception as e:
            logger.error(f"Error parsing Coinbase Derivatives trade: {e}")
            return None
    
    def _parse_depth(self, data: Dict, symbol: str) -> Optional[Dict]:
        """Parse order book depth data from Coinbase Derivatives REST API."""
        try:
            # Extract symbol without -PERP suffix
            clean_symbol = symbol.replace('-PERP', '')
            
            # Parse bid and ask levels
            bid_levels = []
            ask_levels = []
            
            bids = data.get('bids', [])
            for bid in bids:
                level = {'price': float(bid[0]), 'quantity': float(bid[1])}
                bid_levels.append(level)
            
            asks = data.get('asks', [])
            for ask in asks:
                level = {'price': float(ask[0]), 'quantity': float(ask[1])}
                ask_levels.append(level)
            
            return {
                'symbol': clean_symbol,
                'bid_levels': bid_levels,
                'ask_levels': ask_levels,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error parsing Coinbase Derivatives depth: {e}")
            return None
    
    def start(self) -> None:
        """Start the REST API connector."""
        self.running = True
        logger.info("Coinbase Derivatives REST API connector started")
    
    def stop(self) -> None:
        """Stop the REST API connector."""
        self.running = False
        if self.session and not self.session.closed:
            asyncio.create_task(self.session.close())
        logger.info("Coinbase Derivatives REST API connector stopped")

class CoinbaseAdvancedWS:
    """Coinbase Advanced Trade WebSocket connector using official SDK."""
    
    def __init__(self, max_reconnect_attempts: int = 10, reconnect_delay: float = 1.0):
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.running = False
        self.reconnect_count = 0
        self.client = None
        self.trade_queue = None
        self.depth_queue = None
        
    async def stream_trades(self, symbol: str, queue: asyncio.Queue) -> None:
        """
        Stream trade data for a Coinbase Advanced Trade symbol using WebSocket.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC-USD')
            queue: asyncio.Queue to push parsed trade data
        """
        from coinbase.websocket import WSClient
        
        self.trade_queue = queue
        
        def on_message(msg):
            try:
                import json
                data = json.loads(msg)
                
                # Log all incoming messages for debugging
                logger.info(f"🔍 Coinbase Advanced API Data: {msg}")
                
                # Handle ticker messages (price updates)
                if data.get('channel') == 'ticker':
                    events = data.get('events', [])
                    for event in events:
                        tickers = event.get('tickers', [])
                        for ticker in tickers:
                            if ticker.get('product_id') == symbol:
                                logger.info(f"📊 Processing ticker for {symbol}: {ticker}")
                                trade = self._parse_ticker(ticker)
                                if trade:
                                    logger.info(f"✅ Created trade: {trade.symbol} = ${trade.price}")
                                    # Use asyncio to put in queue from sync context
                                    asyncio.create_task(queue.put(trade))
                                    
            except Exception as e:
                logger.error(f"Error processing Coinbase Advanced message: {e}")
        
        while self.running:
            try:
                logger.info(f"Starting Coinbase Advanced WebSocket for trades: {symbol}")
                
                # Initialize WebSocket client (no auth needed for public channels)
                self.client = WSClient(on_message=on_message, verbose=True)
                
                # Open connection and subscribe to ticker channel
                self.client.open()
                self.client.subscribe(product_ids=[symbol], channels=["ticker"])
                
                logger.info(f"Connected to Coinbase Advanced WebSocket for trades: {symbol}")
                
                # Keep connection alive
                while self.running:
                    await asyncio.sleep(1.0)
                    
            except Exception as e:
                logger.error(f"Coinbase Advanced WebSocket error: {e}")
                self.reconnect_count += 1
                
                if self.reconnect_count >= self.max_reconnect_attempts:
                    logger.error(f"Max reconnection attempts reached for {symbol}")
                    break
                    
                logger.info(f"Reconnecting in {self.reconnect_delay} seconds...")
                await asyncio.sleep(self.reconnect_delay)
                self.reconnect_delay = min(self.reconnect_delay * 2, 60)
            finally:
                if self.client:
                    try:
                        self.client.close()
                    except:
                        pass
    
    async def stream_depth(self, symbol: str, queue: asyncio.Queue) -> None:
        """
        Stream order book depth data for a Coinbase Advanced Trade symbol using WebSocket.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC-USD')
            queue: asyncio.Queue to push parsed depth data
        """
        from coinbase.websocket import WSClient
        
        self.depth_queue = queue
        
        def on_message(msg):
            try:
                import json
                data = json.loads(msg)
                
                # Handle level2 messages (order book updates)
                if data.get('channel') == 'level2':
                    events = data.get('events', [])
                    for event in events:
                        updates = event.get('updates', [])
                        for update in updates:
                            if update.get('product_id') == symbol:
                                depth = self._parse_level2_update(update)
                                if depth:
                                    # Use asyncio to put in queue from sync context
                                    asyncio.create_task(queue.put(depth))
                                    
            except Exception as e:
                logger.error(f"Error processing Coinbase Advanced depth message: {e}")
        
        while self.running:
            try:
                logger.info(f"Starting Coinbase Advanced WebSocket for depth: {symbol}")
                
                # Initialize WebSocket client (no auth needed for public channels)
                self.client = WSClient(on_message=on_message, verbose=True)
                
                # Open connection and subscribe to level2 channel
                self.client.open()
                self.client.subscribe(product_ids=[symbol], channels=["level2"])
                
                logger.info(f"Connected to Coinbase Advanced WebSocket for depth: {symbol}")
                
                # Keep connection alive
                while self.running:
                    await asyncio.sleep(1.0)
                    
            except Exception as e:
                logger.error(f"Coinbase Advanced WebSocket error: {e}")
                self.reconnect_count += 1
                
                if self.reconnect_count >= self.max_reconnect_attempts:
                    logger.error(f"Max reconnection attempts reached for {symbol}")
                    break
                    
                logger.info(f"Reconnecting in {self.reconnect_delay} seconds...")
                await asyncio.sleep(self.reconnect_delay)
                self.reconnect_delay = min(self.reconnect_delay * 2, 60)
            finally:
                if self.client:
                    try:
                        self.client.close()
                    except:
                        pass
    
    def _parse_ticker(self, ticker_data: Dict) -> Optional[Trade]:
        """Parse ticker data from Coinbase Advanced Trade WebSocket."""
        try:
            # Debug: Log the actual ticker data structure
            logger.debug(f"Coinbase Advanced ticker data: {ticker_data}")
            
            symbol = ticker_data.get('product_id', '').replace('-USD', '')
            
            # Try different possible price fields
            price = 0.0
            if 'price' in ticker_data:
                price = float(ticker_data.get('price', 0))
            elif 'last_price' in ticker_data:
                price = float(ticker_data.get('last_price', 0))
            elif 'trade_price' in ticker_data:
                price = float(ticker_data.get('trade_price', 0))
            
            # If still no price, log available fields
            if price == 0.0:
                logger.warning(f"No price found in ticker data. Available fields: {list(ticker_data.keys())}")
                return None
            
            # For ticker data, we don't have size/side info, so we'll use a default
            # This creates synthetic trade data for order flow analysis
            size = 1.0  # Default size
            side = 'buy'  # Default side
            
            logger.debug(f"Parsed ticker: {symbol} = ${price}")
            
            return Trade(
                timestamp=datetime.now(),
                price=price,
                size=size,
                side=side,
                symbol=symbol
            )
        except Exception as e:
            logger.error(f"Error parsing Coinbase Advanced ticker: {e}")
            return None
    
    def _parse_level2_update(self, update_data: Dict) -> Optional[Dict]:
        """Parse level2 order book update from Coinbase Advanced Trade WebSocket."""
        try:
            symbol = update_data.get('product_id', '').replace('-USD', '')
            
            # Parse bid and ask levels
            bid_levels = []
            ask_levels = []
            
            # Level2 updates contain individual price level changes
            side = update_data.get('side', '')
            price = float(update_data.get('price', 0))
            size = float(update_data.get('size', 0))
            
            level = {'price': price, 'quantity': size}
            
            if side == 'bid':
                bid_levels.append(level)
            elif side == 'ask':
                ask_levels.append(level)
            
            return {
                'symbol': symbol,
                'bid_levels': bid_levels,
                'ask_levels': ask_levels,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error parsing Coinbase Advanced level2 update: {e}")
            return None
    
    def start(self) -> None:
        """Start the WebSocket connector."""
        self.running = True
        logger.info("Coinbase Advanced WebSocket connector started")
    
    def stop(self) -> None:
        """Stop the WebSocket connector."""
        self.running = False
        if self.client:
            try:
                self.client.close()
            except:
                pass
        logger.info("Coinbase Advanced WebSocket connector stopped")
