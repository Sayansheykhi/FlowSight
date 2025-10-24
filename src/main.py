#!/usr/bin/env python3
"""
Multi-Symbol Order Flow Engine Orchestrator
Coordinates the entire pipeline for multiple symbols: data feeds → metrics → signals → alerts.
"""

import asyncio
import json
import logging
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import time
import os

# Import core modules
from core.feeds import BinanceFuturesWS, CoinbaseFuturesWS, CoinbaseDerivativesWS, CoinbaseDerivativesREST, CoinbaseAdvancedWS, Trade
from core.metrics import (
    compute_cvd, compute_imbalance, compute_vwap, detect_absorption,
    AdvancedOrderFlowMetrics
)
from core.signals import generate_signal, SessionState, update_session_state
from core.utils import in_session, ema, setup_logging, load_config
from core.coinbase_discovery import get_available_gold_futures

# Import Telegram for alerts
try:
    from telegram import Bot
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    Bot = None

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class OrderBookSnapshot:
    """Order book snapshot structure."""
    timestamp: datetime
    bids: List[Dict[str, float]]  # [{'price': float, 'quantity': float}]
    asks: List[Dict[str, float]]

@dataclass
class SignalOutput:
    """Structured signal output."""
    timestamp: str
    symbol: str
    signal: str  # 'LONG', 'SHORT', 'WAIT'
    confidence: int  # 0-100
    price: float
    metrics: Dict[str, float]
    session_active: bool
    reasoning: List[str]

class SymbolState:
    """Independent state management for each symbol."""
    
    def __init__(self, symbol: str, config: Dict[str, Any], telegram_callback=None):
        self.symbol = symbol
        self.config = config
        self.telegram_callback = telegram_callback
        
        # Determine which exchange to use based on symbol format
        self.exchange = self._determine_exchange(symbol)
        
        # WebSocket connections
        if self.exchange == 'binance':
            self.binance_ws = BinanceFuturesWS()
            self.coinbase_ws = None
            self.coinbase_futures_ws = None
            self.coinbase_derivatives_rest = None
            self.coinbase_advanced_ws = None
        elif self.exchange == 'coinbase':
            self.coinbase_ws = CoinbaseFuturesWS()
            self.binance_ws = None
            self.coinbase_futures_ws = None
            self.coinbase_derivatives_rest = None
            self.coinbase_advanced_ws = None
        elif self.exchange == 'coinbase_futures':
            self.coinbase_futures_ws = CoinbaseDerivativesWS()
            self.binance_ws = None
            self.coinbase_ws = None
            self.coinbase_derivatives_rest = None
            self.coinbase_advanced_ws = None
        elif self.exchange == 'coinbase_derivatives':
            self.coinbase_derivatives_rest = CoinbaseDerivativesREST()
            self.binance_ws = None
            self.coinbase_ws = None
            self.coinbase_futures_ws = None
            self.coinbase_advanced_ws = None
        elif self.exchange == 'coinbase_advanced':
            self.coinbase_advanced_ws = CoinbaseAdvancedWS()
            self.binance_ws = None
            self.coinbase_ws = None
            self.coinbase_futures_ws = None
            self.coinbase_derivatives_rest = None
        else:
            # Default to Binance
            self.binance_ws = BinanceFuturesWS()
            self.coinbase_ws = None
            self.coinbase_futures_ws = None
            self.coinbase_derivatives_rest = None
            self.coinbase_advanced_ws = None
        
        # Metrics calculator
        self.metrics_calculator = AdvancedOrderFlowMetrics(
            ema_alpha_cvd=0.1,  # EMA smoothing factor (0.1 = 10% weight to new data)
            ema_alpha_vwap=0.05  # EMA smoothing factor for VWAP
        )
        
        # Data storage
        self.trade_queue = asyncio.Queue()
        self.depth_queue = asyncio.Queue()
        self.trades_bucket = []  # 1-second bucket
        self.depth_bucket = []   # 1-second bucket
        
        # Session state
        self.session_state = SessionState(start_time=datetime.now())
        
        # Latest data
        self.latest_price = 0.0
        self.latest_signal = 'WAIT'
        self.latest_confidence = 0
        
        logger.info(f"SymbolState initialized for {symbol} on {self.exchange}")
    
    def start(self):
        """Start the WebSocket connections for this symbol."""
        logger.info(f"Starting WebSocket for {self.symbol} on {self.exchange}")
        if self.exchange == 'binance' and self.binance_ws:
            logger.info(f"Starting Binance WebSocket for {self.symbol}")
            self.binance_ws.start()
        elif self.exchange == 'coinbase' and self.coinbase_ws:
            logger.info(f"Starting Coinbase WebSocket for {self.symbol}")
            self.coinbase_ws.start()
        elif self.exchange == 'coinbase_futures' and self.coinbase_futures_ws:
            logger.info(f"Starting Coinbase Futures WebSocket for {self.symbol}")
            self.coinbase_futures_ws.start()
        elif self.exchange == 'coinbase_derivatives' and self.coinbase_derivatives_rest:
            logger.info(f"Starting Coinbase Derivatives REST API for {self.symbol}")
            self.coinbase_derivatives_rest.start()
        elif self.exchange == 'coinbase_advanced' and self.coinbase_advanced_ws:
            logger.info(f"Starting Coinbase Advanced WebSocket for {self.symbol}")
            self.coinbase_advanced_ws.start()
        else:
            logger.warning(f"No WebSocket available for {self.symbol} on {self.exchange}")
    
    def _determine_exchange(self, symbol: str) -> str:
        """Determine which exchange to use based on symbol format."""
        if '-' in symbol and symbol.endswith('-USD'):
            # Check if it's a Coinbase Advanced symbol (spot or derivatives)
            if symbol in ['SOL-USD', 'ADA-USD', 'DOGE-USD', 'MATIC-USD', 'XAU-USD', 'XAG-USD', 'BTC-USD', 'ETH-USD']:
                return 'coinbase_advanced'
            else:
                return 'coinbase'
        elif symbol.endswith('-PERP'):
            # All PERP symbols go to Coinbase Advanced (derivatives)
            return 'coinbase_advanced'
        elif symbol.startswith('GOL') and len(symbol) == 8:  # Gold futures like GOLZ2025
            return 'binance'  # Gold futures are available on Binance, not Coinbase
        elif symbol.endswith('usdt') or (symbol.isupper() and len(symbol) > 6):
            # Binance symbols: usdt pairs or futures contracts (including Gold futures)
            return 'binance'
        else:
            return 'binance'  # Default to Binance
    
    async def stream_trades_task(self):
        """Async task to stream trades for this symbol."""
        logger.info(f"Starting trade stream for {self.symbol} on {self.exchange}")
        try:
            if self.exchange == 'binance' and self.binance_ws:
                await self.binance_ws.stream_trades(self.symbol, self.trade_queue)
            elif self.exchange == 'coinbase' and self.coinbase_ws:
                await self.coinbase_ws.stream_trades(self.symbol, self.trade_queue)
            elif self.exchange == 'coinbase_futures' and self.coinbase_futures_ws:
                await self.coinbase_futures_ws.stream_trades(self.symbol, self.trade_queue)
            elif self.exchange == 'coinbase_derivatives' and self.coinbase_derivatives_rest:
                await self.coinbase_derivatives_rest.stream_trades(self.symbol, self.trade_queue)
            elif self.exchange == 'coinbase_advanced' and self.coinbase_advanced_ws:
                await self.coinbase_advanced_ws.stream_trades(self.symbol, self.trade_queue)
        except Exception as e:
            logger.error(f"Error in trade stream for {self.symbol}: {e}")
    
    async def stream_depth_task(self):
        """Async task to stream order book depth for this symbol."""
        logger.info(f"Starting depth stream for {self.symbol} on {self.exchange}")
        try:
            if self.exchange == 'binance' and self.binance_ws:
                await self.binance_ws.stream_depth(self.symbol, self.depth_queue)
            elif self.exchange == 'coinbase' and self.coinbase_ws:
                await self.coinbase_ws.stream_depth(self.symbol, self.depth_queue)
            elif self.exchange == 'coinbase_futures' and self.coinbase_futures_ws:
                await self.coinbase_futures_ws.stream_depth(self.symbol, self.depth_queue)
            elif self.exchange == 'coinbase_derivatives' and self.coinbase_derivatives_rest:
                await self.coinbase_derivatives_rest.stream_depth(self.symbol, self.depth_queue)
            elif self.exchange == 'coinbase_advanced' and self.coinbase_advanced_ws:
                await self.coinbase_advanced_ws.stream_depth(self.symbol, self.depth_queue)
        except Exception as e:
            logger.error(f"Error in depth stream for {self.symbol}: {e}")
    
    async def aggregate_data_task(self):
        """Aggregate data into 1-second buckets for this symbol."""
        logger.info(f"Starting data aggregation task for {self.symbol}")
        
        while True:
            try:
                # Process trades
                while not self.trade_queue.empty():
                    try:
                        trade_data = await asyncio.wait_for(
                            self.trade_queue.get(), timeout=0.1
                        )
                        if trade_data:
                            self.trades_bucket.append(trade_data)
                    except asyncio.TimeoutError:
                        break
                
                # Process depth data
                while not self.depth_queue.empty():
                    try:
                        depth_data = await asyncio.wait_for(
                            self.depth_queue.get(), timeout=0.1
                        )
                        if depth_data:
                            self.depth_bucket.append(depth_data)
                    except asyncio.TimeoutError:
                        break
                
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Error in data aggregation for {self.symbol}: {e}")
                await asyncio.sleep(1)
    
    async def process_bucket_task(self):
        """Process 1-second buckets and generate signals for this symbol."""
        logger.info(f"Starting bucket processing task for {self.symbol}")
        
        while True:
            try:
                # Wait for 1 second
                await asyncio.sleep(1.0)
                
                # Process if we have data
                if self.trades_bucket or self.depth_bucket:
                    await self._process_current_bucket()
                
            except Exception as e:
                logger.error(f"Error in bucket processing for {self.symbol}: {e}")
                await asyncio.sleep(1)
    
    async def _process_current_bucket(self):
        """Process current 1-second bucket for this symbol."""
        try:
            current_time = datetime.now()
            
            # Convert trade data
            trades = []
            for trade_data in self.trades_bucket:
                try:
                    # Check if trade_data is already a Trade object (from Coinbase Advanced)
                    if isinstance(trade_data, Trade):
                        trades.append(trade_data)
                    else:
                        # Convert dictionary to Trade object (from Binance/other exchanges)
                        trade = Trade(
                            timestamp=datetime.fromtimestamp(trade_data['ts'] / 1000),
                            price=float(trade_data['price']),
                            size=float(trade_data['qty']),
                            side=str(trade_data['side']),
                            symbol=str(trade_data.get('symbol', self.symbol))
                        )
                        trades.append(trade)
                except (KeyError, ValueError, TypeError) as e:
                    logger.error(f"Error creating Trade object: {e}")
                    continue
            
            # Convert depth data (use latest snapshot)
            depth = {}
            if self.depth_bucket:
                latest_depth = self.depth_bucket[-1]
                depth = {
                    'bids': latest_depth.get('bid_levels', []),
                    'asks': latest_depth.get('ask_levels', [])
                }
            
            # Clear buckets
            self.trades_bucket.clear()
            self.depth_bucket.clear()
            
            # Compute metrics
            metrics = self.metrics_calculator.compute_all_advanced_metrics(trades, depth)
            
            # Update session state
            total_volume = sum(trade.size for trade in trades)
            self.session_state = update_session_state(
                self.session_state, 
                trade_volume=total_volume,
                signal_generated=False
            )
            
            # Generate signal
            signal_type, confidence = generate_signal(metrics, self.session_state)
            
            # Update session state with signal
            self.session_state = update_session_state(
                self.session_state,
                signal_generated=True
            )
            
            # Get current price
            current_price = trades[-1].price if trades else self.latest_price
            self.latest_price = current_price
            self.latest_signal = signal_type
            self.latest_confidence = confidence
            
            # Check if in trading session
            session_active = in_session(current_time)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(metrics, signal_type, confidence)
            
            # Create signal output
            signal_output = SignalOutput(
                timestamp=current_time.isoformat(),
                symbol=self.symbol,
                signal=signal_type,
                confidence=confidence,
                price=current_price,
                metrics=metrics,
                session_active=session_active,
                reasoning=reasoning
            )
            
            # Output structured JSON
            await self._output_signal(signal_output)
            
        except Exception as e:
            logger.error(f"Error processing bucket for {self.symbol}: {e}")
    
    def _generate_reasoning(self, metrics: Dict[str, float], signal_type: str, confidence: int) -> List[str]:
        """Generate reasoning for the signal."""
        reasoning = []
        
        cvd = metrics.get('cvd', 0.0)
        imbalance = metrics.get('imbalance', 0.0)
        vwap = metrics.get('vwap', 0.0)
        absorption = metrics.get('absorption', 0.0)
        price = metrics.get('price', 0.0)
        
        # CVD reasoning
        if cvd > 0.1:
            reasoning.append(f"CVD rising (+{cvd:.3f}) - buying pressure")
        elif cvd < -0.1:
            reasoning.append(f"CVD falling ({cvd:.3f}) - selling pressure")
        
        # Imbalance reasoning
        threshold = self.config.get('imbalance_threshold', 0.15)
        if imbalance > threshold:
            reasoning.append(f"Strong bid imbalance (+{imbalance:.3f})")
        elif imbalance < -threshold:
            reasoning.append(f"Strong ask imbalance ({imbalance:.3f})")
        
        # Price vs VWAP reasoning
        if vwap > 0:
            price_vs_vwap = ((price / vwap) - 1) * 100
            if price_vs_vwap > 0.5:
                reasoning.append(f"Price {price_vs_vwap:.2f}% above VWAP")
            elif price_vs_vwap < -0.5:
                reasoning.append(f"Price {price_vs_vwap:.2f}% below VWAP")
        
        # Absorption reasoning
        if absorption > 0.3:
            reasoning.append(f"High absorption ({absorption:.3f}) - institutional activity")
        
        # Session reasoning
        if not in_session(datetime.now()):
            reasoning.append("Outside trading sessions")
        
        # Confidence reasoning
        if confidence >= 80:
            reasoning.append("High confidence signal")
        elif confidence >= 60:
            reasoning.append("Medium confidence signal")
        else:
            reasoning.append("Low confidence signal")
        
        return reasoning
    
    async def _output_signal(self, signal_output: SignalOutput):
        """Output structured JSON signal to console."""
        try:
            # Convert to dictionary for JSON serialization
            signal_dict = asdict(signal_output)
            
            # Format for console output
            output = {
                "timestamp": signal_dict["timestamp"],
                "symbol": signal_dict["symbol"],
                "signal": signal_dict["signal"],
                "confidence": signal_dict["confidence"],
                "price": signal_dict["price"],
                "metrics": {
                    "cvd": round(signal_dict["metrics"]["cvd"], 4),
                    "imbalance": round(signal_dict["metrics"]["imbalance"], 4),
                    "vwap": round(signal_dict["metrics"]["vwap"], 2),
                    "absorption": round(signal_dict["metrics"]["absorption"], 4)
                },
                "session_active": signal_dict["session_active"],
                "reasoning": signal_dict["reasoning"]
            }
            
            # Print structured JSON (one line per symbol)
            print(json.dumps(output))
            
            # Save signal to individual log file
            self._save_signal_to_file(signal_output.symbol, output)
            
            # Log signal with emoji
            signal_emoji = "🟩" if signal_output.signal == "LONG" else "🟥" if signal_output.signal == "SHORT" else "⚪"
            logger.info(f"{signal_emoji} {signal_output.symbol}: {signal_output.signal} | "
                       f"Confidence: {signal_output.confidence}% | "
                       f"Price: ${signal_output.price:.2f}")
            
            # Send Telegram alert via callback if configured
            if self.telegram_callback:
                await self.telegram_callback(signal_output)
            
            # Return signal output for Telegram alerts
            return signal_output
            
        except Exception as e:
            logger.error(f"Error outputting signal for {self.symbol}: {e}")
            return None
    
    def _save_signal_to_file(self, symbol: str, signal_data: Dict[str, Any]):
        """Save signal to individual log file for the symbol."""
        try:
            # Create signals directory if it doesn't exist
            signals_dir = "signals"
            if not os.path.exists(signals_dir):
                os.makedirs(signals_dir)
            
            # Create filename: signals_{symbol}.log
            filename = f"signals_{symbol.lower()}.log"
            filepath = os.path.join(signals_dir, filename)
            
            # Append signal data to file
            with open(filepath, "a") as f:
                f.write(json.dumps(signal_data) + "\n")
            
            logger.debug(f"Signal saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving signal to file for {symbol}: {e}")
    
    def stop(self):
        """Stop WebSocket connections for this symbol."""
        if self.exchange == 'binance' and self.binance_ws:
            self.binance_ws.stop()
        elif self.exchange == 'coinbase' and self.coinbase_ws:
            self.coinbase_ws.stop()

class MultiSymbolOrderFlowOrchestrator:
    """Main orchestrator for multiple symbols."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        
        # Setup logging
        setup_logging(
            log_level=self.config.get('logging', {}).get('level', 'INFO'),
            log_file=self.config.get('logging', {}).get('file')
        )
        
        # Initialize symbols
        self.symbols = self._get_symbols()
        self.symbol_states = {}
        
        # Control flags
        self.running = False
        self.tasks = []
        
        # Telegram bot
        self.telegram_bot = None
        self._setup_telegram()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"MultiSymbolOrderFlowOrchestrator initialized for symbols: {self.symbols}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            config = load_config(self.config_path)
            if not config:
                # Create default config
                config = {
                    'symbols': ['btcusdt', 'ethusdt'],
                    'telegram': {
                        'enabled': False,
                        'token': '',
                        'chat_id': ''
                    },
                    'logging': {
                        'level': 'INFO',
                        'file': 'logs/orderflow.log'
                    }
                }
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {
                'symbols': ['btcusdt', 'ethusdt'],
                'telegram': {'enabled': False},
                'logging': {'level': 'INFO'}
            }
    
    def _get_symbols(self) -> List[str]:
        """Get symbols from config, supporting multiple exchanges and automatic Gold futures discovery."""
        symbols = []
        
        # Check for new exchange-based configuration
        exchanges_config = self.config.get('exchanges', {})
        if exchanges_config:
            # Binance symbols
            binance_config = exchanges_config.get('binance', {})
            if binance_config.get('enabled', True):
                binance_symbols = binance_config.get('symbols', [])
                
                # Check if we should auto-discover Gold futures for Binance
                auto_discover_gold = binance_config.get('auto_discover_gold_futures', True)
                
                if auto_discover_gold:
                    try:
                        # Discover Gold futures using a thread executor to avoid event loop conflicts
                        import concurrent.futures
                        import asyncio
                        
                        def run_discovery():
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            try:
                                return loop.run_until_complete(get_available_gold_futures())
                            finally:
                                loop.close()
                        
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(run_discovery)
                            gold_futures = future.result(timeout=10)  # 10 second timeout
                        
                        # Add discovered Gold futures to Binance symbols
                        binance_symbols.extend(gold_futures)
                        logger.info(f"Auto-discovered Gold futures for Binance: {gold_futures}")
                        
                    except Exception as e:
                        logger.error(f"Failed to auto-discover Gold futures: {e}")
                        logger.info("Using hardcoded Gold futures as fallback")
                
                symbols.extend(binance_symbols)
            
            # Coinbase symbols
            coinbase_config = exchanges_config.get('coinbase', {})
            if coinbase_config.get('enabled', False):
                coinbase_symbols = coinbase_config.get('symbols', [])
                symbols.extend(coinbase_symbols)
            
            # Coinbase Futures symbols
            coinbase_futures_config = exchanges_config.get('coinbase_futures', {})
            if coinbase_futures_config.get('enabled', False):
                coinbase_futures_symbols = coinbase_futures_config.get('symbols', [])
                symbols.extend(coinbase_futures_symbols)
            
            # Coinbase Derivatives symbols
            coinbase_derivatives_config = exchanges_config.get('coinbase_derivatives', {})
            if coinbase_derivatives_config.get('enabled', False):
                coinbase_derivatives_symbols = coinbase_derivatives_config.get('symbols', [])
                symbols.extend(coinbase_derivatives_symbols)
            
            # Coinbase Advanced symbols
            coinbase_advanced_config = exchanges_config.get('coinbase_advanced', {})
            if coinbase_advanced_config.get('enabled', False):
                coinbase_advanced_symbols = coinbase_advanced_config.get('symbols', [])
                symbols.extend(coinbase_advanced_symbols)
        
        # Fallback to old configuration format
        if not symbols:
            if 'symbols' in self.config:
                symbols = self.config['symbols']
            elif 'symbol' in self.config:
                symbols = [self.config['symbol']]
            else:
                symbols = ['btcusdt', 'ethusdt']  # Default symbols
        
        # Remove duplicates while preserving order
        seen = set()
        unique_symbols = []
        for symbol in symbols:
            if symbol not in seen:
                seen.add(symbol)
                unique_symbols.append(symbol)
        
        return unique_symbols
    
    def _setup_telegram(self):
        """Setup Telegram bot if configured."""
        telegram_config = self.config.get('telegram', {})
        if telegram_config.get('enabled', False) and TELEGRAM_AVAILABLE:
            bot_token = telegram_config.get('token') or telegram_config.get('bot_token')
            chat_id = telegram_config.get('chat_id')
            
            if bot_token and chat_id:
                try:
                    self.telegram_bot = Bot(token=bot_token)
                    self.telegram_chat_id = chat_id
                    logger.info("Telegram bot initialized successfully")
                    logger.info(f"Bot token: {bot_token[:10]}...")
                    logger.info(f"Chat ID: {chat_id}")
                except Exception as e:
                    logger.error(f"Error initializing Telegram bot: {e}")
                    self.telegram_bot = None
            else:
                logger.warning("Telegram enabled but missing token or chat_id")
                self.telegram_bot = None
        elif telegram_config.get('enabled', False) and not TELEGRAM_AVAILABLE:
            logger.warning("Telegram enabled but python-telegram-bot not installed")
            self.telegram_bot = None
        else:
            self.telegram_bot = None
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
    
    async def _send_telegram_alert(self, signal_output: SignalOutput):
        """Send Telegram alert for high confidence signals."""
        try:
            if not self.telegram_bot or not hasattr(self, 'telegram_chat_id'):
                logger.warning("Telegram bot not properly configured")
                return
            
            # Check confidence threshold from config
            confidence_threshold = self.config.get('signals', {}).get('confidence_threshold', 70)
            if signal_output.confidence < confidence_threshold:
                logger.debug(f"Skipping Telegram alert: confidence={signal_output.confidence} < threshold={confidence_threshold}")
                return
            
            # Format message with enhanced styling
            message = self._format_telegram_message(signal_output)
            
            # Send message
            await self.telegram_bot.send_message(
                chat_id=self.telegram_chat_id, 
                text=message, 
                parse_mode='Markdown',
                disable_web_page_preview=True
            )
            
            logger.info(f"✅ Sent Telegram alert for {signal_output.symbol}: {signal_output.signal} ({signal_output.confidence}%)")
            
        except Exception as e:
            logger.error(f"❌ Error sending Telegram alert: {e}")
    
    def _format_telegram_message(self, signal_output: SignalOutput) -> str:
        """Format signal for Telegram message with enhanced styling."""
        # Signal emoji and styling
        signal_emoji = "🟢" if signal_output.signal == "LONG" else "🔴"
        signal_style = "🟢 *LONG*" if signal_output.signal == "LONG" else "🔴 *SHORT*"
        
        # Confidence styling
        confidence_emoji = "🔥" if signal_output.confidence >= 85 else "⚡" if signal_output.confidence >= 75 else "📊"
        
        # Session emoji
        session_emoji = "🟢" if signal_output.session_active else "🔴"
        
        # Price change indicator
        price_indicator = "📈" if signal_output.signal == "LONG" else "📉"
        
        # Format metrics with emojis
        cvd_emoji = "📊" if abs(signal_output.metrics['cvd']) < 0.1 else "🚀" if signal_output.metrics['cvd'] > 0 else "💥"
        imbalance_emoji = "⚖️" if abs(signal_output.metrics['imbalance']) < 0.1 else "⬆️" if signal_output.metrics['imbalance'] > 0 else "⬇️"
        vwap_emoji = "📊"
        absorption_emoji = "🏢" if signal_output.metrics['absorption'] > 0.3 else "👤"
        
        # Format timestamp
        timestamp = datetime.fromisoformat(signal_output.timestamp.replace('Z', '+00:00'))
        time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')
        
        # Create formatted message
        message = f"""
{signal_emoji} *ORDER FLOW ALERT* {signal_emoji}

🎯 *Signal:* {signal_style}
{confidence_emoji} *Confidence:* {signal_output.confidence}%
💰 *Symbol:* `{signal_output.symbol}`
{price_indicator} *Price:* `${signal_output.price:,.2f}`
{session_emoji} *Session:* {'Active' if signal_output.session_active else 'Closed'}

📊 *Order Flow Metrics:*
{cvd_emoji} CVD: `{signal_output.metrics['cvd']:+.4f}`
{imbalance_emoji} Imbalance: `{signal_output.metrics['imbalance']:+.4f}`
{vwap_emoji} VWAP: `${signal_output.metrics['vwap']:,.2f}`
{absorption_emoji} Absorption: `{signal_output.metrics['absorption']:.4f}`

🧠 *Analysis:*
{chr(10).join(f"• {reason}" for reason in signal_output.reasoning)}

⏰ *Time:* {time_str}
🤖 *OrderFlow Engine*
        """.strip()
        
        return message
    
    async def run(self):
        """Main run loop for multiple symbols."""
        logger.info("Starting Multi-Symbol Order Flow Engine Orchestrator...")
        
        # Initialize symbol states
        for symbol in self.symbols:
            # Create Telegram callback function
            telegram_callback = lambda signal_output: self._send_telegram_alert(signal_output)
            self.symbol_states[symbol] = SymbolState(symbol, self.config, telegram_callback)
            self.symbol_states[symbol].start()
        
        self.running = True
        
        try:
            # Create async tasks for each symbol
            for symbol in self.symbols:
                symbol_state = self.symbol_states[symbol]
                
                # Create tasks for this symbol
                symbol_tasks = [
                    asyncio.create_task(symbol_state.stream_trades_task()),
                    asyncio.create_task(symbol_state.stream_depth_task()),
                    asyncio.create_task(symbol_state.aggregate_data_task()),
                    asyncio.create_task(symbol_state.process_bucket_task())
                ]
                
                self.tasks.extend(symbol_tasks)
            
            logger.info(f"All tasks started successfully for {len(self.symbols)} symbols")
            
            # Wait for all tasks
            await asyncio.gather(*self.tasks, return_exceptions=True)
            
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            await self._shutdown()
    
    async def _shutdown(self):
        """Graceful shutdown."""
        logger.info("Initiating graceful shutdown...")
        
        # Stop running
        self.running = False
        
        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Stop WebSocket connections for all symbols
        for symbol_state in self.symbol_states.values():
            symbol_state.stop()
        
        logger.info("Multi-Symbol Order Flow Engine Orchestrator stopped")

async def main():
    """Main entry point."""
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    
    orchestrator = MultiSymbolOrderFlowOrchestrator(config_path)
    await orchestrator.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)