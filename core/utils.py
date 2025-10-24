"""
Utility functions for the order flow engine.
Includes data processing, logging, and helper functions.
"""

import json
import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import pytz

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logging.error(f"Error loading config from {config_path}: {e}")
        return {}

def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to YAML file."""
    try:
        with open(config_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
    except Exception as e:
        logging.error(f"Error saving config to {config_path}: {e}")

def create_default_config() -> Dict[str, Any]:
    """Create default configuration."""
    return {
        'exchanges': {
            'binance': {
                'enabled': True,
                'symbols': ['BTC', 'ETH', 'ADA', 'SOL', 'MATIC']
            },
            'coinbase': {
                'enabled': False,
                'symbols': ['BTC', 'ETH']
            }
        },
        'signals': {
            'cvd_threshold': 0.3,
            'imbalance_threshold': 0.2,
            'absorption_threshold': 0.4,
            'confidence_threshold': 0.6,
            'min_volume_threshold': 1000
        },
        'alerts': {
            'telegram': {
                'enabled': False,
                'bot_token': '',
                'chat_id': ''
            },
            'webhook': {
                'enabled': False,
                'url': ''
            }
        },
        'dashboard': {
            'enabled': True,
            'port': 8050,
            'host': '127.0.0.1'
        },
        'data': {
            'window_size': 100,
            'update_interval': 1.0  # seconds
        }
    }

def format_signal_for_json(signal) -> Dict[str, Any]:
    """Format trading signal for JSON serialization."""
    return {
        "timestamp": signal.timestamp.isoformat(),
        "symbol": signal.symbol,
        "signal": signal.signal_type.value,
        "confidence": round(signal.confidence, 3),
        "price": round(signal.price, 4),
        "risk_score": round(signal.risk_score, 3),
        "reasoning": signal.reasoning,
        "metrics": {
            "cvd": round(signal.metrics.cvd, 2) if signal.metrics else None,
            "imbalance": round(signal.metrics.imbalance, 2) if signal.metrics else None,
            "absorption": round(signal.metrics.absorption, 2) if signal.metrics else None,
            "vwap": round(signal.metrics.vwap, 4) if signal.metrics else None,
            "bid_ask_spread": round(signal.metrics.bid_ask_spread, 4) if signal.metrics else None,
            "mid_price": round(signal.metrics.mid_price, 4) if signal.metrics else None
        }
    }

def calculate_performance_metrics(signals: List[Any], prices: Dict[str, float]) -> Dict[str, Any]:
    """Calculate performance metrics for signals."""
    if not signals:
        return {}
    
    # Filter non-WAIT signals
    actionable_signals = [s for s in signals if s.signal_type.value != 'WAIT']
    
    if not actionable_signals:
        return {"total_signals": len(signals), "actionable_signals": 0}
    
    # Calculate basic metrics
    long_signals = [s for s in actionable_signals if s.signal_type.value == 'LONG']
    short_signals = [s for s in actionable_signals if s.signal_type.value == 'SHORT']
    
    avg_confidence = np.mean([s.confidence for s in actionable_signals])
    avg_risk_score = np.mean([s.risk_score for s in actionable_signals])
    
    return {
        "total_signals": len(signals),
        "actionable_signals": len(actionable_signals),
        "long_signals": len(long_signals),
        "short_signals": len(short_signals),
        "avg_confidence": round(avg_confidence, 3),
        "avg_risk_score": round(avg_risk_score, 3),
        "signal_distribution": {
            "LONG": len(long_signals),
            "SHORT": len(short_signals),
            "WAIT": len(signals) - len(actionable_signals)
        }
    }

def resample_data(data: pd.DataFrame, timeframe: str = '1min') -> pd.DataFrame:
    """Resample data to different timeframes."""
    if data.empty:
        return data
    
    # Ensure datetime index
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    
    # Resample
    resampled = data.resample(timeframe).agg({
        'price': 'last',
        'size': 'sum',
        'side': lambda x: 'buy' if (x == 'buy').sum() > (x == 'sell').sum() else 'sell'
    })
    
    return resampled.dropna()

def detect_anomalies(data: pd.Series, threshold: float = 2.0) -> List[int]:
    """Detect anomalies in time series data using z-score."""
    if len(data) < 3:
        return []
    
    z_scores = np.abs((data - data.mean()) / data.std())
    anomalies = np.where(z_scores > threshold)[0].tolist()
    
    return anomalies

def calculate_correlation_matrix(data: Dict[str, pd.Series]) -> pd.DataFrame:
    """Calculate correlation matrix for multiple time series."""
    if not data:
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    return df.corr()

def smooth_data(data: pd.Series, window: int = 5) -> pd.Series:
    """Apply moving average smoothing to data."""
    return data.rolling(window=window, center=True).mean()

def calculate_returns(prices: pd.Series) -> pd.Series:
    """Calculate returns from price series."""
    return prices.pct_change().dropna()

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio."""
    if returns.std() == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / returns.std() * np.sqrt(252)  # Annualized

def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown."""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

def validate_symbol(symbol: str) -> bool:
    """Validate trading symbol format."""
    if not symbol or not isinstance(symbol, str):
        return False
    
    # Basic validation - alphanumeric, 3-10 characters
    return symbol.isalnum() and 3 <= len(symbol) <= 10

def format_price(price: float, decimals: int = 4) -> str:
    """Format price with appropriate decimal places."""
    return f"{price:.{decimals}f}"

def format_volume(volume: float) -> str:
    """Format volume with appropriate units."""
    if volume >= 1e9:
        return f"{volume/1e9:.2f}B"
    elif volume >= 1e6:
        return f"{volume/1e6:.2f}M"
    elif volume >= 1e3:
        return f"{volume/1e3:.2f}K"
    else:
        return f"{volume:.2f}"

def get_timeframe_from_seconds(seconds: int) -> str:
    """Convert seconds to pandas timeframe string."""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        return f"{seconds//60}min"
    elif seconds < 86400:
        return f"{seconds//3600}h"
    else:
        return f"{seconds//86400}D"

def create_directory_structure(base_path: str):
    """Create necessary directory structure."""
    base = Path(base_path)
    directories = [
        'logs',
        'data',
        'config',
        'exports'
    ]
    
    for directory in directories:
        (base / directory).mkdir(exist_ok=True)

def cleanup_old_files(directory: str, max_age_days: int = 7):
    """Clean up old files in directory."""
    directory_path = Path(directory)
    if not directory_path.exists():
        return
    
    cutoff_time = datetime.now() - timedelta(days=max_age_days)
    
    for file_path in directory_path.iterdir():
        if file_path.is_file() and datetime.fromtimestamp(file_path.stat().st_mtime) < cutoff_time:
            try:
                file_path.unlink()
                logging.info(f"Deleted old file: {file_path}")
            except Exception as e:
                logging.error(f"Error deleting file {file_path}: {e}")

class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, max_calls: int, time_window: float):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    async def acquire(self):
        """Acquire permission to make a call."""
        now = datetime.now()
        
        # Remove old calls outside time window
        self.calls = [call_time for call_time in self.calls 
                     if (now - call_time).total_seconds() < self.time_window]
        
        # Check if we can make a call
        if len(self.calls) >= self.max_calls:
            sleep_time = self.time_window - (now - self.calls[0]).total_seconds()
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        # Record this call
        self.calls.append(now)

def exponential_moving_average(data: pd.Series, span: int) -> pd.Series:
    """Calculate exponential moving average."""
    return data.ewm(span=span).mean()

def bollinger_bands(data: pd.Series, window: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
    """Calculate Bollinger Bands."""
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    
    return {
        'upper': rolling_mean + (rolling_std * std_dev),
        'middle': rolling_mean,
        'lower': rolling_mean - (rolling_std * std_dev)
    }

def in_session(ts: datetime, tz: str = "America/Los_Angeles") -> bool:
    """
    Check if timestamp is within trading sessions.
    
    Trading sessions:
    - London: 17:00–02:00 LA time (next day)
    - New York: 06:30–13:00 LA time
    
    Args:
        ts: Timestamp to check
        tz: Timezone string (default: "America/Los_Angeles")
    
    Returns:
        True if within trading sessions, False otherwise
    """
    try:
        # Convert timestamp to LA timezone
        la_tz = pytz.timezone(tz)
        
        # If ts is naive, assume it's already in LA time
        if ts.tzinfo is None:
            ts_la = la_tz.localize(ts)
        else:
            # Convert to LA timezone
            ts_la = ts.astimezone(la_tz)
        
        # Extract time components
        hour = ts_la.hour
        minute = ts_la.minute
        time_minutes = hour * 60 + minute
        
        # London session: 17:00–02:00 LA time (next day)
        # This spans from 17:00 (1020 minutes) to 02:00 next day (120 minutes)
        london_start = 17 * 60  # 17:00 = 1020 minutes
        london_end = 2 * 60     # 02:00 = 120 minutes
        
        # NY session: 06:30–13:00 LA time
        ny_start = 6 * 60 + 30  # 06:30 = 390 minutes
        ny_end = 13 * 60        # 13:00 = 780 minutes
        
        # Check London session (spans midnight)
        in_london = (time_minutes >= london_start) or (time_minutes <= london_end)
        
        # Check NY session
        in_ny = ny_start <= time_minutes <= ny_end
        
        # Return True if in either session
        return in_london or in_ny
        
    except Exception as e:
        logging.error(f"Error checking session for timestamp {ts}: {e}")
        return False

def ema(prev: float, x: float, n: int) -> float:
    """
    Calculate exponential moving average.
    
    Classic EMA formula: EMA = (x * α) + (prev * (1 - α))
    where α = 2 / (n + 1)
    
    Args:
        prev: Previous EMA value
        x: Current value
        n: Period (number of periods)
    
    Returns:
        New EMA value
    """
    try:
        # Calculate smoothing factor
        alpha = 2.0 / (n + 1)
        
        # Calculate EMA
        ema_value = (x * alpha) + (prev * (1 - alpha))
        
        return ema_value
        
    except Exception as e:
        logging.error(f"Error calculating EMA: {e}")
        return x  # Return current value as fallback
