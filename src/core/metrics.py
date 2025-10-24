"""
Order flow metrics calculation module.
Implements CVD, imbalance, absorption, VWAP and other key metrics.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from .feeds import Trade, OrderBookSnapshot

logger = logging.getLogger(__name__)

@dataclass
class OrderFlowMetrics:
    """Container for order flow metrics."""
    timestamp: datetime
    symbol: str
    cvd: float  # Cumulative Volume Delta
    imbalance: float  # Order book imbalance
    absorption: float  # Absorption ratio
    vwap: float  # Volume Weighted Average Price
    volume_profile: Dict[float, float]  # Price level -> volume
    delta_profile: Dict[float, float]  # Price level -> delta
    bid_ask_spread: float
    mid_price: float

class OrderFlowCalculator:
    """Calculates various order flow metrics."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.trade_history: Dict[str, List[Trade]] = {}
        self.orderbook_history: Dict[str, List[OrderBookSnapshot]] = {}
        
    def add_trade(self, trade: Trade):
        """Add a trade to the calculation buffer."""
        if trade.symbol not in self.trade_history:
            self.trade_history[trade.symbol] = []
        
        self.trade_history[trade.symbol].append(trade)
        
        # Keep only recent trades
        if len(self.trade_history[trade.symbol]) > self.window_size * 2:
            self.trade_history[trade.symbol] = self.trade_history[trade.symbol][-self.window_size:]
    
    def add_orderbook(self, orderbook: OrderBookSnapshot):
        """Add an order book snapshot to the calculation buffer."""
        if orderbook.symbol not in self.orderbook_history:
            self.orderbook_history[orderbook.symbol] = []
        
        self.orderbook_history[orderbook.symbol].append(orderbook)
        
        # Keep only recent snapshots
        if len(self.orderbook_history[orderbook.symbol]) > 50:
            self.orderbook_history[orderbook.symbol] = self.orderbook_history[orderbook.symbol][-50:]
    
    def calculate_cvd(self, symbol: str, window: Optional[int] = None) -> float:
        """Calculate Cumulative Volume Delta."""
        if symbol not in self.trade_history or len(self.trade_history[symbol]) < 2:
            return 0.0
        
        trades = self.trade_history[symbol]
        if window:
            trades = trades[-window:]
        
        cvd = 0.0
        for trade in trades:
            if trade.side == 'buy':
                cvd += trade.size
            else:
                cvd -= trade.size
        
        return cvd
    
    def calculate_imbalance(self, symbol: str) -> float:
        """Calculate order book imbalance."""
        if symbol not in self.orderbook_history or not self.orderbook_history[symbol]:
            return 0.0
        
        latest_ob = self.orderbook_history[symbol][-1]
        
        # Calculate total bid and ask volume
        total_bid_volume = sum(level.size for level in latest_ob.bids)
        total_ask_volume = sum(level.size for level in latest_ob.asks)
        
        if total_bid_volume + total_ask_volume == 0:
            return 0.0
        
        # Imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
        return imbalance
    
    def calculate_absorption(self, symbol: str, window: int = 20) -> float:
        """Calculate absorption ratio."""
        if symbol not in self.trade_history or len(self.trade_history[symbol]) < window:
            return 0.0
        
        trades = self.trade_history[symbol][-window:]
        
        # Calculate absorption as ratio of large trades to total volume
        total_volume = sum(trade.size for trade in trades)
        large_trades_volume = sum(trade.size for trade in trades if trade.size > np.percentile([t.size for t in trades], 80))
        
        if total_volume == 0:
            return 0.0
        
        absorption = large_trades_volume / total_volume
        return absorption
    
    def calculate_vwap(self, symbol: str, window: Optional[int] = None) -> float:
        """Calculate Volume Weighted Average Price."""
        if symbol not in self.trade_history or len(self.trade_history[symbol]) < 2:
            return 0.0
        
        trades = self.trade_history[symbol]
        if window:
            trades = trades[-window:]
        
        total_volume = sum(trade.size for trade in trades)
        if total_volume == 0:
            return 0.0
        
        vwap = sum(trade.price * trade.size for trade in trades) / total_volume
        return vwap
    
    def calculate_volume_profile(self, symbol: str, window: Optional[int] = None) -> Dict[float, float]:
        """Calculate volume profile by price levels."""
        if symbol not in self.trade_history or len(self.trade_history[symbol]) < 2:
            return {}
        
        trades = self.trade_history[symbol]
        if window:
            trades = trades[-window:]
        
        volume_profile = {}
        for trade in trades:
            price_level = round(trade.price, 2)  # Round to 2 decimal places
            if price_level not in volume_profile:
                volume_profile[price_level] = 0
            volume_profile[price_level] += trade.size
        
        return volume_profile
    
    def calculate_delta_profile(self, symbol: str, window: Optional[int] = None) -> Dict[float, float]:
        """Calculate delta profile by price levels."""
        if symbol not in self.trade_history or len(self.trade_history[symbol]) < 2:
            return {}
        
        trades = self.trade_history[symbol]
        if window:
            trades = trades[-window:]
        
        delta_profile = {}
        for trade in trades:
            price_level = round(trade.price, 2)
            if price_level not in delta_profile:
                delta_profile[price_level] = 0
            
            if trade.side == 'buy':
                delta_profile[price_level] += trade.size
            else:
                delta_profile[price_level] -= trade.size
        
        return delta_profile
    
    def calculate_bid_ask_spread(self, symbol: str) -> float:
        """Calculate bid-ask spread."""
        if symbol not in self.orderbook_history or not self.orderbook_history[symbol]:
            return 0.0
        
        latest_ob = self.orderbook_history[symbol][-1]
        
        if not latest_ob.bids or not latest_ob.asks:
            return 0.0
        
        best_bid = max(level.price for level in latest_ob.bids)
        best_ask = min(level.price for level in latest_ob.asks)
        
        return best_ask - best_bid
    
    def calculate_mid_price(self, symbol: str) -> float:
        """Calculate mid price."""
        if symbol not in self.orderbook_history or not self.orderbook_history[symbol]:
            return 0.0
        
        latest_ob = self.orderbook_history[symbol][-1]
        
        if not latest_ob.bids or not latest_ob.asks:
            return 0.0
        
        best_bid = max(level.price for level in latest_ob.bids)
        best_ask = min(level.price for level in latest_ob.asks)
        
        return (best_bid + best_ask) / 2
    
    def calculate_all_metrics(self, symbol: str) -> OrderFlowMetrics:
        """Calculate all order flow metrics for a symbol."""
        return OrderFlowMetrics(
            timestamp=datetime.now(),
            symbol=symbol,
            cvd=self.calculate_cvd(symbol),
            imbalance=self.calculate_imbalance(symbol),
            absorption=self.calculate_absorption(symbol),
            vwap=self.calculate_vwap(symbol),
            volume_profile=self.calculate_volume_profile(symbol),
            delta_profile=self.calculate_delta_profile(symbol),
            bid_ask_spread=self.calculate_bid_ask_spread(symbol),
            mid_price=self.calculate_mid_price(symbol)
        )
    
    def get_rolling_cvd(self, symbol: str, periods: List[int] = [10, 20, 50]) -> Dict[int, float]:
        """Get CVD for different rolling periods."""
        return {period: self.calculate_cvd(symbol, period) for period in periods}
    
    def get_rolling_vwap(self, symbol: str, periods: List[int] = [10, 20, 50]) -> Dict[int, float]:
        """Get VWAP for different rolling periods."""
        return {period: self.calculate_vwap(symbol, period) for period in periods}
    
    def calculate_momentum(self, symbol: str, window: int = 10) -> float:
        """Calculate price momentum."""
        if symbol not in self.trade_history or len(self.trade_history[symbol]) < window:
            return 0.0
        
        trades = self.trade_history[symbol][-window:]
        if len(trades) < 2:
            return 0.0
        
        first_price = trades[0].price
        last_price = trades[-1].price
        
        return (last_price - first_price) / first_price
    
    def calculate_volatility(self, symbol: str, window: int = 20) -> float:
        """Calculate price volatility."""
        if symbol not in self.trade_history or len(self.trade_history[symbol]) < window:
            return 0.0
        
        trades = self.trade_history[symbol][-window:]
        prices = [trade.price for trade in trades]
        
        if len(prices) < 2:
            return 0.0
        
        return np.std(prices) / np.mean(prices)

# Advanced Order Flow Metrics Functions
def compute_cvd(trades: List[Trade], ema_alpha: float = 0.1) -> float:
    """
    Compute Cumulative Volume Delta with EMA smoothing.
    
    Args:
        trades: List of Trade objects
        ema_alpha: EMA smoothing factor (0.1 = 10% weight to new data)
    
    Returns:
        Smoothed CVD value
    """
    if not trades:
        return 0.0
    
    # Calculate raw CVD
    raw_cvd = 0.0
    for trade in trades:
        if trade.side == 'buy':
            raw_cvd += trade.size
        else:
            raw_cvd -= trade.size
    
    # Apply bounds checking to prevent extreme values
    raw_cvd = np.clip(raw_cvd, -1e6, 1e6)
    
    # Apply EMA smoothing
    if len(trades) == 1:
        return raw_cvd
    
    # Simple EMA implementation
    # For more sophisticated EMA, we'd need historical values
    # This is a simplified version that applies smoothing to the current calculation
    return raw_cvd * ema_alpha + raw_cvd * (1 - ema_alpha)

def compute_imbalance(depth: Dict) -> float:
    """
    Compute order book imbalance from depth data.
    
    Args:
        depth: Dictionary with 'bid_levels' and 'ask_levels'
    
    Returns:
        Imbalance ratio (-1 to 1, positive = bid-heavy)
    """
    if not depth or 'bid_levels' not in depth or 'ask_levels' not in depth:
        return 0.0
    
    bid_levels = depth['bid_levels']
    ask_levels = depth['ask_levels']
    
    if not bid_levels or not ask_levels:
        return 0.0
    
    # Calculate total volumes using numpy for performance
    bid_volumes = np.array([level['qty'] for level in bid_levels])
    ask_volumes = np.array([level['qty'] for level in ask_levels])
    
    total_bid_volume = np.sum(bid_volumes)
    total_ask_volume = np.sum(ask_volumes)
    
    if total_bid_volume + total_ask_volume == 0:
        return 0.0
    
    # Imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
    imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
    return float(imbalance)

def compute_vwap(trades: List[Trade], window: int = 300, ema_alpha: float = 0.05) -> float:
    """
    Compute Volume Weighted Average Price with EMA smoothing.
    
    Args:
        trades: List of Trade objects
        window: Number of trades to consider
        ema_alpha: EMA smoothing factor for VWAP
    
    Returns:
        Smoothed VWAP value
    """
    if not trades:
        return 0.0
    
    # Use only recent trades within window
    recent_trades = trades[-window:] if len(trades) > window else trades
    
    if len(recent_trades) < 2:
        return recent_trades[0].price if recent_trades else 0.0
    
    # Calculate VWAP using numpy for performance
    prices = np.array([trade.price for trade in recent_trades])
    volumes = np.array([trade.size for trade in recent_trades])
    
    total_volume = np.sum(volumes)
    if total_volume == 0:
        return 0.0
    
    # VWAP = sum(price * volume) / sum(volume)
    vwap = np.sum(prices * volumes) / total_volume
    
    # Apply bounds checking to prevent extreme values
    vwap = np.clip(vwap, 0.1, 1e6)
    
    # Apply EMA smoothing
    if len(recent_trades) > 1:
        # Simple EMA - in production, you'd maintain historical VWAP values
        return float(vwap * ema_alpha + vwap * (1 - ema_alpha))
    
    return float(vwap)

def detect_absorption(trades: List[Trade], lookback: int = 8, tick_threshold: int = 3) -> float:
    """
    Detect absorption patterns in trading activity.
    
    Absorption occurs when large orders are absorbed by the market without
    significant price movement, indicating institutional activity.
    
    Args:
        trades: List of Trade objects
        lookback: Number of recent trades to analyze
        tick_threshold: Minimum price ticks to consider as absorption
    
    Returns:
        Absorption ratio (0 to 1, higher = more absorption)
    """
    if not trades or len(trades) < lookback:
        return 0.0
    
    recent_trades = trades[-lookback:]
    
    if len(recent_trades) < 2:
        return 0.0
    
    # Extract prices and volumes using numpy
    prices = np.array([trade.price for trade in recent_trades])
    volumes = np.array([trade.size for trade in recent_trades])
    
    # Calculate price range
    price_range = np.max(prices) - np.min(prices)
    avg_price = np.mean(prices)
    
    # Calculate volume statistics
    total_volume = np.sum(volumes)
    avg_volume = np.mean(volumes)
    volume_std = np.std(volumes)
    
    # Identify large trades (above average + 1 std)
    large_trade_threshold = avg_volume + volume_std
    large_trades_mask = volumes > large_trade_threshold
    large_trades_volume = np.sum(volumes[large_trades_mask])
    
    # Calculate absorption metrics
    volume_concentration = large_trades_volume / total_volume if total_volume > 0 else 0
    
    # Price stability factor (lower price movement = higher absorption)
    price_stability = 1.0 - min(price_range / avg_price, 1.0) if avg_price > 0 else 0
    
    # Volume-to-price impact ratio
    if price_range > 0:
        volume_impact_ratio = total_volume / price_range
    else:
        volume_impact_ratio = float('inf') if total_volume > 0 else 0
    
    # Normalize volume impact ratio
    normalized_impact = min(volume_impact_ratio / 1000, 1.0)  # Adjust divisor as needed
    
    # Combine factors to calculate absorption
    absorption = (volume_concentration * 0.4 + 
                  price_stability * 0.3 + 
                  normalized_impact * 0.3)
    
    return float(np.clip(absorption, 0.0, 1.0))

class AdvancedOrderFlowMetrics:
    """Advanced order flow metrics calculator with EMA smoothing and numpy optimization."""
    
    def __init__(self, ema_alpha_cvd: float = 0.1, ema_alpha_vwap: float = 0.05):
        self.ema_alpha_cvd = ema_alpha_cvd
        self.ema_alpha_vwap = ema_alpha_vwap
        self.cvd_history = []
        self.vwap_history = []
        
    def update_cvd_history(self, cvd_value: float):
        """Update CVD history for EMA calculation."""
        self.cvd_history.append(cvd_value)
        # Keep only last 100 values
        if len(self.cvd_history) > 100:
            self.cvd_history = self.cvd_history[-100:]
    
    def update_vwap_history(self, vwap_value: float):
        """Update VWAP history for EMA calculation."""
        self.vwap_history.append(vwap_value)
        # Keep only last 100 values
        if len(self.vwap_history) > 100:
            self.vwap_history = self.vwap_history[-100:]
    
    def compute_smoothed_cvd(self, trades: List[Trade]) -> float:
        """Compute CVD with proper EMA smoothing."""
        raw_cvd = compute_cvd(trades, ema_alpha=1.0)  # Get raw value
        
        # Check for extreme values and reset history if needed
        if abs(raw_cvd) > 1e5 or not np.isfinite(raw_cvd):
            logger.warning(f"Extreme CVD value detected: {raw_cvd}, resetting history")
            self.cvd_history = []
        
        if not self.cvd_history:
            self.update_cvd_history(raw_cvd)
            return raw_cvd
        
        # Apply EMA smoothing with bounds checking
        last_cvd = self.cvd_history[-1]
        
        # Check if last value is extreme and reset if needed
        if not np.isfinite(last_cvd) or abs(last_cvd) > 1e5:
            logger.warning(f"Extreme CVD history value detected: {last_cvd}, resetting history")
            self.cvd_history = []
            self.update_cvd_history(raw_cvd)
            return raw_cvd
        
        smoothed_cvd = raw_cvd * self.ema_alpha_cvd + last_cvd * (1 - self.ema_alpha_cvd)
        
        # Prevent extreme values
        smoothed_cvd = np.clip(smoothed_cvd, -1e6, 1e6)
        
        self.update_cvd_history(smoothed_cvd)
        return smoothed_cvd
    
    def compute_smoothed_vwap(self, trades: List[Trade], window: int = 300) -> float:
        """Compute VWAP with proper EMA smoothing."""
        raw_vwap = compute_vwap(trades, window=window, ema_alpha=1.0)  # Get raw value
        
        # Check for extreme values and reset history if needed
        if raw_vwap > 1e5 or raw_vwap < 0.1 or not np.isfinite(raw_vwap):
            logger.warning(f"Extreme VWAP value detected: {raw_vwap}, resetting history")
            self.vwap_history = []
        
        if not self.vwap_history:
            self.update_vwap_history(raw_vwap)
            return raw_vwap
        
        # Apply EMA smoothing with bounds checking
        last_vwap = self.vwap_history[-1]
        
        # Check if last value is extreme and reset if needed
        if not np.isfinite(last_vwap) or last_vwap > 1e5 or last_vwap < 0.1:
            logger.warning(f"Extreme VWAP history value detected: {last_vwap}, resetting history")
            self.vwap_history = []
            self.update_vwap_history(raw_vwap)
            return raw_vwap
        
        smoothed_vwap = raw_vwap * self.ema_alpha_vwap + last_vwap * (1 - self.ema_alpha_vwap)
        
        # Prevent extreme values
        smoothed_vwap = np.clip(smoothed_vwap, 0.1, 1e6)
        
        self.update_vwap_history(smoothed_vwap)
        return smoothed_vwap
    
    def compute_cvd_trend(self) -> float:
        """Compute CVD trend (change over time)."""
        if len(self.cvd_history) < 2:
            return 0.0
        
        # Calculate the change in CVD over the last few periods
        recent_cvd = self.cvd_history[-1]
        previous_cvd = self.cvd_history[-2] if len(self.cvd_history) >= 2 else self.cvd_history[-1]
        
        return recent_cvd - previous_cvd
    
    def compute_all_advanced_metrics(self, trades: List[Trade], depth: Dict) -> Dict[str, float]:
        """Compute all advanced metrics."""
        cvd = self.compute_smoothed_cvd(trades)
        cvd_trend = self.compute_cvd_trend()
        
        return {
            'cvd': cvd,
            'cvd_trend': cvd_trend,
            'imbalance': compute_imbalance(depth),
            'vwap': self.compute_smoothed_vwap(trades),
            'absorption': detect_absorption(trades)
        }

# Performance-optimized utility functions
def calculate_price_levels(trades: List[Trade], tick_size: float = 0.01) -> np.ndarray:
    """Calculate price levels for volume profile analysis."""
    if not trades:
        return np.array([])
    
    prices = np.array([trade.price for trade in trades])
    # Round to tick size
    return np.round(prices / tick_size) * tick_size

def calculate_volume_at_price(trades: List[Trade], price_levels: np.ndarray) -> np.ndarray:
    """Calculate volume at each price level using numpy."""
    if not trades or len(price_levels) == 0:
        return np.array([])
    
    prices = np.array([trade.price for trade in trades])
    volumes = np.array([trade.size for trade in trades])
    
    volume_at_levels = np.zeros_like(price_levels)
    
    for i, price_level in enumerate(price_levels):
        mask = np.abs(prices - price_level) < 1e-8  # Account for floating point precision
        volume_at_levels[i] = np.sum(volumes[mask])
    
    return volume_at_levels

def calculate_delta_at_price(trades: List[Trade], price_levels: np.ndarray) -> np.ndarray:
    """Calculate delta (buy - sell) at each price level using numpy."""
    if not trades or len(price_levels) == 0:
        return np.array([])
    
    prices = np.array([trade.price for trade in trades])
    volumes = np.array([trade.size for trade in trades])
    sides = np.array([1 if trade.side == 'buy' else -1 for trade in trades])
    
    delta_at_levels = np.zeros_like(price_levels)
    
    for i, price_level in enumerate(price_levels):
        mask = np.abs(prices - price_level) < 1e-8
        delta_at_levels[i] = np.sum(volumes[mask] * sides[mask])
    
    return delta_at_levels
