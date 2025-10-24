#!/usr/bin/env python3
"""
Test script for advanced order flow metrics.
Tests the new compute_cvd, compute_imbalance, compute_vwap, and detect_absorption functions.
"""

import asyncio
import time
import random
from datetime import datetime, timedelta
from core.feeds import Trade
from core.metrics import (
    compute_cvd, compute_imbalance, compute_vwap, detect_absorption,
    AdvancedOrderFlowMetrics, calculate_price_levels, 
    calculate_volume_at_price, calculate_delta_at_price
)

def create_sample_trades(symbol: str, count: int = 100, base_price: float = 50000.0) -> list:
    """Create sample trade data for testing."""
    trades = []
    current_price = base_price
    
    for i in range(count):
        # Simulate price movement
        price_change = random.uniform(-0.001, 0.001) * current_price
        current_price += price_change
        
        # Create trade
        trade = Trade(
            timestamp=datetime.now() - timedelta(seconds=count-i),
            price=round(current_price, 2),
            size=random.uniform(0.001, 1.0),
            side=random.choice(['buy', 'sell']),
            symbol=symbol
        )
        trades.append(trade)
    
    return trades

def create_sample_depth(symbol: str) -> dict:
    """Create sample order book depth data."""
    base_price = 50000.0
    
    # Create bid levels (below base price)
    bid_levels = []
    for i in range(10):
        price = base_price - (i + 1) * 0.5
        qty = random.uniform(0.1, 2.0)
        bid_levels.append({'price': price, 'qty': qty})
    
    # Create ask levels (above base price)
    ask_levels = []
    for i in range(10):
        price = base_price + (i + 1) * 0.5
        qty = random.uniform(0.1, 2.0)
        ask_levels.append({'price': price, 'qty': qty})
    
    return {
        'symbol': symbol,
        'bid_levels': bid_levels,
        'ask_levels': ask_levels
    }

def test_compute_cvd():
    """Test CVD computation."""
    print("\n=== Testing CVD Computation ===")
    
    # Create sample trades
    trades = create_sample_trades("BTCUSDT", 50)
    
    # Test basic CVD
    cvd = compute_cvd(trades)
    print(f"Raw CVD: {cvd:.4f}")
    
    # Test with EMA smoothing
    cvd_smoothed = compute_cvd(trades, ema_alpha=0.1)
    print(f"CVD with EMA smoothing: {cvd_smoothed:.4f}")
    
    # Test with different EMA factors
    for alpha in [0.05, 0.1, 0.2]:
        cvd_test = compute_cvd(trades, ema_alpha=alpha)
        print(f"CVD (EMA α={alpha}): {cvd_test:.4f}")
    
    # Verify CVD calculation manually
    manual_cvd = sum(trade.size if trade.side == 'buy' else -trade.size for trade in trades)
    print(f"Manual CVD verification: {manual_cvd:.4f}")
    print(f"CVD calculation correct: {abs(cvd - manual_cvd) < 1e-10}")

def test_compute_imbalance():
    """Test imbalance computation."""
    print("\n=== Testing Imbalance Computation ===")
    
    # Create sample depth data
    depth = create_sample_depth("BTCUSDT")
    
    # Test imbalance calculation
    imbalance = compute_imbalance(depth)
    print(f"Order book imbalance: {imbalance:.4f}")
    
    # Test with empty depth
    empty_imbalance = compute_imbalance({})
    print(f"Empty depth imbalance: {empty_imbalance:.4f}")
    
    # Test with missing levels
    partial_depth = {'bid_levels': [], 'ask_levels': depth['ask_levels']}
    partial_imbalance = compute_imbalance(partial_depth)
    print(f"Partial depth imbalance: {partial_imbalance:.4f}")
    
    # Manual verification
    bid_volume = sum(level['qty'] for level in depth['bid_levels'])
    ask_volume = sum(level['qty'] for level in depth['ask_levels'])
    manual_imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
    print(f"Manual imbalance verification: {manual_imbalance:.4f}")
    print(f"Imbalance calculation correct: {abs(imbalance - manual_imbalance) < 1e-10}")

def test_compute_vwap():
    """Test VWAP computation."""
    print("\n=== Testing VWAP Computation ===")
    
    # Create sample trades
    trades = create_sample_trades("BTCUSDT", 100)
    
    # Test VWAP with different windows
    for window in [10, 50, 100]:
        vwap = compute_vwap(trades, window=window)
        print(f"VWAP (window={window}): {vwap:.4f}")
    
    # Test with EMA smoothing
    vwap_smoothed = compute_vwap(trades, window=50, ema_alpha=0.05)
    print(f"VWAP with EMA smoothing: {vwap_smoothed:.4f}")
    
    # Manual verification
    recent_trades = trades[-50:]  # Last 50 trades
    total_volume = sum(trade.size for trade in recent_trades)
    weighted_price = sum(trade.price * trade.size for trade in recent_trades)
    manual_vwap = weighted_price / total_volume if total_volume > 0 else 0
    print(f"Manual VWAP verification: {manual_vwap:.4f}")
    print(f"VWAP calculation correct: {abs(compute_vwap(trades, window=50) - manual_vwap) < 1e-10}")

def test_detect_absorption():
    """Test absorption detection."""
    print("\n=== Testing Absorption Detection ===")
    
    # Create sample trades with different patterns
    trades_normal = create_sample_trades("BTCUSDT", 20)
    trades_absorbed = create_sample_trades("BTCUSDT", 20)
    
    # Modify trades to simulate absorption (large volume, small price movement)
    base_price = 50000.0
    for i, trade in enumerate(trades_absorbed):
        trade.price = base_price + random.uniform(-0.1, 0.1)  # Very small price movement
        trade.size = random.uniform(5.0, 20.0)  # Large volume
    
    # Test absorption detection
    absorption_normal = detect_absorption(trades_normal)
    absorption_absorbed = detect_absorption(trades_absorbed)
    
    print(f"Normal trading absorption: {absorption_normal:.4f}")
    print(f"Absorbed trading absorption: {absorption_absorbed:.4f}")
    
    # Test with different parameters
    for lookback in [5, 8, 15]:
        absorption = detect_absorption(trades_absorbed, lookback=lookback)
        print(f"Absorption (lookback={lookback}): {absorption:.4f}")
    
    # Test edge cases
    empty_absorption = detect_absorption([])
    single_trade_absorption = detect_absorption(trades_normal[:1])
    
    print(f"Empty trades absorption: {empty_absorption:.4f}")
    print(f"Single trade absorption: {single_trade_absorption:.4f}")

def test_advanced_metrics_class():
    """Test the AdvancedOrderFlowMetrics class."""
    print("\n=== Testing AdvancedOrderFlowMetrics Class ===")
    
    # Create metrics calculator
    metrics_calc = AdvancedOrderFlowMetrics(ema_alpha_cvd=0.1, ema_alpha_vwap=0.05)
    
    # Create sample data
    trades = create_sample_trades("BTCUSDT", 100)
    depth = create_sample_depth("BTCUSDT")
    
    # Test individual metrics
    cvd = metrics_calc.compute_smoothed_cvd(trades)
    vwap = metrics_calc.compute_smoothed_vwap(trades)
    
    print(f"Smoothed CVD: {cvd:.4f}")
    print(f"Smoothed VWAP: {vwap:.4f}")
    
    # Test all metrics together
    all_metrics = metrics_calc.compute_all_advanced_metrics(trades, depth)
    print(f"All metrics: {all_metrics}")
    
    # Test multiple updates (simulating real-time updates)
    print("\nSimulating real-time updates:")
    for i in range(5):
        # Add new trade
        new_trade = Trade(
            timestamp=datetime.now(),
            price=50000 + i * 10,
            size=random.uniform(0.1, 1.0),
            side=random.choice(['buy', 'sell']),
            symbol="BTCUSDT"
        )
        trades.append(new_trade)
        
        # Update metrics
        updated_metrics = metrics_calc.compute_all_advanced_metrics(trades, depth)
        print(f"Update {i+1}: CVD={updated_metrics['cvd']:.4f}, "
              f"VWAP={updated_metrics['vwap']:.4f}, "
              f"Absorption={updated_metrics['absorption']:.4f}")

def test_performance_optimized_functions():
    """Test performance-optimized utility functions."""
    print("\n=== Testing Performance-Optimized Functions ===")
    
    # Create sample trades
    trades = create_sample_trades("BTCUSDT", 1000)  # Large dataset for performance testing
    
    # Test price levels calculation
    start_time = time.time()
    price_levels = calculate_price_levels(trades, tick_size=0.01)
    levels_time = time.time() - start_time
    print(f"Price levels calculation time: {levels_time:.4f}s")
    print(f"Unique price levels: {len(set(price_levels))}")
    
    # Test volume at price calculation
    start_time = time.time()
    volume_at_levels = calculate_volume_at_price(trades, price_levels)
    volume_time = time.time() - start_time
    print(f"Volume at price calculation time: {volume_time:.4f}s")
    print(f"Total volume: {sum(volume_at_levels):.4f}")
    
    # Test delta at price calculation
    start_time = time.time()
    delta_at_levels = calculate_delta_at_price(trades, price_levels)
    delta_time = time.time() - start_time
    print(f"Delta at price calculation time: {delta_time:.4f}s")
    print(f"Total delta: {sum(delta_at_levels):.4f}")
    
    # Performance comparison with manual calculation
    start_time = time.time()
    manual_volume = {}
    for trade in trades:
        price_level = round(trade.price, 2)
        if price_level not in manual_volume:
            manual_volume[price_level] = 0
        manual_volume[price_level] += trade.size
    manual_time = time.time() - start_time
    
    print(f"Manual volume calculation time: {manual_time:.4f}s")
    print(f"NumPy optimization speedup: {manual_time/volume_time:.2f}x")

def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n=== Testing Edge Cases ===")
    
    # Test with empty data
    empty_trades = []
    empty_depth = {}
    
    print(f"CVD with empty trades: {compute_cvd(empty_trades):.4f}")
    print(f"Imbalance with empty depth: {compute_imbalance(empty_depth):.4f}")
    print(f"VWAP with empty trades: {compute_vwap(empty_trades):.4f}")
    print(f"Absorption with empty trades: {detect_absorption(empty_trades):.4f}")
    
    # Test with single trade
    single_trade = [Trade(
        timestamp=datetime.now(),
        price=50000.0,
        size=1.0,
        side='buy',
        symbol='BTCUSDT'
    )]
    
    print(f"CVD with single trade: {compute_cvd(single_trade):.4f}")
    print(f"VWAP with single trade: {compute_vwap(single_trade):.4f}")
    
    # Test with extreme values
    extreme_trades = [
        Trade(datetime.now(), 0.001, 1000000, 'buy', 'BTCUSDT'),
        Trade(datetime.now(), 1000000, 0.001, 'sell', 'BTCUSDT')
    ]
    
    print(f"CVD with extreme values: {compute_cvd(extreme_trades):.4f}")
    print(f"VWAP with extreme values: {compute_vwap(extreme_trades):.4f}")

def main():
    """Run all tests."""
    print("Advanced Order Flow Metrics Test Suite")
    print("=" * 60)
    
    try:
        test_compute_cvd()
        test_compute_imbalance()
        test_compute_vwap()
        test_detect_absorption()
        test_advanced_metrics_class()
        test_performance_optimized_functions()
        test_edge_cases()
        
        print("\n" + "=" * 60)
        print("✓ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
