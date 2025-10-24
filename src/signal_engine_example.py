#!/usr/bin/env python3
"""
Simple example demonstrating the signal engine logic.
Shows how to use generate_signal function with real-world scenarios.
"""

from datetime import datetime
from core.signals import generate_signal, SessionState, update_session_state

def example_long_signal():
    """Example of a LONG signal generation."""
    print("Example 1: LONG Signal")
    print("-" * 20)
    
    # Create session state
    session_state = SessionState(start_time=datetime.now())
    
    # Metrics indicating strong buying pressure
    metrics = {
        'cvd': 150.0,           # Positive CVD
        'cvd_trend': 25.0,      # CVD rising strongly
        'imbalance': 0.25,      # Strong bid imbalance (> 0.15)
        'price': 50100.0,       # Price above VWAP
        'vwap': 50000.0
    }
    
    signal_type, confidence = generate_signal(metrics, session_state)
    
    print(f"Metrics:")
    print(f"  CVD: {metrics['cvd']}")
    print(f"  CVD Trend: {metrics['cvd_trend']} (rising)")
    print(f"  Imbalance: {metrics['imbalance']} (bid-heavy)")
    print(f"  Price: {metrics['price']}")
    print(f"  VWAP: {metrics['vwap']}")
    print(f"  Price vs VWAP: {metrics['price']/metrics['vwap']:.4f}")
    
    print(f"\nSignal: {signal_type}")
    print(f"Confidence: {confidence}/100")
    
    return signal_type, confidence

def example_short_signal():
    """Example of a SHORT signal generation."""
    print("\nExample 2: SHORT Signal")
    print("-" * 20)
    
    # Create session state
    session_state = SessionState(start_time=datetime.now())
    
    # Metrics indicating strong selling pressure
    metrics = {
        'cvd': -120.0,          # Negative CVD
        'cvd_trend': -20.0,     # CVD falling
        'imbalance': -0.22,     # Strong ask imbalance (< -0.15)
        'price': 49900.0,       # Price below VWAP
        'vwap': 50000.0
    }
    
    signal_type, confidence = generate_signal(metrics, session_state)
    
    print(f"Metrics:")
    print(f"  CVD: {metrics['cvd']}")
    print(f"  CVD Trend: {metrics['cvd_trend']} (falling)")
    print(f"  Imbalance: {metrics['imbalance']} (ask-heavy)")
    print(f"  Price: {metrics['price']}")
    print(f"  VWAP: {metrics['vwap']}")
    print(f"  Price vs VWAP: {metrics['price']/metrics['vwap']:.4f}")
    
    print(f"\nSignal: {signal_type}")
    print(f"Confidence: {confidence}/100")
    
    return signal_type, confidence

def example_wait_signal():
    """Example of a WAIT signal generation."""
    print("\nExample 3: WAIT Signal")
    print("-" * 20)
    
    # Create session state
    session_state = SessionState(start_time=datetime.now())
    
    # Metrics with mixed signals
    metrics = {
        'cvd': 50.0,            # Positive CVD
        'cvd_trend': 5.0,       # CVD rising weakly
        'imbalance': 0.10,      # Imbalance below threshold
        'price': 50100.0,       # Price above VWAP
        'vwap': 50000.0
    }
    
    signal_type, confidence = generate_signal(metrics, session_state)
    
    print(f"Metrics:")
    print(f"  CVD: {metrics['cvd']}")
    print(f"  CVD Trend: {metrics['cvd_trend']} (rising weakly)")
    print(f"  Imbalance: {metrics['imbalance']} (below 0.15 threshold)")
    print(f"  Price: {metrics['price']}")
    print(f"  VWAP: {metrics['vwap']}")
    print(f"  Price vs VWAP: {metrics['price']/metrics['vwap']:.4f}")
    
    print(f"\nSignal: {signal_type}")
    print(f"Confidence: {confidence}/100")
    
    print(f"\nReason: CVD is rising and price is above VWAP, but imbalance")
    print(f"({metrics['imbalance']}) is below the 0.15 threshold required for LONG signal.")
    
    return signal_type, confidence

def example_session_impact():
    """Example showing how session state affects confidence."""
    print("\nExample 4: Session State Impact")
    print("-" * 30)
    
    # Same metrics for both sessions
    metrics = {
        'cvd': 100.0,
        'cvd_trend': 15.0,
        'imbalance': 0.20,
        'price': 50100.0,
        'vwap': 50000.0
    }
    
    # Fresh session
    fresh_session = SessionState(start_time=datetime.now())
    signal1, confidence1 = generate_signal(metrics, fresh_session)
    
    # Active session
    active_session = SessionState(
        start_time=datetime.now(),
        total_volume=1000.0,
        trade_count=50,
        signal_count=5,
        consecutive_signals=1
    )
    signal2, confidence2 = generate_signal(metrics, active_session)
    
    print(f"Same metrics, different session states:")
    print(f"  Fresh Session: {signal1} (confidence: {confidence1})")
    print(f"  Active Session: {signal2} (confidence: {confidence2})")
    print(f"  Confidence difference: {confidence2 - confidence1}")

def example_confidence_factors():
    """Example showing confidence calculation breakdown."""
    print("\nExample 5: Confidence Calculation")
    print("-" * 30)
    
    session_state = SessionState(start_time=datetime.now())
    
    # Strong signal metrics
    metrics = {
        'cvd': 200.0,
        'cvd_trend': 30.0,      # Strong CVD trend
        'imbalance': 0.30,      # Strong imbalance
        'price': 50200.0,       # Price well above VWAP
        'vwap': 50000.0
    }
    
    signal_type, confidence = generate_signal(metrics, session_state)
    
    print(f"Strong Signal Metrics:")
    print(f"  CVD Trend: {metrics['cvd_trend']} (strong)")
    print(f"  Imbalance: {metrics['imbalance']} (strong)")
    print(f"  Price/VWAP: {metrics['price']/metrics['vwap']:.4f} (well above)")
    
    print(f"\nResult: {signal_type} with {confidence}/100 confidence")
    print(f"This high confidence comes from:")
    print(f"  - Strong CVD trend ({metrics['cvd_trend']})")
    print(f"  - Strong imbalance ({metrics['imbalance']})")
    print(f"  - Price significantly above VWAP")

def main():
    """Run all examples."""
    print("Signal Engine Examples")
    print("=" * 50)
    
    try:
        example_long_signal()
        example_short_signal()
        example_wait_signal()
        example_session_impact()
        example_confidence_factors()
        
        print("\n" + "=" * 50)
        print("✓ All examples completed!")
        
        print("\nKey Rules:")
        print("  LONG: CVD rising + imbalance > 0.15 + price >= VWAP")
        print("  SHORT: CVD falling + imbalance < -0.15 + price <= VWAP")
        print("  WAIT: All other conditions")
        print("  Confidence: 0-100 based on alignment strength and session activity")
        
    except Exception as e:
        print(f"\n✗ Example failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
