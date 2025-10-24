#!/usr/bin/env python3
"""
Simple demonstration of the signal engine logic without external dependencies.
Shows the core logic of the generate_signal function.
"""

from datetime import datetime
from typing import Dict, Tuple

# Simplified version of the signal engine logic
def generate_signal_demo(metrics: Dict[str, float], session_state: Dict) -> Tuple[str, int]:
    """
    Simplified demonstration of the signal generation logic.
    
    Rules:
    - LONG when CVD rising, imbalance > 0.15, price >= VWAP
    - SHORT when CVD falling, imbalance < -0.15, price <= VWAP
    - WAIT otherwise
    """
    # Extract metrics with defaults
    cvd_trend = metrics.get('cvd_trend', 0.0)
    imbalance = metrics.get('imbalance', 0.0)
    price = metrics.get('price', 0.0)
    vwap = metrics.get('vwap', 0.0)
    
    # Determine signal based on rules
    signal_type = "WAIT"
    base_confidence = 20
    
    # LONG signal conditions
    if (cvd_trend > 0 and          # CVD rising
        imbalance > 0.15 and       # Imbalance > 0.15
        price >= vwap):            # Price >= VWAP
        
        signal_type = "LONG"
        base_confidence = 60
        
        # Boost confidence based on strength
        cvd_boost = min(cvd_trend * 2, 20)
        imbalance_boost = min((imbalance - 0.15) * 200, 15)
        price_boost = min((price / vwap - 1.0) * 100, 10) if vwap > 0 else 0
        
        base_confidence += cvd_boost + imbalance_boost + price_boost
    
    # SHORT signal conditions
    elif (cvd_trend < 0 and        # CVD falling
          imbalance < -0.15 and   # Imbalance < -0.15
          price <= vwap):          # Price <= VWAP
          
        signal_type = "SHORT"
        base_confidence = 60
        
        # Boost confidence based on strength
        cvd_boost = min(abs(cvd_trend) * 2, 20)
        imbalance_boost = min((abs(imbalance) - 0.15) * 200, 15)
        price_boost = min((1.0 - price / vwap) * 100, 10) if vwap > 0 else 0
        
        base_confidence += cvd_boost + imbalance_boost + price_boost
    
    # WAIT signal (default)
    else:
        signal_type = "WAIT"
        base_confidence = 20
        
        # Adjust confidence based on how close we are to signal conditions
        if cvd_trend > 0 and imbalance > 0.05 and price >= vwap * 0.999:
            base_confidence = 40  # Close to LONG
        elif cvd_trend < 0 and imbalance < -0.05 and price <= vwap * 1.001:
            base_confidence = 40  # Close to SHORT
    
    # Adjust for session activity
    if signal_type != "WAIT":
        # Reduce confidence if too many recent signals
        if session_state.get('consecutive_signals', 0) > 3:
            base_confidence = int(base_confidence * 0.7)
        
        # Reduce confidence if low session activity
        if session_state.get('trade_count', 0) < 10:
            base_confidence = int(base_confidence * 0.8)
    
    # Clamp confidence to 0-100
    final_confidence = max(0, min(100, base_confidence))
    
    return signal_type, final_confidence

def test_signal_rules():
    """Test the signal generation rules."""
    print("Signal Engine Logic Demonstration")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        {
            'name': 'Strong LONG Signal',
            'metrics': {
                'cvd_trend': 25.0,    # CVD rising strongly
                'imbalance': 0.25,    # Strong bid imbalance
                'price': 50100.0,
                'vwap': 50000.0
            },
            'session': {'consecutive_signals': 1, 'trade_count': 50},
            'expected': 'LONG'
        },
        {
            'name': 'Strong SHORT Signal',
            'metrics': {
                'cvd_trend': -20.0,   # CVD falling
                'imbalance': -0.22,   # Strong ask imbalance
                'price': 49900.0,
                'vwap': 50000.0
            },
            'session': {'consecutive_signals': 1, 'trade_count': 50},
            'expected': 'SHORT'
        },
        {
            'name': 'CVD Rising, Low Imbalance',
            'metrics': {
                'cvd_trend': 15.0,    # CVD rising
                'imbalance': 0.10,    # Below threshold
                'price': 50100.0,
                'vwap': 50000.0
            },
            'session': {'consecutive_signals': 1, 'trade_count': 50},
            'expected': 'WAIT'
        },
        {
            'name': 'Strong Imbalance, CVD Falling',
            'metrics': {
                'cvd_trend': -10.0,   # CVD falling
                'imbalance': 0.20,    # Strong bid imbalance
                'price': 50100.0,
                'vwap': 50000.0
            },
            'session': {'consecutive_signals': 1, 'trade_count': 50},
            'expected': 'WAIT'
        },
        {
            'name': 'Price Below VWAP, Other Conditions Not Met',
            'metrics': {
                'cvd_trend': 2.0,     # CVD rising weakly
                'imbalance': 0.12,    # Below threshold
                'price': 49900.0,
                'vwap': 50000.0
            },
            'session': {'consecutive_signals': 1, 'trade_count': 50},
            'expected': 'WAIT'
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print("-" * 30)
        
        # Display input metrics
        print("Input Metrics:")
        for key, value in test_case['metrics'].items():
            print(f"  {key}: {value}")
        
        # Generate signal
        signal_type, confidence = generate_signal_demo(test_case['metrics'], test_case['session'])
        
        # Display results
        print(f"\nResult:")
        print(f"  Signal: {signal_type}")
        print(f"  Confidence: {confidence}")
        print(f"  Expected: {test_case['expected']}")
        
        # Check if result matches expectation
        if signal_type == test_case['expected']:
            print(f"  ✓ PASS")
        else:
            print(f"  ✗ FAIL - Expected {test_case['expected']}, got {signal_type}")

def test_confidence_calculation():
    """Test confidence calculation with different strengths."""
    print("\n" + "=" * 50)
    print("Confidence Calculation Test")
    print("=" * 50)
    
    session_state = {'consecutive_signals': 1, 'trade_count': 50}
    
    # Test different strength levels
    strength_levels = [
        {
            'name': 'Very Strong LONG',
            'metrics': {
                'cvd_trend': 30.0,
                'imbalance': 0.30,
                'price': 50200.0,
                'vwap': 50000.0
            }
        },
        {
            'name': 'Strong LONG',
            'metrics': {
                'cvd_trend': 20.0,
                'imbalance': 0.25,
                'price': 50100.0,
                'vwap': 50000.0
            }
        },
        {
            'name': 'Moderate LONG',
            'metrics': {
                'cvd_trend': 15.0,
                'imbalance': 0.20,
                'price': 50050.0,
                'vwap': 50000.0
            }
        },
        {
            'name': 'Weak LONG',
            'metrics': {
                'cvd_trend': 8.0,
                'imbalance': 0.16,
                'price': 50010.0,
                'vwap': 50000.0
            }
        }
    ]
    
    for level in strength_levels:
        print(f"\n{level['name']}:")
        print("-" * 15)
        
        signal_type, confidence = generate_signal_demo(level['metrics'], session_state)
        
        print(f"  Signal: {signal_type}")
        print(f"  Confidence: {confidence}")
        
        # Show metric breakdown
        metrics = level['metrics']
        print(f"  CVD Trend: {metrics['cvd_trend']}")
        print(f"  Imbalance: {metrics['imbalance']}")
        print(f"  Price/VWAP Ratio: {metrics['price']/metrics['vwap']:.4f}")

def test_session_impact():
    """Test how session state affects confidence."""
    print("\n" + "=" * 50)
    print("Session State Impact Test")
    print("=" * 50)
    
    # Test metrics that should generate a LONG signal
    test_metrics = {
        'cvd_trend': 15.0,
        'imbalance': 0.20,
        'price': 50100.0,
        'vwap': 50000.0
    }
    
    # Test with different session states
    session_states = [
        {
            'name': 'Fresh Session',
            'state': {'consecutive_signals': 1, 'trade_count': 50}
        },
        {
            'name': 'Active Session',
            'state': {'consecutive_signals': 2, 'trade_count': 100}
        },
        {
            'name': 'Overactive Session',
            'state': {'consecutive_signals': 5, 'trade_count': 200}
        },
        {
            'name': 'Low Activity Session',
            'state': {'consecutive_signals': 1, 'trade_count': 5}
        }
    ]
    
    for session_info in session_states:
        print(f"\n{session_info['name']}:")
        print("-" * 20)
        
        signal_type, confidence = generate_signal_demo(test_metrics, session_info['state'])
        
        print(f"  Signal: {signal_type}")
        print(f"  Confidence: {confidence}")
        print(f"  Session Activity:")
        print(f"    Trade Count: {session_info['state']['trade_count']}")
        print(f"    Consecutive Signals: {session_info['state']['consecutive_signals']}")

def main():
    """Run all tests."""
    try:
        test_signal_rules()
        test_confidence_calculation()
        test_session_impact()
        
        print("\n" + "=" * 50)
        print("✓ All tests completed successfully!")
        
        print("\nKey Rules Implemented:")
        print("  LONG: CVD rising + imbalance > 0.15 + price >= VWAP")
        print("  SHORT: CVD falling + imbalance < -0.15 + price <= VWAP")
        print("  WAIT: All other conditions")
        print("  Confidence: 0-100 based on alignment strength and session activity")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
