#!/usr/bin/env python3
"""
Test script for the new signal engine logic.
Tests the generate_signal function with various scenarios.
"""

import time
import random
from datetime import datetime, timedelta
from core.signals import generate_signal, SessionState, update_session_state

def create_test_scenarios():
    """Create various test scenarios for signal generation."""
    scenarios = []
    
    # Scenario 1: Strong LONG signal
    scenarios.append({
        'name': 'Strong LONG Signal',
        'metrics': {
            'cvd': 150.0,
            'cvd_trend': 25.0,  # CVD rising
            'imbalance': 0.25,   # Strong bid imbalance
            'price': 50100.0,
            'vwap': 50000.0      # Price above VWAP
        },
        'expected': 'LONG'
    })
    
    # Scenario 2: Strong SHORT signal
    scenarios.append({
        'name': 'Strong SHORT Signal',
        'metrics': {
            'cvd': -120.0,
            'cvd_trend': -20.0,  # CVD falling
            'imbalance': -0.22,   # Strong ask imbalance
            'price': 49900.0,
            'vwap': 50000.0      # Price below VWAP
        },
        'expected': 'SHORT'
    })
    
    # Scenario 3: Weak LONG signal (close to threshold)
    scenarios.append({
        'name': 'Weak LONG Signal',
        'metrics': {
            'cvd': 50.0,
            'cvd_trend': 5.0,    # CVD rising weakly
            'imbalance': 0.16,   # Just above threshold
            'price': 50001.0,
            'vwap': 50000.0      # Price barely above VWAP
        },
        'expected': 'LONG'
    })
    
    # Scenario 4: CVD rising but imbalance too low
    scenarios.append({
        'name': 'CVD Rising, Low Imbalance',
        'metrics': {
            'cvd': 100.0,
            'cvd_trend': 15.0,   # CVD rising
            'imbalance': 0.10,   # Below threshold
            'price': 50100.0,
            'vwap': 50000.0      # Price above VWAP
        },
        'expected': 'WAIT'
    })
    
    # Scenario 5: Strong imbalance but CVD falling
    scenarios.append({
        'name': 'Strong Imbalance, CVD Falling',
        'metrics': {
            'cvd': -80.0,
            'cvd_trend': -10.0,  # CVD falling
            'imbalance': 0.20,   # Strong bid imbalance
            'price': 50100.0,
            'vwap': 50000.0      # Price above VWAP
        },
        'expected': 'WAIT'
    })
    
    # Scenario 6: Price below VWAP but other conditions not met
    scenarios.append({
        'name': 'Price Below VWAP, Other Conditions Not Met',
        'metrics': {
            'cvd': 30.0,
            'cvd_trend': 2.0,    # CVD rising weakly
            'imbalance': 0.12,   # Below threshold
            'price': 49900.0,
            'vwap': 50000.0      # Price below VWAP
        },
        'expected': 'WAIT'
    })
    
    # Scenario 7: Neutral conditions
    scenarios.append({
        'name': 'Neutral Conditions',
        'metrics': {
            'cvd': 5.0,
            'cvd_trend': 0.5,    # CVD barely rising
            'imbalance': 0.05,   # Low imbalance
            'price': 50000.0,
            'vwap': 50000.0      # Price at VWAP
        },
        'expected': 'WAIT'
    })
    
    # Scenario 8: Edge case - exactly at thresholds
    scenarios.append({
        'name': 'Edge Case - At Thresholds',
        'metrics': {
            'cvd': 0.0,
            'cvd_trend': 0.0,    # CVD not changing
            'imbalance': 0.15,   # Exactly at threshold
            'price': 50000.0,
            'vwap': 50000.0      # Price at VWAP
        },
        'expected': 'WAIT'
    })
    
    return scenarios

def test_signal_generation():
    """Test signal generation with various scenarios."""
    print("Signal Engine Logic Test Suite")
    print("=" * 50)
    
    # Create session state
    session_state = SessionState(start_time=datetime.now())
    
    # Test scenarios
    scenarios = create_test_scenarios()
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\nTest {i}: {scenario['name']}")
        print("-" * 30)
        
        # Display input metrics
        print("Input Metrics:")
        for key, value in scenario['metrics'].items():
            print(f"  {key}: {value}")
        
        # Generate signal
        signal_type, confidence = generate_signal(scenario['metrics'], session_state)
        
        # Display results
        print(f"\nResult:")
        print(f"  Signal: {signal_type}")
        print(f"  Confidence: {confidence}")
        print(f"  Expected: {scenario['expected']}")
        
        # Check if result matches expectation
        if signal_type == scenario['expected']:
            print(f"  ✓ PASS")
        else:
            print(f"  ✗ FAIL - Expected {scenario['expected']}, got {signal_type}")
        
        # Update session state
        session_state = update_session_state(session_state, signal_generated=True)

def test_session_state_impact():
    """Test how session state affects signal confidence."""
    print("\n" + "=" * 50)
    print("Session State Impact Test")
    print("=" * 50)
    
    # Test metrics that should generate a LONG signal
    test_metrics = {
        'cvd': 100.0,
        'cvd_trend': 15.0,
        'imbalance': 0.20,
        'price': 50100.0,
        'vwap': 50000.0
    }
    
    # Test with different session states
    session_states = [
        {
            'name': 'Fresh Session',
            'state': SessionState(start_time=datetime.now())
        },
        {
            'name': 'Active Session',
            'state': SessionState(
                start_time=datetime.now() - timedelta(hours=1),
                total_volume=1000.0,
                trade_count=50,
                signal_count=5,
                consecutive_signals=1
            )
        },
        {
            'name': 'High Activity Session',
            'state': SessionState(
                start_time=datetime.now() - timedelta(hours=2),
                total_volume=5000.0,
                trade_count=200,
                signal_count=20,
                consecutive_signals=3
            )
        },
        {
            'name': 'Overactive Session',
            'state': SessionState(
                start_time=datetime.now() - timedelta(hours=1),
                total_volume=2000.0,
                trade_count=100,
                signal_count=15,  # High signal frequency
                consecutive_signals=5
            )
        }
    ]
    
    for session_info in session_states:
        print(f"\n{session_info['name']}:")
        print("-" * 20)
        
        signal_type, confidence = generate_signal(test_metrics, session_info['state'])
        
        print(f"  Signal: {signal_type}")
        print(f"  Confidence: {confidence}")
        print(f"  Session Activity:")
        print(f"    Trade Count: {session_info['state'].trade_count}")
        print(f"    Signal Count: {session_info['state'].signal_count}")
        print(f"    Consecutive Signals: {session_info['state'].consecutive_signals}")

def test_confidence_calculation():
    """Test confidence calculation with different metric strengths."""
    print("\n" + "=" * 50)
    print("Confidence Calculation Test")
    print("=" * 50)
    
    session_state = SessionState(start_time=datetime.now())
    
    # Test different strength levels
    strength_levels = [
        {
            'name': 'Very Strong',
            'metrics': {
                'cvd': 200.0,
                'cvd_trend': 30.0,
                'imbalance': 0.30,
                'price': 50200.0,
                'vwap': 50000.0
            }
        },
        {
            'name': 'Strong',
            'metrics': {
                'cvd': 150.0,
                'cvd_trend': 20.0,
                'imbalance': 0.25,
                'price': 50100.0,
                'vwap': 50000.0
            }
        },
        {
            'name': 'Moderate',
            'metrics': {
                'cvd': 100.0,
                'cvd_trend': 15.0,
                'imbalance': 0.20,
                'price': 50050.0,
                'vwap': 50000.0
            }
        },
        {
            'name': 'Weak',
            'metrics': {
                'cvd': 50.0,
                'cvd_trend': 8.0,
                'imbalance': 0.16,
                'price': 50010.0,
                'vwap': 50000.0
            }
        }
    ]
    
    for level in strength_levels:
        print(f"\n{level['name']} Signal:")
        print("-" * 15)
        
        signal_type, confidence = generate_signal(level['metrics'], session_state)
        
        print(f"  Signal: {signal_type}")
        print(f"  Confidence: {confidence}")
        
        # Show metric breakdown
        metrics = level['metrics']
        print(f"  CVD Trend: {metrics['cvd_trend']}")
        print(f"  Imbalance: {metrics['imbalance']}")
        print(f"  Price/VWAP Ratio: {metrics['price']/metrics['vwap']:.4f}")

def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "=" * 50)
    print("Edge Cases Test")
    print("=" * 50)
    
    session_state = SessionState(start_time=datetime.now())
    
    edge_cases = [
        {
            'name': 'Zero Values',
            'metrics': {
                'cvd': 0.0,
                'cvd_trend': 0.0,
                'imbalance': 0.0,
                'price': 0.0,
                'vwap': 0.0
            }
        },
        {
            'name': 'Missing Values',
            'metrics': {}
        },
        {
            'name': 'Extreme Values',
            'metrics': {
                'cvd': 10000.0,
                'cvd_trend': 1000.0,
                'imbalance': 1.0,
                'price': 100000.0,
                'vwap': 50000.0
            }
        },
        {
            'name': 'Negative Values',
            'metrics': {
                'cvd': -1000.0,
                'cvd_trend': -100.0,
                'imbalance': -1.0,
                'price': 1000.0,
                'vwap': 50000.0
            }
        }
    ]
    
    for case in edge_cases:
        print(f"\n{case['name']}:")
        print("-" * 15)
        
        try:
            signal_type, confidence = generate_signal(case['metrics'], session_state)
            print(f"  Signal: {signal_type}")
            print(f"  Confidence: {confidence}")
            print(f"  ✓ Handled gracefully")
        except Exception as e:
            print(f"  ✗ Error: {e}")

def test_real_time_simulation():
    """Simulate real-time signal generation."""
    print("\n" + "=" * 50)
    print("Real-Time Simulation")
    print("=" * 50)
    
    session_state = SessionState(start_time=datetime.now())
    
    print("Simulating 30 seconds of trading...")
    print("Press Ctrl+C to stop early")
    
    try:
        for i in range(30):
            # Generate random metrics
            base_price = 50000.0
            price_change = random.uniform(-100, 100)
            current_price = base_price + price_change
            
            metrics = {
                'cvd': random.uniform(-200, 200),
                'cvd_trend': random.uniform(-30, 30),
                'imbalance': random.uniform(-0.5, 0.5),
                'price': current_price,
                'vwap': base_price + random.uniform(-50, 50)
            }
            
            # Generate signal
            signal_type, confidence = generate_signal(metrics, session_state)
            
            # Update session state
            trade_volume = random.uniform(0.1, 5.0)
            signal_generated = signal_type != "WAIT"
            session_state = update_session_state(session_state, trade_volume, signal_generated)
            
            # Display results
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] Signal: {signal_type:5} | Confidence: {confidence:3} | "
                  f"CVD Trend: {metrics['cvd_trend']:6.1f} | "
                  f"Imbalance: {metrics['imbalance']:6.3f} | "
                  f"Price: {metrics['price']:8.1f}")
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    
    print(f"\nSession Summary:")
    print(f"  Total Trades: {session_state.trade_count}")
    print(f"  Total Volume: {session_state.total_volume:.2f}")
    print(f"  Signals Generated: {session_state.signal_count}")
    print(f"  Consecutive Signals: {session_state.consecutive_signals}")

def main():
    """Run all tests."""
    try:
        test_signal_generation()
        test_session_state_impact()
        test_confidence_calculation()
        test_edge_cases()
        test_real_time_simulation()
        
        print("\n" + "=" * 50)
        print("✓ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
