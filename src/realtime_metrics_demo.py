#!/usr/bin/env python3
"""
Real-time order flow metrics demonstration.
Shows metrics being updated per second with live data simulation.
"""

import asyncio
import time
import random
from datetime import datetime, timedelta
from core.feeds import Trade
from core.metrics import (
    compute_cvd, compute_imbalance, compute_vwap, detect_absorption,
    AdvancedOrderFlowMetrics
)

class RealTimeMetricsDemo:
    """Real-time metrics demonstration."""
    
    def __init__(self):
        self.metrics_calc = AdvancedOrderFlowMetrics()
        self.trades = []
        self.depth_data = None
        self.running = False
        
    def generate_realistic_trade(self, symbol: str, base_price: float) -> Trade:
        """Generate a realistic trade with market dynamics."""
        # Simulate price movement with some trend
        price_change = random.uniform(-0.002, 0.002) * base_price
        current_price = base_price + price_change
        
        # Simulate realistic trade sizes (more small trades than large)
        if random.random() < 0.7:  # 70% small trades
            size = random.uniform(0.001, 0.1)
        elif random.random() < 0.9:  # 20% medium trades
            size = random.uniform(0.1, 1.0)
        else:  # 10% large trades
            size = random.uniform(1.0, 10.0)
        
        # Simulate buy/sell pressure
        side = random.choice(['buy', 'sell'])
        
        return Trade(
            timestamp=datetime.now(),
            price=round(current_price, 2),
            size=round(size, 6),
            side=side,
            symbol=symbol
        )
    
    def generate_realistic_depth(self, symbol: str, base_price: float) -> dict:
        """Generate realistic order book depth."""
        # Create bid levels
        bid_levels = []
        for i in range(15):
            price = base_price - (i + 1) * 0.5
            # Larger quantities at better prices
            qty = random.uniform(0.1, 5.0) * (1 - i * 0.05)
            bid_levels.append({'price': round(price, 2), 'qty': round(qty, 4)})
        
        # Create ask levels
        ask_levels = []
        for i in range(15):
            price = base_price + (i + 1) * 0.5
            # Larger quantities at better prices
            qty = random.uniform(0.1, 5.0) * (1 - i * 0.05)
            ask_levels.append({'price': round(price, 2), 'qty': round(qty, 4)})
        
        return {
            'symbol': symbol,
            'bid_levels': bid_levels,
            'ask_levels': ask_levels
        }
    
    async def simulate_market_data(self, symbol: str = "BTCUSDT"):
        """Simulate real-time market data generation."""
        base_price = 50000.0
        
        while self.running:
            # Generate new trade
            new_trade = self.generate_realistic_trade(symbol, base_price)
            self.trades.append(new_trade)
            
            # Update base price based on trade
            base_price = new_trade.price
            
            # Keep only last 1000 trades
            if len(self.trades) > 1000:
                self.trades = self.trades[-1000:]
            
            # Generate new depth data every 5 trades
            if len(self.trades) % 5 == 0:
                self.depth_data = self.generate_realistic_depth(symbol, base_price)
            
            # Wait before next trade (simulate real-time)
            await asyncio.sleep(random.uniform(0.1, 0.5))
    
    async def update_metrics_per_second(self):
        """Update and display metrics every second."""
        while self.running:
            if len(self.trades) > 0:
                # Calculate all metrics
                metrics = self.metrics_calc.compute_all_advanced_metrics(
                    self.trades, self.depth_data or {}
                )
                
                # Display metrics
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"\n[{timestamp}] Order Flow Metrics:")
                print(f"  CVD: {metrics['cvd']:>10.4f}")
                print(f"  Imbalance: {metrics['imbalance']:>6.4f}")
                print(f"  VWAP: {metrics['vwap']:>10.2f}")
                print(f"  Absorption: {metrics['absorption']:>6.4f}")
                
                # Show recent trades summary
                recent_trades = self.trades[-5:] if len(self.trades) >= 5 else self.trades
                if recent_trades:
                    avg_price = sum(t.price for t in recent_trades) / len(recent_trades)
                    total_volume = sum(t.size for t in recent_trades)
                    buy_volume = sum(t.size for t in recent_trades if t.side == 'buy')
                    sell_volume = sum(t.size for t in recent_trades if t.side == 'sell')
                    
                    print(f"  Recent Trades (last 5):")
                    print(f"    Avg Price: {avg_price:>8.2f}")
                    print(f"    Total Volume: {total_volume:>6.4f}")
                    print(f"    Buy/Sell Ratio: {buy_volume/sell_volume if sell_volume > 0 else 'inf':>6.2f}")
                
                # Show order book summary
                if self.depth_data:
                    best_bid = max(level['price'] for level in self.depth_data['bid_levels'])
                    best_ask = min(level['price'] for level in self.depth_data['ask_levels'])
                    spread = best_ask - best_bid
                    mid_price = (best_bid + best_ask) / 2
                    
                    print(f"  Order Book:")
                    print(f"    Mid Price: {mid_price:>8.2f}")
                    print(f"    Spread: {spread:>8.4f}")
                    print(f"    Best Bid: {best_bid:>8.2f}")
                    print(f"    Best Ask: {best_ask:>8.2f}")
            
            # Wait 1 second before next update
            await asyncio.sleep(1.0)
    
    async def start_demo(self, duration: int = 60):
        """Start the real-time metrics demonstration."""
        print("Real-Time Order Flow Metrics Demo")
        print("=" * 50)
        print(f"Running for {duration} seconds...")
        print("Press Ctrl+C to stop early")
        print("=" * 50)
        
        self.running = True
        
        # Start data generation and metrics updates concurrently
        tasks = [
            asyncio.create_task(self.simulate_market_data()),
            asyncio.create_task(self.update_metrics_per_second())
        ]
        
        try:
            # Run for specified duration
            await asyncio.sleep(duration)
            
        except KeyboardInterrupt:
            print("\nDemo stopped by user")
        
        finally:
            # Stop the demo
            self.running = False
            
            # Cancel tasks
            for task in tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)
            
            print("\nDemo completed!")
            print(f"Total trades processed: {len(self.trades)}")

async def main():
    """Main function."""
    demo = RealTimeMetricsDemo()
    await demo.start_demo(duration=30)  # Run for 30 seconds

if __name__ == "__main__":
    asyncio.run(main())
