#!/usr/bin/env python3
"""
Test script for automatic Gold futures discovery.
Demonstrates how the system automatically discovers available Gold futures.
"""

import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.coinbase_discovery import get_available_gold_futures, discover_coinbase_gold_futures

async def test_discovery():
    """Test the Gold futures discovery functionality."""
    print("🔍 Testing Gold Futures Auto-Discovery")
    print("=" * 50)
    
    try:
        print("📡 Attempting to discover Gold futures from Coinbase API...")
        discovered_futures = await discover_coinbase_gold_futures()
        
        if discovered_futures:
            print(f"✅ Successfully discovered {len(discovered_futures)} Gold futures:")
            for future in discovered_futures:
                print(f"   • {future}")
        else:
            print("⚠️  No Gold futures discovered from API")
            
    except Exception as e:
        print(f"❌ API discovery failed: {e}")
    
    print("\n🔄 Testing fallback mechanism...")
    try:
        available_futures = await get_available_gold_futures()
        print(f"✅ Available Gold futures (with fallback): {len(available_futures)}")
        for future in available_futures:
            print(f"   • {future}")
            
    except Exception as e:
        print(f"❌ Fallback mechanism failed: {e}")
    
    print("\n📊 Summary:")
    print(f"   • Total Gold futures available: {len(available_futures)}")
    print(f"   • These will be automatically added to the Order Flow Engine")
    print(f"   • No manual configuration required!")

if __name__ == "__main__":
    asyncio.run(test_discovery())
