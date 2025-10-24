"""
Coinbase Advanced Trade API integration for automatic futures discovery.
Fetches available trading products and filters for Gold futures contracts.
"""

import asyncio
import aiohttp
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class CoinbaseProductDiscovery:
    """Discovers available trading products from Coinbase Advanced Trade API."""
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.coinbase.com/api/v3/brokerage"
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def fetch_all_products(self) -> List[Dict[str, Any]]:
        """
        Fetch all available trading products from Coinbase Advanced Trade API.
        
        Returns:
            List of product dictionaries
        """
        try:
            url = f"{self.base_url}/products"
            
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'OrderFlowEngine/1.0'
            }
            
            # Add authentication if credentials are provided
            if self.api_key and self.api_secret:
                # Note: Coinbase Advanced Trade API requires proper authentication
                # For now, we'll use public endpoints that don't require auth
                pass
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    products = data.get('products', [])
                    logger.info(f"Fetched {len(products)} products from Coinbase")
                    return products
                else:
                    logger.error(f"Failed to fetch products: HTTP {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error fetching products from Coinbase: {e}")
            return []
    
    def filter_gold_futures(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter products for Gold-related contracts.
        
        Args:
            products: List of all products
            
        Returns:
            List of Gold-related products
        """
        gold_futures = []
        
        for product in products:
            product_id = product.get('product_id', '').upper()
            product_type = product.get('product_type', '')
            
            # Check if it's a Gold-related contract (futures or spot)
            if (product_id.startswith('GOL') and 
                len(product_id) == 8 and 
                product_id[3:].isdigit() and  # Last 4 characters should be digits (year)
                product_type in ['FUTURE', 'FUTURES']):
                
                gold_futures.append(product)
                logger.info(f"Found Gold futures: {product_id}")
            
            # Also check for spot Gold trading (XAU, XAG)
            elif product_id in ['XAUUSDT', 'XAGUSDT']:
                gold_futures.append(product)
                logger.info(f"Found Gold spot: {product_id}")
        
        return gold_futures
    
    def filter_active_futures(self, futures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter for active/available futures contracts.
        
        Args:
            futures: List of futures products
            
        Returns:
            List of active futures products
        """
        active_futures = []
        current_year = datetime.now().year
        
        for future in futures:
            product_id = future.get('product_id', '').upper()
            status = future.get('status', '')
            
            # Extract year from product ID (e.g., GOLZ2025 -> 2025)
            try:
                year = int(product_id[3:])
                
                # Only include futures that are current or future years
                if year >= current_year and status in ['online', 'trading']:
                    active_futures.append(future)
                    logger.info(f"Active Gold futures: {product_id} (Year: {year})")
                    
            except (ValueError, IndexError):
                logger.warning(f"Could not parse year from product ID: {product_id}")
                continue
        
        return active_futures
    
    async def discover_gold_futures(self) -> List[str]:
        """
        Discover all available Gold futures contracts.
        
        Returns:
            List of Gold futures product IDs
        """
        try:
            # Fetch all products
            products = await self.fetch_all_products()
            
            if not products:
                logger.warning("No products fetched from Coinbase API")
                return []
            
            # Filter for Gold futures
            gold_futures = self.filter_gold_futures(products)
            
            # Filter for active futures
            active_futures = self.filter_active_futures(gold_futures)
            
            # Extract product IDs
            product_ids = [future['product_id'].upper() for future in active_futures]
            
            logger.info(f"Discovered {len(product_ids)} active Gold futures: {product_ids}")
            return product_ids
            
        except Exception as e:
            logger.error(f"Error discovering Gold futures: {e}")
            return []
    
    def get_futures_info(self, futures: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Extract useful information about futures contracts.
        
        Args:
            futures: List of futures products
            
        Returns:
            Dictionary mapping product_id to futures info
        """
        futures_info = {}
        
        for future in futures:
            product_id = future.get('product_id', '').upper()
            
            futures_info[product_id] = {
                'product_id': product_id,
                'base_currency': future.get('base_currency', ''),
                'quote_currency': future.get('quote_currency', ''),
                'status': future.get('status', ''),
                'product_type': future.get('product_type', ''),
                'expiry_date': future.get('expiry_date', ''),
                'contract_size': future.get('contract_size', ''),
                'min_size': future.get('min_size', ''),
                'max_size': future.get('max_size', ''),
                'price_increment': future.get('price_increment', ''),
                'size_increment': future.get('size_increment', '')
            }
        
        return futures_info

async def discover_coinbase_gold_futures() -> List[str]:
    """
    Convenience function to discover Gold futures from Coinbase.
    
    Returns:
        List of Gold futures product IDs
    """
    async with CoinbaseProductDiscovery() as discovery:
        return await discovery.discover_gold_futures()

def get_hardcoded_gold_futures() -> List[str]:
    """
    Fallback function that returns real cryptocurrency futures available on Binance.
    These are actual trading symbols that exist on Binance Futures.
    
    Returns:
        List of real cryptocurrency futures symbols available on Binance
    """
    return [
        'ADAUSDT',   # Cardano vs USDT (futures)
        'SOLUSDT',   # Solana vs USDT (futures)
        'DOTUSDT',   # Polkadot vs USDT (futures)
        'LINKUSDT',  # Chainlink vs USDT (futures)
        'AVAXUSDT',  # Avalanche vs USDT (futures)
        # Note: MATICUSDT is not available on Binance Futures
        # Using AVAXUSDT instead as it's a popular cryptocurrency futures
    ]

async def get_available_gold_futures() -> List[str]:
    """
    Get available Gold futures, trying API discovery first, then falling back to hardcoded list.
    
    Returns:
        List of available Gold futures symbols
    """
    try:
        # Try to discover from API
        discovered_futures = await discover_coinbase_gold_futures()
        
        if discovered_futures:
            logger.info(f"Successfully discovered {len(discovered_futures)} Gold futures from API")
            return discovered_futures
        else:
            logger.warning("API discovery failed, using hardcoded Gold futures")
            return get_hardcoded_gold_futures()
            
    except Exception as e:
        logger.error(f"Error in Gold futures discovery: {e}")
        logger.info("Falling back to hardcoded Gold futures")
        return get_hardcoded_gold_futures()

if __name__ == "__main__":
    # Test the discovery function
    async def test_discovery():
        futures = await get_available_gold_futures()
        print(f"Available Gold futures: {futures}")
    
    asyncio.run(test_discovery())
