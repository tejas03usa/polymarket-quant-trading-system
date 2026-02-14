"""Polymarket CLOB API websocket streamer."""
import asyncio
import json
import logging
import aiohttp
from collections import deque, defaultdict
from datetime import datetime
from typing import Dict, Any, Optional, List

import config


class PolymarketStreamer:
    """Streams real-time order book data from Polymarket CLOB."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.session = None
        self.running = False
        
        # Active markets tracking
        self.active_markets = {}
        self.market_orderbooks = defaultdict(lambda: {'bids': [], 'asks': []})
        self.market_prices = {}
        
    async def start(self):
        """Start the Polymarket data stream."""
        self.running = True
        self.logger.info("Starting Polymarket stream...")
        
        self.session = aiohttp.ClientSession()
        
        # Fetch active BTC markets
        await self._fetch_active_markets()
        
        # Start polling order books (Polymarket uses REST API for most data)
        while self.running:
            try:
                await self._update_all_orderbooks()
                await asyncio.sleep(2)  # Poll every 2 seconds
            except Exception as e:
                self.logger.error(f"Polymarket stream error: {e}", exc_info=True)
                await asyncio.sleep(5)
    
    async def stop(self):
        """Stop the Polymarket stream."""
        self.running = False
        if self.session:
            await self.session.close()
        self.logger.info("Polymarket stream stopped.")
    
    async def _fetch_active_markets(self):
        """Fetch active BTC-related markets from Polymarket."""
        try:
            url = f"{config.POLYMARKET_CLOB_API_URL}/markets"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    markets = await response.json()
                    
                    # Filter for BTC markets
                    for market in markets:
                        question = market.get('question', '').lower()
                        if any(keyword in question for keyword in ['btc', 'bitcoin']):
                            if market.get('active', False):
                                self.active_markets[market['condition_id']] = market
                                self.logger.info(f"Found active market: {market.get('question', 'Unknown')}")
                    
                    self.logger.info(f"Tracking {len(self.active_markets)} active BTC markets")
                else:
                    self.logger.warning(f"Failed to fetch markets: {response.status}")
        
        except Exception as e:
            self.logger.error(f"Error fetching active markets: {e}", exc_info=True)
    
    async def _update_all_orderbooks(self):
        """Update order books for all active markets."""
        tasks = []
        
        for condition_id in self.active_markets.keys():
            tasks.append(self._fetch_orderbook(condition_id))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _fetch_orderbook(self, condition_id: str):
        """Fetch order book for a specific market."""
        try:
            url = f"{config.POLYMARKET_CLOB_API_URL}/book"
            params = {'token_id': condition_id}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Process order book
                    bids = []
                    asks = []
                    
                    for bid in data.get('bids', []):
                        bids.append({
                            'price': float(bid.get('price', 0)),
                            'size': float(bid.get('size', 0))
                        })
                    
                    for ask in data.get('asks', []):
                        asks.append({
                            'price': float(ask.get('price', 0)),
                            'size': float(ask.get('size', 0))
                        })
                    
                    self.market_orderbooks[condition_id] = {
                        'bids': sorted(bids, key=lambda x: x['price'], reverse=True)[:50],
                        'asks': sorted(asks, key=lambda x: x['price'])[:50],
                        'timestamp': datetime.now()
                    }
                    
                    # Calculate mid price
                    if bids and asks:
                        mid_price = (bids[0]['price'] + asks[0]['price']) / 2
                        self.market_prices[condition_id] = mid_price
        
        except Exception as e:
            self.logger.warning(f"Error fetching orderbook for {condition_id}: {e}")
    
    def get_latest_data(self) -> Optional[Dict[str, Any]]:
        """Get the latest aggregated Polymarket data."""
        if not self.market_orderbooks:
            return None
        
        # Get the most liquid market
        best_market_id = self._get_most_liquid_market()
        
        if not best_market_id:
            return None
        
        orderbook = self.market_orderbooks[best_market_id]
        market_info = self.active_markets.get(best_market_id, {})
        
        return {
            'market_id': best_market_id,
            'market_question': market_info.get('question', 'Unknown'),
            'orderbook': orderbook,
            'price': self.market_prices.get(best_market_id, 0.5),
            'timestamp': orderbook.get('timestamp', datetime.now()),
            'total_markets': len(self.active_markets),
            'spread': self._calculate_spread(orderbook)
        }
    
    def _get_most_liquid_market(self) -> Optional[str]:
        """Get the market ID with the highest liquidity."""
        best_market = None
        max_liquidity = 0
        
        for market_id, orderbook in self.market_orderbooks.items():
            # Calculate liquidity as total size on both sides
            bid_liquidity = sum(b['size'] for b in orderbook.get('bids', []))
            ask_liquidity = sum(a['size'] for a in orderbook.get('asks', []))
            total_liquidity = bid_liquidity + ask_liquidity
            
            if total_liquidity > max_liquidity:
                max_liquidity = total_liquidity
                best_market = market_id
        
        return best_market
    
    def _calculate_spread(self, orderbook: Dict) -> float:
        """Calculate bid-ask spread for a market."""
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        if not bids or not asks:
            return 0.0
        
        best_bid = bids[0]['price']
        best_ask = asks[0]['price']
        
        return best_ask - best_bid
