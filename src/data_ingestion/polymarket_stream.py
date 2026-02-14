"""Polymarket CLOB API stream handler."""

import asyncio
import json
import time
from collections import deque
from typing import Dict, List, Optional, Deque

import aiohttp
import websockets
from loguru import logger

import config


class PolymarketStream:
    """Handles real-time Polymarket CLOB API data streaming."""

    def __init__(self):
        """Initialize Polymarket stream."""
        self.ws = None
        self.session = None
        self.running = False
        self.reconnect_count = 0
        
        # Active markets
        self.active_markets: List[Dict] = []
        self.market_ids: List[str] = []
        
        # Data buffers
        self.orderbook_buffer: Deque = deque(maxlen=config.DATA_BUFFER_SIZE)
        self.price_buffer: Deque = deque(maxlen=config.DATA_BUFFER_SIZE)
        
        # Latest data
        self.latest_orderbooks: Dict[str, Dict] = {}
        self.latest_prices: Dict[str, float] = {}
        
        # Statistics
        self.messages_received = 0
        self.last_message_time = None

    async def start(self):
        """Start the Polymarket data stream."""
        self.running = True
        logger.info("Starting Polymarket stream...")
        
        # Create aiohttp session
        self.session = aiohttp.ClientSession()
        
        # Fetch active BTC markets
        await self._fetch_active_markets()
        
        if not self.active_markets:
            logger.warning("No active BTC markets found on Polymarket")
            return
        
        # Start WebSocket connection
        while self.running:
            try:
                await self._connect_and_stream()
            except Exception as e:
                logger.error(f"Polymarket stream error: {e}")
                if self.running:
                    await self._handle_reconnect()

    async def _fetch_active_markets(self):
        """Fetch active BTC prediction markets from Polymarket."""
        try:
            url = f"{config.POLYMARKET_API_URL}/markets"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    markets = await response.json()
                    
                    # Filter for BTC-related markets
                    self.active_markets = [
                        market for market in markets
                        if any(keyword.lower() in market.get("question", "").lower() 
                               for keyword in config.POLYMARKET_MARKET_KEYWORDS)
                        and market.get("active", False)
                    ]
                    
                    self.market_ids = [market["condition_id"] for market in self.active_markets]
                    
                    logger.info(f"Found {len(self.active_markets)} active BTC markets:")
                    for market in self.active_markets:
                        logger.info(f"  - {market.get('question')}")
                else:
                    logger.error(f"Failed to fetch markets: {response.status}")
                    
        except Exception as e:
            logger.error(f"Error fetching Polymarket markets: {e}")

    async def _connect_and_stream(self):
        """Connect to Polymarket WebSocket and stream data."""
        uri = config.POLYMARKET_WS_URL
        
        async with websockets.connect(uri) as websocket:
            self.ws = websocket
            logger.info(f"Connected to Polymarket WebSocket: {uri}")
            
            # Subscribe to markets
            await self._subscribe()
            
            # Reset reconnect count
            self.reconnect_count = 0
            
            # Receive messages
            async for message in websocket:
                if not self.running:
                    break
                    
                await self._process_message(message)

    async def _subscribe(self):
        """Subscribe to Polymarket market channels."""
        for market_id in self.market_ids:
            subscribe_message = {
                "type": "subscribe",
                "channel": "book",
                "market": market_id,
            }
            
            await self.ws.send(json.dumps(subscribe_message))
            logger.debug(f"Subscribed to market: {market_id}")
        
        logger.info(f"Subscribed to {len(self.market_ids)} Polymarket markets")

    async def _process_message(self, message: str):
        """Process incoming WebSocket message."""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            
            # Update statistics
            self.messages_received += 1
            self.last_message_time = time.time()
            
            # Route message to handler
            if msg_type == "book":
                await self._handle_orderbook(data)
            elif msg_type == "trade":
                await self._handle_trade(data)
            elif msg_type == "subscribed":
                logger.debug(f"Subscription confirmed: {data}")
            elif msg_type == "error":
                logger.error(f"Polymarket error: {data.get('message')}")
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode message: {e}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    async def _handle_orderbook(self, data: Dict):
        """Handle order book messages."""
        market_id = data.get("market")
        bids = data.get("bids", [])
        asks = data.get("asks", [])
        
        orderbook_data = {
            "timestamp": time.time() * 1000,
            "market_id": market_id,
            "bids": [[float(b["price"]), float(b["size"])] for b in bids[:10]],  # Top 10
            "asks": [[float(a["price"]), float(a["size"])] for a in asks[:10]],
        }
        
        # Calculate mid price
        if bids and asks:
            best_bid = float(bids[0]["price"])
            best_ask = float(asks[0]["price"])
            mid_price = (best_bid + best_ask) / 2
            
            orderbook_data["mid_price"] = mid_price
            orderbook_data["spread"] = best_ask - best_bid
            
            # Store latest price
            self.latest_prices[market_id] = mid_price
        
        # Store latest orderbook
        self.latest_orderbooks[market_id] = orderbook_data
        self.orderbook_buffer.append(orderbook_data)
        
        logger.debug(f"Order book [{market_id[:8]}...]: Mid={orderbook_data.get('mid_price', 0):.4f}")

    async def _handle_trade(self, data: Dict):
        """Handle trade messages."""
        trade_data = {
            "timestamp": time.time() * 1000,
            "market_id": data.get("market"),
            "price": float(data.get("price", 0)),
            "size": float(data.get("size", 0)),
            "side": data.get("side"),
        }
        
        self.price_buffer.append(trade_data)
        
        logger.debug(f"Trade [{trade_data['market_id'][:8]}...]: {trade_data['side']} @ {trade_data['price']}")

    async def _handle_reconnect(self):
        """Handle reconnection logic."""
        self.reconnect_count += 1
        
        if self.reconnect_count > config.WS_MAX_RETRIES:
            logger.error(f"Max reconnection attempts reached. Stopping.")
            self.running = False
            return
        
        wait_time = min(config.WS_RECONNECT_DELAY * self.reconnect_count, 60)
        logger.warning(f"Reconnecting in {wait_time}s (attempt {self.reconnect_count})...")
        await asyncio.sleep(wait_time)

    def get_latest_data(self) -> Optional[Dict]:
        """Get the latest Polymarket data snapshot."""
        if not self.latest_orderbooks:
            return None
            
        return {
            "timestamp": time.time() * 1000,
            "markets": self.active_markets,
            "orderbooks": self.latest_orderbooks,
            "prices": self.latest_prices,
        }

    def get_market_price(self, market_id: str) -> Optional[float]:
        """Get the latest mid price for a market."""
        return self.latest_prices.get(market_id)

    def get_buffer_data(self, n: int = 100) -> list:
        """Get recent order book data."""
        return list(self.orderbook_buffer)[-n:]

    def get_statistics(self) -> Dict:
        """Get stream statistics."""
        return {
            "messages_received": self.messages_received,
            "last_message_time": self.last_message_time,
            "reconnect_count": self.reconnect_count,
            "active_markets": len(self.active_markets),
            "orderbook_buffer_size": len(self.orderbook_buffer),
        }

    async def stop(self):
        """Stop the Polymarket stream."""
        logger.info("Stopping Polymarket stream...")
        self.running = False
        
        if self.ws:
            await self.ws.close()
            
        if self.session:
            await self.session.close()
            
        logger.info("✅ Polymarket stream stopped")


if __name__ == "__main__":
    # Test the stream
    async def test():
        stream = PolymarketStream()
        
        async def print_stats():
            await asyncio.sleep(5)  # Wait for initial data
            
            while True:
                await asyncio.sleep(10)
                stats = stream.get_statistics()
                logger.info(f"Stats: {stats}")
                
                latest = stream.get_latest_data()
                if latest:
                    logger.info(f"Tracking {len(latest['prices'])} markets")
                    for market_id, price in latest['prices'].items():
                        logger.info(f"  Market {market_id[:8]}...: ${price:.4f}")
        
        await asyncio.gather(
            stream.start(),
            print_stats(),
        )
    
    asyncio.run(test())