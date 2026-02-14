"""Coinbase Advanced Trade API WebSocket stream handler."""

import asyncio
import json
import time
from collections import deque
from typing import Dict, Optional, Deque

import websockets
from loguru import logger

import config


class CoinbaseStream:
    """Handles real-time Coinbase WebSocket data streaming."""

    def __init__(self):
        """Initialize Coinbase stream."""
        self.ws = None
        self.running = False
        self.reconnect_count = 0
        
        # Data buffers
        self.ticker_buffer: Deque = deque(maxlen=config.DATA_BUFFER_SIZE)
        self.orderbook_buffer: Deque = deque(maxlen=config.DATA_BUFFER_SIZE)
        self.trades_buffer: Deque = deque(maxlen=config.DATA_BUFFER_SIZE)
        
        # Latest data
        self.latest_ticker = None
        self.latest_orderbook = None
        self.latest_trade = None
        
        # Statistics
        self.messages_received = 0
        self.last_message_time = None

    async def start(self):
        """Start the Coinbase WebSocket connection."""
        self.running = True
        logger.info("Starting Coinbase WebSocket stream...")
        
        while self.running:
            try:
                await self._connect_and_stream()
            except Exception as e:
                logger.error(f"Coinbase stream error: {e}")
                if self.running:
                    await self._handle_reconnect()

    async def _connect_and_stream(self):
        """Connect to Coinbase WebSocket and stream data."""
        uri = config.COINBASE_WS_URL
        
        async with websockets.connect(uri) as websocket:
            self.ws = websocket
            logger.info(f"Connected to Coinbase WebSocket: {uri}")
            
            # Subscribe to channels
            await self._subscribe()
            
            # Reset reconnect count on successful connection
            self.reconnect_count = 0
            
            # Start receiving messages
            async for message in websocket:
                if not self.running:
                    break
                    
                await self._process_message(message)

    async def _subscribe(self):
        """Subscribe to Coinbase channels."""
        subscribe_message = {
            "type": "subscribe",
            "product_ids": [config.TRADING_PAIR],
            "channels": config.COINBASE_CHANNELS,
        }
        
        await self.ws.send(json.dumps(subscribe_message))
        logger.info(f"Subscribed to Coinbase channels: {config.COINBASE_CHANNELS}")

    async def _process_message(self, message: str):
        """Process incoming WebSocket message."""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            
            # Update statistics
            self.messages_received += 1
            self.last_message_time = time.time()
            
            # Route message to appropriate handler
            if msg_type == "ticker":
                await self._handle_ticker(data)
            elif msg_type == "l2update":
                await self._handle_orderbook_update(data)
            elif msg_type == "match":
                await self._handle_trade(data)
            elif msg_type == "subscriptions":
                logger.debug(f"Subscription confirmed: {data}")
            elif msg_type == "error":
                logger.error(f"Coinbase error: {data.get('message')}")
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode message: {e}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    async def _handle_ticker(self, data: Dict):
        """Handle ticker messages."""
        ticker_data = {
            "timestamp": time.time() * 1000,  # Convert to milliseconds
            "price": float(data.get("price", 0)),
            "volume_24h": float(data.get("volume_24h", 0)),
            "best_bid": float(data.get("best_bid", 0)),
            "best_ask": float(data.get("best_ask", 0)),
            "product_id": data.get("product_id"),
        }
        
        self.latest_ticker = ticker_data
        self.ticker_buffer.append(ticker_data)
        
        logger.debug(f"Ticker: {ticker_data['price']} | Bid: {ticker_data['best_bid']} | Ask: {ticker_data['best_ask']}")

    async def _handle_orderbook_update(self, data: Dict):
        """Handle order book update messages."""
        orderbook_data = {
            "timestamp": time.time() * 1000,
            "changes": data.get("changes", []),
            "product_id": data.get("product_id"),
        }
        
        self.latest_orderbook = orderbook_data
        self.orderbook_buffer.append(orderbook_data)
        
        logger.debug(f"Order book update: {len(orderbook_data['changes'])} changes")

    async def _handle_trade(self, data: Dict):
        """Handle trade/match messages."""
        trade_data = {
            "timestamp": time.time() * 1000,
            "trade_id": data.get("trade_id"),
            "price": float(data.get("price", 0)),
            "size": float(data.get("size", 0)),
            "side": data.get("side"),
            "product_id": data.get("product_id"),
        }
        
        self.latest_trade = trade_data
        self.trades_buffer.append(trade_data)
        
        logger.debug(f"Trade: {trade_data['side']} {trade_data['size']} @ {trade_data['price']}")

    async def _handle_reconnect(self):
        """Handle reconnection logic."""
        self.reconnect_count += 1
        
        if self.reconnect_count > config.WS_MAX_RETRIES:
            logger.error(f"Max reconnection attempts ({config.WS_MAX_RETRIES}) reached. Stopping.")
            self.running = False
            return
        
        wait_time = min(config.WS_RECONNECT_DELAY * self.reconnect_count, 60)
        logger.warning(f"Reconnecting in {wait_time}s (attempt {self.reconnect_count}/{config.WS_MAX_RETRIES})...")
        await asyncio.sleep(wait_time)

    def get_latest_data(self) -> Optional[Dict]:
        """Get the latest combined data snapshot."""
        if not self.latest_ticker:
            return None
            
        return {
            "timestamp": self.latest_ticker["timestamp"],
            "ticker": self.latest_ticker,
            "orderbook": self.latest_orderbook,
            "trade": self.latest_trade,
        }

    def get_buffer_data(self, buffer_type: str, n: int = 100) -> list:
        """Get recent data from specified buffer.
        
        Args:
            buffer_type: 'ticker', 'orderbook', or 'trades'
            n: Number of recent items to return
            
        Returns:
            List of recent data items
        """
        buffer_map = {
            "ticker": self.ticker_buffer,
            "orderbook": self.orderbook_buffer,
            "trades": self.trades_buffer,
        }
        
        buffer = buffer_map.get(buffer_type)
        if buffer is None:
            logger.warning(f"Unknown buffer type: {buffer_type}")
            return []
            
        return list(buffer)[-n:]

    def get_statistics(self) -> Dict:
        """Get stream statistics."""
        return {
            "messages_received": self.messages_received,
            "last_message_time": self.last_message_time,
            "reconnect_count": self.reconnect_count,
            "ticker_buffer_size": len(self.ticker_buffer),
            "orderbook_buffer_size": len(self.orderbook_buffer),
            "trades_buffer_size": len(self.trades_buffer),
        }

    async def stop(self):
        """Stop the WebSocket connection."""
        logger.info("Stopping Coinbase WebSocket stream...")
        self.running = False
        
        if self.ws:
            await self.ws.close()
            
        logger.info("✅ Coinbase stream stopped")


if __name__ == "__main__":
    # Test the stream
    async def test():
        stream = CoinbaseStream()
        
        async def print_stats():
            while True:
                await asyncio.sleep(10)
                stats = stream.get_statistics()
                logger.info(f"Stats: {stats}")
                
                latest = stream.get_latest_data()
                if latest:
                    logger.info(f"Latest price: {latest['ticker']['price']}")
        
        # Run stream and stats printer
        await asyncio.gather(
            stream.start(),
            print_stats(),
        )
    
    asyncio.run(test())