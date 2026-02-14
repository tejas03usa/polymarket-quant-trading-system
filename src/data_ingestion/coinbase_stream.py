"""Coinbase Advanced Trade API websocket streamer."""
import asyncio
import json
import logging
import time
from collections import deque
from datetime import datetime
from typing import Dict, Any, Optional

import websockets
import hmac
import hashlib
import base64

import config


class CoinbaseStreamer:
    """Streams real-time BTC-USD data from Coinbase Advanced Trade API."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ws = None
        self.running = False
        
        # Data buffers
        self.trade_buffer = deque(maxlen=config.DATA_BUFFER_SIZE)
        self.orderbook = {'bids': deque(maxlen=config.ORDER_BOOK_DEPTH),
                          'asks': deque(maxlen=config.ORDER_BOOK_DEPTH)}
        self.latest_price = None
        self.latest_timestamp = None
        
    async def start(self):
        """Start the websocket connection."""
        self.running = True
        self.logger.info("Starting Coinbase WebSocket stream...")
        
        while self.running:
            try:
                await self._connect_and_stream()
            except Exception as e:
                self.logger.error(f"Coinbase stream error: {e}", exc_info=True)
                await asyncio.sleep(5)
    
    async def stop(self):
        """Stop the websocket connection."""
        self.running = False
        if self.ws:
            await self.ws.close()
        self.logger.info("Coinbase stream stopped.")
    
    async def _connect_and_stream(self):
        """Connect to websocket and stream data."""
        async with websockets.connect(config.COINBASE_WS_URL) as ws:
            self.ws = ws
            
            # Subscribe to channels
            subscribe_message = self._create_subscribe_message()
            await ws.send(json.dumps(subscribe_message))
            
            self.logger.info("Connected to Coinbase WebSocket")
            
            async for message in ws:
                try:
                    data = json.loads(message)
                    await self._process_message(data)
                except json.JSONDecodeError:
                    self.logger.warning(f"Invalid JSON: {message}")
                except Exception as e:
                    self.logger.error(f"Error processing message: {e}")
    
    def _create_subscribe_message(self) -> Dict:
        """Create subscription message for Coinbase websocket."""
        timestamp = str(int(time.time()))
        
        message = {
            "type": "subscribe",
            "product_ids": [config.TRADING_PAIR],
            "channels": [
                "ticker",
                "level2",
                "matches"
            ]
        }
        
        # Add authentication if API keys are provided
        if config.COINBASE_API_KEY and config.COINBASE_API_SECRET:
            signature = self._create_signature(timestamp, "GET", "/users/self/verify", "")
            message.update({
                "signature": signature,
                "key": config.COINBASE_API_KEY,
                "timestamp": timestamp,
                "passphrase": ""
            })
        
        return message
    
    def _create_signature(self, timestamp: str, method: str, path: str, body: str) -> str:
        """Create HMAC signature for authentication."""
        message = timestamp + method + path + body
        signature = hmac.new(
            base64.b64decode(config.COINBASE_API_SECRET),
            message.encode('utf-8'),
            hashlib.sha256
        )
        return base64.b64encode(signature.digest()).decode('utf-8')
    
    async def _process_message(self, data: Dict):
        """Process incoming websocket message."""
        msg_type = data.get('type')
        
        if msg_type == 'ticker':
            self._process_ticker(data)
        elif msg_type == 'l2update':
            self._process_l2_update(data)
        elif msg_type == 'match':
            self._process_match(data)
    
    def _process_ticker(self, data: Dict):
        """Process ticker update."""
        try:
            self.latest_price = float(data['price'])
            self.latest_timestamp = datetime.fromisoformat(data['time'].replace('Z', '+00:00'))
        except (KeyError, ValueError) as e:
            self.logger.warning(f"Error processing ticker: {e}")
    
    def _process_l2_update(self, data: Dict):
        """Process Level 2 order book update."""
        try:
            changes = data.get('changes', [])
            
            for change in changes:
                side, price, size = change
                price = float(price)
                size = float(size)
                
                if side == 'buy':
                    self.orderbook['bids'].append({'price': price, 'size': size})
                else:
                    self.orderbook['asks'].append({'price': price, 'size': size})
            
            # Sort order books
            self.orderbook['bids'] = deque(
                sorted(self.orderbook['bids'], key=lambda x: x['price'], reverse=True)[:config.ORDER_BOOK_DEPTH]
            )
            self.orderbook['asks'] = deque(
                sorted(self.orderbook['asks'], key=lambda x: x['price'])[:config.ORDER_BOOK_DEPTH]
            )
            
        except (KeyError, ValueError) as e:
            self.logger.warning(f"Error processing L2 update: {e}")
    
    def _process_match(self, data: Dict):
        """Process trade match."""
        try:
            trade = {
                'price': float(data['price']),
                'size': float(data['size']),
                'side': data['side'],
                'timestamp': datetime.fromisoformat(data['time'].replace('Z', '+00:00'))
            }
            self.trade_buffer.append(trade)
        except (KeyError, ValueError) as e:
            self.logger.warning(f"Error processing match: {e}")
    
    def get_latest_data(self) -> Optional[Dict[str, Any]]:
        """Get the latest aggregated market data."""
        if not self.latest_price:
            return None
        
        return {
            'price': self.latest_price,
            'timestamp': self.latest_timestamp,
            'orderbook': {
                'bids': list(self.orderbook['bids'])[:10],
                'asks': list(self.orderbook['asks'])[:10]
            },
            'recent_trades': list(self.trade_buffer)[-100:],
            'bid_ask_spread': self._calculate_spread()
        }
    
    def _calculate_spread(self) -> float:
        """Calculate bid-ask spread."""
        if not self.orderbook['bids'] or not self.orderbook['asks']:
            return 0.0
        
        best_bid = self.orderbook['bids'][0]['price']
        best_ask = self.orderbook['asks'][0]['price']
        
        return (best_ask - best_bid) / best_bid
