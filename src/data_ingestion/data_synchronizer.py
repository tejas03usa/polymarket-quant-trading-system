"""Data synchronization module for aligning Coinbase and Polymarket streams."""

import asyncio
import time
from typing import Dict, Optional
from collections import deque

from loguru import logger

import config


class DataSynchronizer:
    """Synchronizes data from Coinbase and Polymarket streams."""

    def __init__(self, coinbase_stream, polymarket_stream):
        """Initialize data synchronizer.
        
        Args:
            coinbase_stream: CoinbaseStream instance
            polymarket_stream: PolymarketStream instance
        """
        self.coinbase = coinbase_stream
        self.polymarket = polymarket_stream
        self.running = False
        
        # Synchronized data buffer
        self.synced_data_buffer = deque(maxlen=config.DATA_BUFFER_SIZE)
        self.latest_synced_data = None
        
        # Statistics
        self.sync_count = 0
        self.sync_errors = 0
        self.last_sync_time = None
        
        # Lag tracking (for correlation analysis)
        self.lag_buffer = deque(maxlen=1000)

    async def start(self):
        """Start the synchronization loop."""
        self.running = True
        logger.info("Starting data synchronization...")
        
        while self.running:
            try:
                await self._synchronize_data()
                await asyncio.sleep(0.1)  # 100ms sync interval
            except Exception as e:
                logger.error(f"Synchronization error: {e}")
                self.sync_errors += 1
                await asyncio.sleep(1)

    async def _synchronize_data(self):
        """Synchronize data from both streams."""
        # Get latest data from both streams
        coinbase_data = self.coinbase.get_latest_data()
        polymarket_data = self.polymarket.get_latest_data()
        
        # Check if both streams have data
        if coinbase_data is None or polymarket_data is None:
            return
        
        # Get timestamps
        cb_timestamp = coinbase_data["timestamp"]
        pm_timestamp = polymarket_data["timestamp"]
        
        # Calculate time difference (lag)
        lag = abs(cb_timestamp - pm_timestamp)
        self.lag_buffer.append(lag)
        
        # Check if timestamps are within tolerance
        if lag > config.TIMESTAMP_TOLERANCE:
            logger.debug(f"Timestamp lag: {lag:.2f}ms (tolerance: {config.TIMESTAMP_TOLERANCE}ms)")
        
        # Create synchronized data object
        synced_data = {
            "timestamp": max(cb_timestamp, pm_timestamp),  # Use latest timestamp
            "sync_time": time.time() * 1000,
            "lag_ms": lag,
            "coinbase": {
                "price": coinbase_data["ticker"]["price"],
                "volume_24h": coinbase_data["ticker"]["volume_24h"],
                "best_bid": coinbase_data["ticker"]["best_bid"],
                "best_ask": coinbase_data["ticker"]["best_ask"],
                "orderbook": coinbase_data.get("orderbook"),
                "latest_trade": coinbase_data.get("trade"),
            },
            "polymarket": {
                "markets": polymarket_data["markets"],
                "orderbooks": polymarket_data["orderbooks"],
                "prices": polymarket_data["prices"],
            },
            "metadata": {
                "coinbase_timestamp": cb_timestamp,
                "polymarket_timestamp": pm_timestamp,
            }
        }
        
        # Store synchronized data
        self.latest_synced_data = synced_data
        self.synced_data_buffer.append(synced_data)
        
        # Update statistics
        self.sync_count += 1
        self.last_sync_time = time.time()
        
        # Log every 100 syncs
        if self.sync_count % 100 == 0:
            avg_lag = sum(self.lag_buffer) / len(self.lag_buffer) if self.lag_buffer else 0
            logger.info(
                f"Sync #{self.sync_count} | "
                f"BTC: ${coinbase_data['ticker']['price']:.2f} | "
                f"PM Markets: {len(polymarket_data['prices'])} | "
                f"Avg Lag: {avg_lag:.2f}ms"
            )

    async def get_latest_data(self) -> Optional[Dict]:
        """Get the latest synchronized data.
        
        Returns:
            Latest synchronized data snapshot or None
        """
        return self.latest_synced_data

    def get_historical_data(self, n: int = 100) -> list:
        """Get recent synchronized data.
        
        Args:
            n: Number of recent data points to return
            
        Returns:
            List of recent synchronized data points
        """
        return list(self.synced_data_buffer)[-n:]

    def calculate_correlation_lag(self, window: int = 300) -> Optional[float]:
        """Calculate correlation lag between Coinbase and Polymarket.
        
        Args:
            window: Time window in seconds
            
        Returns:
            Average lag in milliseconds or None
        """
        if not self.lag_buffer:
            return None
            
        # Get recent lags within window
        current_time = time.time() * 1000
        recent_lags = [
            lag for lag in self.lag_buffer
            if current_time - lag < window * 1000
        ]
        
        if not recent_lags:
            return None
            
        return sum(recent_lags) / len(recent_lags)

    def detect_price_divergence(self, threshold: float = 0.05) -> Optional[Dict]:
        """Detect if Polymarket prices diverge from Coinbase spot price.
        
        Args:
            threshold: Divergence threshold (5% default)
            
        Returns:
            Divergence information or None
        """
        if not self.latest_synced_data:
            return None
            
        coinbase_price = self.latest_synced_data["coinbase"]["price"]
        polymarket_prices = self.latest_synced_data["polymarket"]["prices"]
        
        divergences = []
        
        for market_id, pm_price in polymarket_prices.items():
            # Calculate implied probability-adjusted comparison
            # Polymarket prices are probabilities (0-1), need context for comparison
            # This is a simplified check - actual logic depends on market structure
            
            divergence = {
                "market_id": market_id,
                "polymarket_price": pm_price,
                "coinbase_price": coinbase_price,
            }
            
            divergences.append(divergence)
        
        return {
            "timestamp": self.latest_synced_data["timestamp"],
            "divergences": divergences,
        } if divergences else None

    def get_statistics(self) -> Dict:
        """Get synchronization statistics."""
        avg_lag = sum(self.lag_buffer) / len(self.lag_buffer) if self.lag_buffer else 0
        max_lag = max(self.lag_buffer) if self.lag_buffer else 0
        min_lag = min(self.lag_buffer) if self.lag_buffer else 0
        
        return {
            "sync_count": self.sync_count,
            "sync_errors": self.sync_errors,
            "last_sync_time": self.last_sync_time,
            "buffer_size": len(self.synced_data_buffer),
            "lag_stats": {
                "avg_ms": avg_lag,
                "max_ms": max_lag,
                "min_ms": min_lag,
                "samples": len(self.lag_buffer),
            }
        }

    async def stop(self):
        """Stop the synchronization loop."""
        logger.info("Stopping data synchronization...")
        self.running = False
        logger.info("✅ Data synchronization stopped")


if __name__ == "__main__":
    # Test synchronization
    from .coinbase_stream import CoinbaseStream
    from .polymarket_stream import PolymarketStream
    
    async def test():
        cb_stream = CoinbaseStream()
        pm_stream = PolymarketStream()
        synchronizer = DataSynchronizer(cb_stream, pm_stream)
        
        async def print_stats():
            await asyncio.sleep(10)
            
            while True:
                await asyncio.sleep(15)
                stats = synchronizer.get_statistics()
                logger.info(f"Sync Stats: {stats}")
                
                latest = await synchronizer.get_latest_data()
                if latest:
                    logger.info(f"Latest synced data: Coinbase ${latest['coinbase']['price']:.2f}")
        
        await asyncio.gather(
            cb_stream.start(),
            pm_stream.start(),
            synchronizer.start(),
            print_stats(),
        )
    
    asyncio.run(test())