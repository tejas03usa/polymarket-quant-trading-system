"""SQLite database for storing trading data."""
import aiosqlite
import logging
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd

import config


class TradingDatabase:
    """Manages storage of trading data and model training history."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.db_path = config.DATABASE_PATH
        self.connection = None
    
    async def initialize(self):
        """Initialize database and create tables."""
        try:
            # Ensure directory exists
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Connect to database
            self.connection = await aiosqlite.connect(self.db_path)
            
            # Create tables
            await self._create_tables()
            
            self.logger.info(f"Database initialized at {self.db_path}")
        
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}", exc_info=True)
    
    async def _create_tables(self):
        """Create database tables."""
        # Features table
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                feature_data TEXT NOT NULL,
                label INTEGER,
                outcome REAL
            )
        """)
        
        # Trades table
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE NOT NULL,
                timestamp REAL NOT NULL,
                action TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                slippage REAL,
                confidence REAL,
                predicted_roi REAL,
                actual_roi REAL,
                pnl REAL,
                status TEXT NOT NULL,
                state TEXT
            )
        """)
        
        # Model performance table
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                accuracy REAL NOT NULL,
                xgb_accuracy REAL,
                lgb_accuracy REAL,
                lstm_accuracy REAL,
                samples INTEGER
            )
        """)
        
        await self.connection.commit()
    
    async def store_features(self, features: Dict[str, float], label: int = None, outcome: float = None):
        """Store feature vector with optional label and outcome."""
        try:
            feature_json = json.dumps(features)
            timestamp = features.get('timestamp', datetime.now().timestamp())
            
            await self.connection.execute(
                "INSERT INTO features (timestamp, feature_data, label, outcome) VALUES (?, ?, ?, ?)",
                (timestamp, feature_json, label, outcome)
            )
            await self.connection.commit()
        
        except Exception as e:
            self.logger.error(f"Error storing features: {e}")
    
    async def store_trade(self, trade: Dict[str, Any]):
        """Store trade record."""
        try:
            await self.connection.execute("""
                INSERT INTO trades (
                    trade_id, timestamp, action, side, quantity, entry_price,
                    exit_price, slippage, confidence, predicted_roi, actual_roi,
                    pnl, status, state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade['trade_id'],
                trade['timestamp'].timestamp(),
                trade['action'],
                trade['side'],
                trade['quantity'],
                trade['entry_price'],
                trade.get('exit_price'),
                trade.get('slippage'),
                trade.get('confidence'),
                trade.get('predicted_roi'),
                trade.get('actual_roi'),
                trade.get('pnl'),
                trade['status'],
                json.dumps(trade.get('state')) if trade.get('state') else None
            ))
            await self.connection.commit()
            
            self.logger.debug(f"Trade {trade['trade_id']} stored in database")
        
        except Exception as e:
            self.logger.error(f"Error storing trade: {e}", exc_info=True)
    
    async def get_recent_training_data(self, minutes: int = 15) -> pd.DataFrame:
        """Get recent feature data for model training."""
        try:
            cutoff_time = (datetime.now() - timedelta(minutes=minutes)).timestamp()
            
            cursor = await self.connection.execute(
                "SELECT feature_data, label, outcome FROM features WHERE timestamp >= ? ORDER BY timestamp",
                (cutoff_time,)
            )
            
            rows = await cursor.fetchall()
            
            if not rows:
                return pd.DataFrame()
            
            # Parse feature data
            data = []
            for row in rows:
                features = json.loads(row[0])
                features['label'] = row[1]
                features['outcome'] = row[2]
                data.append(features)
            
            return pd.DataFrame(data)
        
        except Exception as e:
            self.logger.error(f"Error fetching training data: {e}", exc_info=True)
            return pd.DataFrame()
    
    async def get_recent_trades(self, minutes: int = 15) -> List[Dict[str, Any]]:
        """Get recent trades for RL agent update."""
        try:
            cutoff_time = (datetime.now() - timedelta(minutes=minutes)).timestamp()
            
            cursor = await self.connection.execute(
                "SELECT * FROM trades WHERE timestamp >= ? ORDER BY timestamp",
                (cutoff_time,)
            )
            
            rows = await cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            trades = []
            for row in rows:
                trade = dict(zip(columns, row))
                # Parse state JSON
                if trade.get('state'):
                    trade['state'] = json.loads(trade['state'])
                trades.append(trade)
            
            return trades
        
        except Exception as e:
            self.logger.error(f"Error fetching recent trades: {e}", exc_info=True)
            return []
    
    async def store_model_performance(self, accuracy: float, xgb_acc: float = None,
                                     lgb_acc: float = None, lstm_acc: float = None,
                                     samples: int = None):
        """Store model performance metrics."""
        try:
            await self.connection.execute("""
                INSERT INTO model_performance (timestamp, accuracy, xgb_accuracy, lgb_accuracy, lstm_accuracy, samples)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().timestamp(),
                accuracy,
                xgb_acc,
                lgb_acc,
                lstm_acc,
                samples
            ))
            await self.connection.commit()
        
        except Exception as e:
            self.logger.error(f"Error storing model performance: {e}")
    
    async def close(self):
        """Close database connection."""
        if self.connection:
            await self.connection.close()
            self.logger.info("Database connection closed")
