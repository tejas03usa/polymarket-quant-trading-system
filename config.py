"""Configuration file for Polymarket Quant Trading System."""
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
COINBASE_API_KEY = os.getenv('COINBASE_API_KEY', '')
COINBASE_API_SECRET = os.getenv('COINBASE_API_SECRET', '')
POLYMARKET_API_KEY = os.getenv('POLYMARKET_API_KEY', '')
POLYMARKET_PRIVATE_KEY = os.getenv('POLYMARKET_PRIVATE_KEY', '')
DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL', '')

# Websocket URLs
COINBASE_WS_URL = 'wss://advanced-trade-ws.coinbase.com'
POLYMARKET_CLOB_WS_URL = 'wss://ws-subscriptions-clob.polymarket.com/ws/market'
POLYMARKET_CLOB_API_URL = 'https://clob.polymarket.com'

# Trading Parameters
TRADING_PAIR = 'BTC-USD'
POLYMARKET_MARKETS = ['BTC']  # Filter for BTC-related markets
POSITION_SIZE = 100  # USD equivalent for paper trading
MAX_SLIPPAGE = 0.02  # 2% max slippage tolerance

# Model Parameters
MODEL_RETRAIN_INTERVAL = 900  # 15 minutes in seconds
MIN_ACCURACY_THRESHOLD = 0.70  # 70% minimum accuracy
CONFIDENCE_THRESHOLD = 0.75  # 75% confidence to trigger trade

# Feature Engineering
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2
ATR_PERIOD = 14
ROLLING_CORRELATION_WINDOW = 100
VOLATILITY_WINDOW = 30

# RL Parameters
RL_LEARNING_RATE = 0.01
RL_DISCOUNT_FACTOR = 0.95
RL_EPSILON = 0.1  # Exploration rate

# Data Storage
DATABASE_PATH = 'data/trading_data.db'
MODEL_PATH = 'models/'
LOG_PATH = 'logs/'

# Logging
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Ensemble Model Weights
XGBOOST_WEIGHT = 0.4
LIGHTGBM_WEIGHT = 0.3
LSTM_WEIGHT = 0.3

# LSTM Parameters
LSTM_SEQUENCE_LENGTH = 60
LSTM_UNITS = 128
LSTM_DROPOUT = 0.2
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32

# Data Buffer Sizes
DATA_BUFFER_SIZE = 10000
ORDER_BOOK_DEPTH = 50

# Discord Notification Settings
DISCORD_EMBED_COLOR = 0x00FF00  # Green for buy signals
DISCORD_EMBED_COLOR_SELL = 0xFF0000  # Red for sell signals
DISCORD_NOTIFICATION_ENABLED = True
