"""Configuration file for Polymarket Quantitative Trading System."""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Project Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
HISTORICAL_DIR = DATA_DIR / "historical"
MODELS_DIR = DATA_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, HISTORICAL_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# API CREDENTIALS
# ============================================================================

# Coinbase Advanced Trade API
COINBASE_API_KEY = os.getenv("COINBASE_API_KEY", "")
COINBASE_API_SECRET = os.getenv("COINBASE_API_SECRET", "")
COINBASE_WS_URL = "wss://advanced-trade-ws.coinbase.com"

# Polymarket CLOB API
POLYMARKET_API_KEY = os.getenv("POLYMARKET_API_KEY", "")
POLYMARKET_PRIVATE_KEY = os.getenv("POLYMARKET_PRIVATE_KEY", "")
POLYMARKET_API_URL = "https://clob.polymarket.com"
POLYMARKET_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws"

# Discord Webhook
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# ============================================================================
# TRADING PARAMETERS
# ============================================================================

# Market Configuration
BASE_CURRENCY = "BTC"
QUOTE_CURRENCY = "USD"
TRADING_PAIR = f"{BASE_CURRENCY}-{QUOTE_CURRENCY}"

# Paper Trading Settings
INITIAL_CAPITAL = 10000.0  # Starting capital in USD
MAX_POSITION_SIZE = 0.1  # Maximum 10% of capital per trade
MIN_TRADE_SIZE = 10.0  # Minimum trade size in USD
SLIPPAGE_MODEL = "realistic"  # "realistic" or "optimistic"

# Risk Management
MAX_DRAWDOWN = 0.20  # Stop trading if drawdown exceeds 20%
MAX_DAILY_TRADES = 50
STOP_LOSS_PCT = 0.05  # 5% stop loss
TAKE_PROFIT_PCT = 0.10  # 10% take profit

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

# Technical Indicators Configuration
TECHNICAL_INDICATORS = {
    "RSI": {"periods": [14, 21, 28]},
    "MACD": {"fast": 12, "slow": 26, "signal": 9},
    "BBANDS": {"period": 20, "std": 2},
    "ATR": {"period": 14},
    "ADX": {"period": 14},
    "STOCH": {"k_period": 14, "d_period": 3},
    "CCI": {"period": 20},
    "MFI": {"period": 14},
    "WILLIAMS_R": {"period": 14},
    "ROC": {"period": 12},
    "MOMENTUM": {"period": 10},
    "VWAP": {},
    "OBV": {},
    "EMA": {"periods": [9, 21, 50, 200]},
    "SMA": {"periods": [20, 50, 100, 200]},
}

# Quantitative Models Configuration
QUANT_MODELS = {
    "order_book_imbalance": {"depth_levels": 10},
    "velocity": {"windows": [1, 5, 15, 60]},  # seconds
    "correlation_lag": {"window": 300, "max_lag": 60},  # 5 min window, 60s max lag
    "volatility_smile": {"windows": [60, 300, 900]},  # 1min, 5min, 15min
    "spread_analysis": {"bid_ask_levels": 5},
    "volume_profile": {"bins": 20},
    "price_momentum": {"windows": [5, 10, 30, 60]},  # seconds
    "market_depth": {"price_levels": 10},
}

# Data Window Settings
FEATURE_WINDOW_SIZE = 100  # Number of data points for feature calculation
MIN_DATA_POINTS = 50  # Minimum points before generating features

# ============================================================================
# MACHINE LEARNING
# ============================================================================

# Model Training
RETRAIN_INTERVAL = 900  # 15 minutes in seconds
TRAINING_WINDOW_SIZE = 1000  # Number of historical samples
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

# Model Performance Thresholds
MIN_ACCURACY = 0.70  # Minimum 70% accuracy
MIN_CONFIDENCE = 0.75  # Minimum 75% confidence for trade execution
MIN_PREDICTED_ROI = 0.03  # Minimum 3% predicted ROI

# Ensemble Configuration
ENSEMBLE_MODELS = {
    "xgboost": {
        "weight": 0.35,
        "params": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "objective": "binary:logistic",
        },
    },
    "lightgbm": {
        "weight": 0.35,
        "params": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "objective": "binary",
        },
    },
    "lstm": {
        "weight": 0.30,
        "params": {
            "units": 64,
            "dropout": 0.2,
            "epochs": 50,
            "batch_size": 32,
        },
    },
}

# Q-Learning Configuration
QL_CONFIG = {
    "learning_rate": 0.1,
    "discount_factor": 0.95,
    "epsilon": 0.1,  # Exploration rate
    "epsilon_decay": 0.995,
    "min_epsilon": 0.01,
    "reward_profitable": 1.0,
    "reward_loss": -1.0,
}

# LSTM Configuration
LSTM_CONFIG = {
    "sequence_length": 60,  # 60 time steps
    "n_features": 100,  # Will be dynamically set based on feature engineering
    "hidden_units": [64, 32],
    "dropout": 0.2,
    "batch_size": 32,
    "epochs": 50,
}

# ============================================================================
# DATA INGESTION
# ============================================================================

# Websocket Settings
WS_RECONNECT_DELAY = 5  # seconds
WS_PING_INTERVAL = 30  # seconds
WS_MAX_RETRIES = 10

# Data Synchronization
TIMESTAMP_TOLERANCE = 100  # milliseconds
DATA_BUFFER_SIZE = 1000  # Maximum buffer size

# Coinbase Channels
COINBASE_CHANNELS = [
    "ticker",
    "level2",
    "matches",
]

# Polymarket Markets
POLYMARKET_MARKET_KEYWORDS = [
    "BTC",
    "Bitcoin",
    "$100k",
    "$100,000",
]

# ============================================================================
# DATABASE
# ============================================================================

DATABASE_URL = f"sqlite:///{DATA_DIR}/trades.db"
DATABASE_ECHO = False  # Set to True for SQL debugging

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
LOG_ROTATION = "500 MB"
LOG_RETENTION = "30 days"

# ============================================================================
# DISCORD NOTIFICATIONS
# ============================================================================

# Notification Settings
NOTIFY_ON_SIGNAL = True
NOTIFY_ON_TRADE = True
NOTIFY_ON_ERROR = True
NOTIFY_ON_RETRAIN = True

# Embed Colors
COLOR_BUY = 0x00FF00  # Green
COLOR_SELL = 0xFF0000  # Red
COLOR_INFO = 0x3498DB  # Blue
COLOR_WARNING = 0xF39C12  # Orange
COLOR_ERROR = 0xE74C3C  # Dark Red

# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

# Metrics
TRACK_METRICS = True
METRICS_UPDATE_INTERVAL = 60  # seconds

# Performance Thresholds
ALERT_ON_LOW_ACCURACY = True
ACCURACY_ALERT_THRESHOLD = 0.65

ALERT_ON_HIGH_DRAWDOWN = True
DRAWDOWN_ALERT_THRESHOLD = 0.15

# ============================================================================
# DEVELOPMENT & TESTING
# ============================================================================

DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"
TEST_MODE = os.getenv("TEST_MODE", "False").lower() == "true"

# Mock Data (for testing without API access)
USE_MOCK_DATA = os.getenv("USE_MOCK_DATA", "False").lower() == "true"
MOCK_DATA_INTERVAL = 1  # seconds between mock data points

# Validation
def validate_config():
    """Validate critical configuration settings."""
    errors = []
    
    if not COINBASE_API_KEY and not USE_MOCK_DATA:
        errors.append("COINBASE_API_KEY is required")
    
    if not COINBASE_API_SECRET and not USE_MOCK_DATA:
        errors.append("COINBASE_API_SECRET is required")
    
    if not POLYMARKET_API_KEY and not USE_MOCK_DATA:
        errors.append("POLYMARKET_API_KEY is required")
    
    if not DISCORD_WEBHOOK_URL:
        errors.append("DISCORD_WEBHOOK_URL is required")
    
    if INITIAL_CAPITAL <= 0:
        errors.append("INITIAL_CAPITAL must be positive")
    
    if not 0 < MAX_POSITION_SIZE <= 1:
        errors.append("MAX_POSITION_SIZE must be between 0 and 1")
    
    if errors:
        raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))

if __name__ == "__main__":
    # Run validation when config is executed directly
    try:
        validate_config()
        print("✅ Configuration is valid")
    except ValueError as e:
        print(f"❌ Configuration errors:\n{e}")