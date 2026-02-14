# Polymarket Quantitative Trading System

## 🚀 Overview

A sophisticated, self-learning Python trading system that trades Polymarket Prediction Shares (specifically Crypto/BTC markets) by using Coinbase Spot Data as a leading indicator.

**Phase 1**: Paper Trading & Discord Signal Bot

## 🎯 Key Features

- **Real-time Data Ingestion**: Streams from Coinbase Advanced Trade API and Polymarket CLOB API
- **50 Technical Indicators**: Using pandas-ta (RSI, MACD, Bollinger Bands, ATR, VWAP, etc.)
- **50 Quantitative Models**: Order book imbalance, velocity, correlation lag, volatility smile, etc.
- **Self-Learning Engine**: Ensemble ML (XGBoost + LightGBM + LSTM) with Q-Learning RL
- **Auto-Retraining**: Model retrains every 15 minutes on latest market data
- **Paper Trading**: Simulates fills with realistic slippage based on order book liquidity
- **Discord Integration**: Rich embed notifications for every trade signal

## 📊 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA INGESTION LAYER                     │
├──────────────────────────┬──────────────────────────────────┤
│  Coinbase WebSocket      │  Polymarket CLOB API             │
│  (BTC-USD)               │  (BTC Prediction Markets)        │
└──────────────┬───────────┴────────────┬─────────────────────┘
               │                        │
               v                        v
┌─────────────────────────────────────────────────────────────┐
│                  FEATURE ENGINEERING LAYER                   │
├──────────────────────────┬──────────────────────────────────┤
│  50 Technical Indicators │  50 Quantitative Models          │
│  (pandas-ta)             │  (Custom algorithms)             │
└──────────────┬───────────┴────────────┬─────────────────────┘
               │                        │
               v                        v
┌─────────────────────────────────────────────────────────────┐
│                   MACHINE LEARNING ENGINE                    │
├──────────────────────────────────────────────────────────────┤
│  • XGBoost + LightGBM + LSTM (Voting Classifier)            │
│  • Q-Learning Reinforcement Learning Agent                   │
│  • Auto-retrain every 15 minutes                             │
│  • Target Accuracy: >70%                                     │
└──────────────┬───────────────────────────────────────────────┘
               │
               v
┌─────────────────────────────────────────────────────────────┐
│                    EXECUTION & REPORTING                     │
├──────────────────────────────────────────────────────────────┤
│  • Paper Trading Engine (realistic slippage)                 │
│  • Discord Webhook Notifications                             │
│  • Performance Metrics & Logging                             │
└─────────────────────────────────────────────────────────────┘
```

## 🛠️ Installation

### Prerequisites

- Python 3.9+
- pip
- Virtual environment (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/tejas03usa/polymarket-quant-trading-system.git
cd polymarket-quant-trading-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Edit .env with your API keys
nano .env  # or use your preferred editor
```

## 🔑 Configuration

Edit `config.py` or `.env` file with your credentials:

```python
# Coinbase API
COINBASE_API_KEY = "your_coinbase_api_key"
COINBASE_API_SECRET = "your_coinbase_api_secret"

# Polymarket API
POLYMARKET_API_KEY = "your_polymarket_api_key"
POLYMARKET_PRIVATE_KEY = "your_polymarket_private_key"

# Discord Webhook
DISCORD_WEBHOOK_URL = "your_discord_webhook_url"
```

## 🚀 Usage

### Run the Trading System

```bash
python main.py
```

### Run Individual Components

```bash
# Test data ingestion
python src/data_ingestion/coinbase_stream.py

# Test feature engineering
python src/features/technical_indicators.py

# Test ML model
python src/ml_engine/ensemble_model.py

# Test Discord notifications
python src/execution/discord_notifier.py
```

## 📁 Project Structure

```
polymarket-quant-trading-system/
├── README.md
├── requirements.txt
├── config.py
├── .env.example
├── .gitignore
├── main.py
├── data/
│   ├── historical/           # Historical data storage
│   ├── models/              # Saved ML models
│   └── trades.db            # SQLite database for trades
├── src/
│   ├── __init__.py
│   ├── data_ingestion/
│   │   ├── __init__.py
│   │   ├── coinbase_stream.py
│   │   ├── polymarket_stream.py
│   │   └── data_synchronizer.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── technical_indicators.py
│   │   ├── quantitative_models.py
│   │   └── feature_pipeline.py
│   ├── ml_engine/
│   │   ├── __init__.py
│   │   ├── ensemble_model.py
│   │   ├── lstm_model.py
│   │   ├── q_learning_agent.py
│   │   └── model_trainer.py
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── paper_trading.py
│   │   ├── discord_notifier.py
│   │   └── trade_executor.py
│   └── utils/
│       ├── __init__.py
│       ├── database.py
│       ├── logger.py
│       └── metrics.py
└── tests/
    ├── __init__.py
    ├── test_data_ingestion.py
    ├── test_features.py
    └── test_ml_engine.py
```

## 🧠 Machine Learning Strategy

### Ensemble Model
- **XGBoost**: Gradient boosting for non-linear patterns
- **LightGBM**: Fast, efficient tree-based learning
- **LSTM**: Captures temporal dependencies in price movements

### Reinforcement Learning
- **Q-Learning Agent**: Learns optimal trading policies
- **Reward Structure**: +1 for profitable trades, -1 for losses
- **Exploration vs Exploitation**: Epsilon-greedy strategy

### Auto-Retraining
- Frequency: Every 15 minutes
- Data: Rolling 15-minute window of features + outcomes
- Target: >70% prediction accuracy

## 📈 Performance Metrics

- **Accuracy**: Percentage of correct trade predictions
- **Sharpe Ratio**: Risk-adjusted returns
- **Win Rate**: Profitable trades / Total trades
- **Average ROI**: Mean return on investment per trade
- **Max Drawdown**: Largest peak-to-trough decline

## 🔔 Discord Notifications

Example notification format:

```
🚨 SIGNAL DETECTED
🟢 BUY "Yes" Share
Market: Will BTC hit $100k by Friday?
Confidence: 84%
Predicted ROI: 5%
🧠 Model Consensus: Strong Buy
Entry Price: $0.72
Target Price: $0.78
Stop Loss: $0.68
```

## ⚠️ Risk Disclaimer

This is a **paper trading system** for educational and research purposes. Real money trading involves substantial risk. Past performance does not guarantee future results. Always:

- Start with paper trading
- Test extensively before using real funds
- Never invest more than you can afford to lose
- Understand the risks of prediction markets
- Comply with local regulations

## 🛣️ Roadmap

### Phase 1 (Current): Paper Trading & Discord Bot ✅
- [x] Real-time data ingestion
- [x] Feature engineering (50+50)
- [x] Ensemble ML + RL
- [x] Paper trading engine
- [x] Discord notifications

### Phase 2: Live Trading
- [ ] Real money execution
- [ ] Position sizing algorithms
- [ ] Risk management system
- [ ] Multi-market support

### Phase 3: Advanced Features
- [ ] Sentiment analysis integration
- [ ] Alternative data sources
- [ ] Advanced RL (PPO, A3C)
- [ ] Web dashboard

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

MIT License - see LICENSE file for details

## 📧 Contact

For questions or support, please open an issue on GitHub.

## 🙏 Acknowledgments

- Coinbase Advanced Trade API
- Polymarket CLOB API
- pandas-ta for technical indicators
- scikit-learn, XGBoost, LightGBM, TensorFlow
- Discord for notifications

---

**Built with ❤️ by quantitative developers for the prediction markets community**