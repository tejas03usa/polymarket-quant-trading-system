"""Main orchestrator that coordinates all system components."""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

from src.data_ingestion.coinbase_stream import CoinbaseStreamer
from src.data_ingestion.polymarket_stream import PolymarketStreamer
from src.features.feature_engine import FeatureEngine
from src.models.ensemble_model import EnsembleModel
from src.models.rl_agent import RLAgent
from src.execution.paper_trader import PaperTrader
from src.execution.discord_notifier import DiscordNotifier
from src.data_storage.database import TradingDatabase
import config


class TradingOrchestrator:
    """Orchestrates all components of the trading system."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.running = False
        
        # Initialize components
        self.coinbase_stream = CoinbaseStreamer()
        self.polymarket_stream = PolymarketStreamer()
        self.feature_engine = FeatureEngine()
        self.ensemble_model = EnsembleModel()
        self.rl_agent = RLAgent()
        self.paper_trader = PaperTrader()
        self.discord_notifier = DiscordNotifier()
        self.database = TradingDatabase()
        
        # State management
        self.last_retrain_time = datetime.now()
        self.trade_count = 0
        
    async def start(self):
        """Start all system components."""
        self.logger.info("Starting Trading Orchestrator...")
        self.running = True
        
        # Initialize database
        await self.database.initialize()
        
        # Load or initialize models
        self.ensemble_model.load_or_initialize()
        self.rl_agent.load_or_initialize()
        
        # Start data streams
        tasks = [
            asyncio.create_task(self.coinbase_stream.start()),
            asyncio.create_task(self.polymarket_stream.start()),
            asyncio.create_task(self._main_loop())
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def stop(self):
        """Stop all system components."""
        self.logger.info("Stopping Trading Orchestrator...")
        self.running = False
        
        await self.coinbase_stream.stop()
        await self.polymarket_stream.stop()
        await self.database.close()
        
        self.logger.info("All components stopped.")
    
    async def _main_loop(self):
        """Main decision loop."""
        self.logger.info("Main loop started. Waiting for initial data...")
        
        # Wait for initial data
        await asyncio.sleep(10)
        
        while self.running:
            try:
                # Check if it's time to retrain
                if self._should_retrain():
                    await self._retrain_models()
                
                # Get synchronized data
                coinbase_data = self.coinbase_stream.get_latest_data()
                polymarket_data = self.polymarket_stream.get_latest_data()
                
                if not coinbase_data or not polymarket_data:
                    await asyncio.sleep(1)
                    continue
                
                # Generate features
                features = await self.feature_engine.generate_features(
                    coinbase_data, 
                    polymarket_data
                )
                
                if features is None:
                    await asyncio.sleep(1)
                    continue
                
                # Get prediction from ensemble model
                prediction = self.ensemble_model.predict(features)
                
                # Get RL agent decision
                rl_decision = self.rl_agent.get_action(features, prediction)
                
                # Make trading decision
                signal = self._make_trading_decision(prediction, rl_decision, features)
                
                if signal:
                    await self._execute_trade(signal, coinbase_data, polymarket_data)
                
                # Sleep briefly
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}", exc_info=True)
                await asyncio.sleep(5)
    
    def _should_retrain(self) -> bool:
        """Check if it's time to retrain models."""
        elapsed = (datetime.now() - self.last_retrain_time).total_seconds()
        return elapsed >= config.MODEL_RETRAIN_INTERVAL
    
    async def _retrain_models(self):
        """Retrain the ensemble model with recent data."""
        self.logger.info("Starting model retraining...")
        
        try:
            # Get recent training data from database
            training_data = await self.database.get_recent_training_data(
                minutes=15
            )
            
            if len(training_data) < 100:
                self.logger.warning(f"Insufficient data for retraining: {len(training_data)} samples")
                return
            
            # Retrain ensemble model
            accuracy = await self.ensemble_model.retrain(training_data)
            
            # Retrain RL agent
            rl_reward = await self.rl_agent.update_from_history(
                await self.database.get_recent_trades(minutes=15)
            )
            
            self.logger.info(f"Retraining complete. Accuracy: {accuracy:.2%}, RL Reward: {rl_reward:.4f}")
            
            # Save models
            self.ensemble_model.save()
            self.rl_agent.save()
            
            self.last_retrain_time = datetime.now()
            
            # Send Discord notification
            if config.DISCORD_NOTIFICATION_ENABLED:
                await self.discord_notifier.send_model_update(
                    accuracy=accuracy,
                    rl_reward=rl_reward,
                    samples=len(training_data)
                )
            
        except Exception as e:
            self.logger.error(f"Error during retraining: {e}", exc_info=True)
    
    def _make_trading_decision(self, prediction: Dict[str, Any], 
                               rl_decision: Dict[str, Any],
                               features: Dict[str, float]) -> Dict[str, Any]:
        """Make final trading decision based on model outputs."""
        confidence = prediction['confidence']
        
        # Check confidence threshold
        if confidence < config.CONFIDENCE_THRESHOLD:
            return None
        
        # Check RL agent agreement
        if rl_decision['action'] == 'HOLD':
            return None
        
        # Combine predictions
        signal = {
            'action': prediction['action'],
            'confidence': confidence,
            'predicted_roi': prediction['expected_return'],
            'model_consensus': self._get_consensus_label(confidence),
            'rl_q_value': rl_decision['q_value'],
            'timestamp': datetime.now()
        }
        
        return signal
    
    def _get_consensus_label(self, confidence: float) -> str:
        """Get consensus label based on confidence."""
        if confidence >= 0.90:
            return "Very Strong"
        elif confidence >= 0.85:
            return "Strong"
        elif confidence >= 0.75:
            return "Moderate"
        else:
            return "Weak"
    
    async def _execute_trade(self, signal: Dict[str, Any], 
                            coinbase_data: Dict[str, Any],
                            polymarket_data: Dict[str, Any]):
        """Execute a paper trade and send notifications."""
        try:
            # Execute paper trade
            trade_result = await self.paper_trader.execute_trade(
                signal=signal,
                polymarket_orderbook=polymarket_data['orderbook']
            )
            
            # Store trade in database
            await self.database.store_trade(trade_result)
            
            # Update RL agent with immediate feedback
            self.rl_agent.store_experience(
                state=signal,
                action=signal['action'],
                reward=0,  # Will be updated when trade closes
                next_state=None
            )
            
            # Send Discord notification
            if config.DISCORD_NOTIFICATION_ENABLED:
                await self.discord_notifier.send_trade_signal(
                    signal=signal,
                    trade_result=trade_result,
                    coinbase_price=coinbase_data.get('price'),
                    polymarket_price=polymarket_data.get('price')
                )
            
            self.trade_count += 1
            self.logger.info(f"Trade #{self.trade_count} executed: {signal['action']} with {signal['confidence']:.1%} confidence")
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}", exc_info=True)
