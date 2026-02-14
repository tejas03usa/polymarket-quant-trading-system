"""Reinforcement Learning agent using Q-Learning."""
import numpy as np
import pickle
import logging
from pathlib import Path
from collections import deque
from typing import Dict, Any, Optional
from datetime import datetime

import config


class RLAgent:
    """Simple Q-Learning agent for trade execution decisions."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Q-table (simplified state-action mapping)
        self.q_table = {}
        
        # Hyperparameters
        self.learning_rate = config.RL_LEARNING_RATE
        self.discount_factor = config.RL_DISCOUNT_FACTOR
        self.epsilon = config.RL_EPSILON
        
        # Experience replay buffer
        self.experience_buffer = deque(maxlen=1000)
        
        # Performance tracking
        self.total_reward = 0.0
        self.episode_rewards = []
        
    def load_or_initialize(self):
        """Load existing Q-table or initialize new one."""
        model_path = Path(config.MODEL_PATH) / 'rl_agent.pkl'
        
        try:
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.q_table = data['q_table']
                    self.total_reward = data.get('total_reward', 0.0)
                    self.episode_rewards = data.get('episode_rewards', [])
                self.logger.info("Loaded existing RL agent")
            else:
                self.logger.info("Initialized new RL agent")
        
        except Exception as e:
            self.logger.error(f"Error loading RL agent: {e}", exc_info=True)
    
    def get_action(self, features: Dict[str, float], 
                   model_prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Get action from RL agent."""
        try:
            # Create state representation
            state = self._create_state(features, model_prediction)
            
            # Epsilon-greedy action selection
            if np.random.random() < self.epsilon:
                # Explore: random action
                action = np.random.choice(['BUY', 'SELL', 'HOLD'])
            else:
                # Exploit: choose best action from Q-table
                action = self._get_best_action(state)
            
            # Get Q-value
            q_value = self._get_q_value(state, action)
            
            return {
                'action': action,
                'q_value': q_value,
                'state': state,
                'exploration': np.random.random() < self.epsilon
            }
        
        except Exception as e:
            self.logger.error(f"Error getting action: {e}", exc_info=True)
            return {'action': 'HOLD', 'q_value': 0.0, 'state': None, 'exploration': False}
    
    def _create_state(self, features: Dict[str, float], 
                     model_prediction: Dict[str, Any]) -> str:
        """Create discrete state representation."""
        # Extract key features for state
        confidence = model_prediction.get('confidence', 0.5)
        action = model_prediction.get('action', 'HOLD')
        
        # Get market regime features
        volatility = features.get('realized_volatility', 0)
        trend = features.get('trend_strength', 0)
        ob_imbalance = features.get('cb_ob_imbalance', 0)
        
        # Discretize features
        conf_bucket = 'high' if confidence > 0.8 else 'med' if confidence > 0.6 else 'low'
        vol_bucket = 'high' if volatility > 0.5 else 'med' if volatility > 0.2 else 'low'
        trend_bucket = 'up' if trend > 0.001 else 'down' if trend < -0.001 else 'flat'
        imb_bucket = 'bid' if ob_imbalance > 0.1 else 'ask' if ob_imbalance < -0.1 else 'bal'
        
        # Create state string
        state = f"{action}_{conf_bucket}_{vol_bucket}_{trend_bucket}_{imb_bucket}"
        return state
    
    def _get_best_action(self, state: str) -> str:
        """Get best action for state from Q-table."""
        if state not in self.q_table:
            return 'HOLD'
        
        actions = ['BUY', 'SELL', 'HOLD']
        q_values = [self.q_table[state].get(a, 0.0) for a in actions]
        best_action_idx = np.argmax(q_values)
        
        return actions[best_action_idx]
    
    def _get_q_value(self, state: str, action: str) -> float:
        """Get Q-value for state-action pair."""
        if state not in self.q_table:
            self.q_table[state] = {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0}
        
        return self.q_table[state].get(action, 0.0)
    
    def store_experience(self, state: Dict[str, Any], action: str, 
                        reward: float, next_state: Optional[Dict[str, Any]]):
        """Store experience in replay buffer."""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'timestamp': datetime.now()
        }
        self.experience_buffer.append(experience)
    
    async def update_from_history(self, trade_history: list) -> float:
        """Update Q-values from recent trade outcomes."""
        try:
            if not trade_history:
                return 0.0
            
            total_reward = 0.0
            updates = 0
            
            for trade in trade_history:
                # Extract trade information
                state_str = trade.get('state')
                action = trade.get('action')
                outcome = trade.get('outcome', 0)  # Profit/loss
                
                if not state_str or not action:
                    continue
                
                # Calculate reward (+1 for profit, -1 for loss)
                if outcome > 0:
                    reward = 1.0
                elif outcome < 0:
                    reward = -1.0
                else:
                    reward = 0.0
                
                # Update Q-value
                self._update_q_value(state_str, action, reward, None)
                
                total_reward += reward
                updates += 1
            
            avg_reward = total_reward / updates if updates > 0 else 0.0
            self.episode_rewards.append(avg_reward)
            self.total_reward += total_reward
            
            self.logger.info(f"RL agent updated with {updates} experiences. Avg reward: {avg_reward:.3f}")
            
            # Decay epsilon (reduce exploration over time)
            self.epsilon = max(0.01, self.epsilon * 0.995)
            
            return avg_reward
        
        except Exception as e:
            self.logger.error(f"Error updating from history: {e}", exc_info=True)
            return 0.0
    
    def _update_q_value(self, state: str, action: str, reward: float, 
                       next_state: Optional[str]):
        """Update Q-value using Q-learning update rule."""
        # Initialize state in Q-table if not exists
        if state not in self.q_table:
            self.q_table[state] = {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0}
        
        # Get current Q-value
        current_q = self.q_table[state][action]
        
        # Get max Q-value for next state
        if next_state and next_state in self.q_table:
            max_next_q = max(self.q_table[next_state].values())
        else:
            max_next_q = 0.0
        
        # Q-learning update rule
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        # Update Q-table
        self.q_table[state][action] = new_q
    
    def save(self):
        """Save RL agent to disk."""
        try:
            model_path = Path(config.MODEL_PATH) / 'rl_agent.pkl'
            
            data = {
                'q_table': self.q_table,
                'total_reward': self.total_reward,
                'episode_rewards': self.episode_rewards[-100:],  # Keep last 100
                'epsilon': self.epsilon
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(data, f)
            
            self.logger.info(f"RL agent saved. Q-table size: {len(self.q_table)}")
        
        except Exception as e:
            self.logger.error(f"Error saving RL agent: {e}", exc_info=True)
