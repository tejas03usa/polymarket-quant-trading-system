"""Discord webhook notifier for trade signals."""
import logging
import aiohttp
from typing import Dict, Any
from datetime import datetime

import config


class DiscordNotifier:
    """Sends rich embed notifications to Discord."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.webhook_url = config.DISCORD_WEBHOOK_URL
    
    async def send_trade_signal(self, signal: Dict[str, Any], trade_result: Dict[str, Any],
                               coinbase_price: float, polymarket_price: float):
        """Send trade signal notification to Discord."""
        if not config.DISCORD_NOTIFICATION_ENABLED or not self.webhook_url:
            return
        
        try:
            # Determine color based on action
            if signal['action'] == 'BUY':
                color = config.DISCORD_EMBED_COLOR
                emoji = "🟢"
            else:
                color = config.DISCORD_EMBED_COLOR_SELL
                emoji = "🔴"
            
            # Create rich embed
            embed = {
                "title": f"{emoji} SIGNAL DETECTED | {signal['action']} \"Yes\" Share",
                "description": f"**Confidence:** {signal['confidence']:.1%}\n**Predicted ROI:** {signal['predicted_roi']:.2%}\n**🧠 Model Consensus:** {signal['model_consensus']}",
                "color": color,
                "fields": [
                    {
                        "name": "📊 Market Prices",
                        "value": f"Coinbase BTC: ${coinbase_price:,.2f}\nPolymarket: ${polymarket_price:.4f}",
                        "inline": True
                    },
                    {
                        "name": "💰 Trade Details",
                        "value": f"Shares: {trade_result['quantity']:.2f}\nEntry: ${trade_result['entry_price']:.4f}\nSlippage: {trade_result['slippage']:.2%}",
                        "inline": True
                    },
                    {
                        "name": "🎯 AI Analysis",
                        "value": f"RL Q-Value: {signal.get('rl_q_value', 0):.3f}\nModel Agreement: {signal.get('model_agreement', 0):.1%}",
                        "inline": False
                    }
                ],
                "footer": {
                    "text": f"Trade ID: {trade_result['trade_id']} | Polymarket Quant System"
                },
                "timestamp": datetime.now().isoformat()
            }
            
            payload = {
                "embeds": [embed]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    if response.status == 204:
                        self.logger.info("Trade signal sent to Discord")
                    else:
                        self.logger.warning(f"Discord webhook returned status {response.status}")
        
        except Exception as e:
            self.logger.error(f"Error sending Discord notification: {e}", exc_info=True)
    
    async def send_model_update(self, accuracy: float, rl_reward: float, samples: int):
        """Send model retraining update to Discord."""
        if not config.DISCORD_NOTIFICATION_ENABLED or not self.webhook_url:
            return
        
        try:
            # Choose color based on accuracy
            if accuracy >= config.MIN_ACCURACY_THRESHOLD:
                color = 0x00FF00  # Green
                emoji = "✅"
            else:
                color = 0xFFA500  # Orange
                emoji = "⚠️"
            
            embed = {
                "title": f"{emoji} Model Retraining Complete",
                "description": f"Models updated with latest market data",
                "color": color,
                "fields": [
                    {
                        "name": "🎯 Accuracy",
                        "value": f"{accuracy:.2%}",
                        "inline": True
                    },
                    {
                        "name": "🤖 RL Reward",
                        "value": f"{rl_reward:.4f}",
                        "inline": True
                    },
                    {
                        "name": "📈 Training Samples",
                        "value": f"{samples:,}",
                        "inline": True
                    }
                ],
                "footer": {
                    "text": "Auto-retraining every 15 minutes"
                },
                "timestamp": datetime.now().isoformat()
            }
            
            payload = {
                "embeds": [embed]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    if response.status == 204:
                        self.logger.info("Model update sent to Discord")
        
        except Exception as e:
            self.logger.error(f"Error sending model update: {e}", exc_info=True)
    
    async def send_error_alert(self, error_message: str):
        """Send error alert to Discord."""
        if not config.DISCORD_NOTIFICATION_ENABLED or not self.webhook_url:
            return
        
        try:
            embed = {
                "title": "🚨 System Error",
                "description": error_message[:2000],  # Discord limit
                "color": 0xFF0000,  # Red
                "timestamp": datetime.now().isoformat()
            }
            
            payload = {"embeds": [embed]}
            
            async with aiohttp.ClientSession() as session:
                await session.post(self.webhook_url, json=payload)
        
        except Exception as e:
            self.logger.error(f"Error sending error alert: {e}")
