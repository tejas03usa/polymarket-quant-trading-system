"""Paper trading execution engine."""
import logging
import uuid
from typing import Dict, Any
from datetime import datetime

import config


class PaperTrader:
    """Simulates trade execution with slippage calculations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.active_positions = {}
        self.trade_history = []
        self.total_pnl = 0.0
        
    async def execute_trade(self, signal: Dict[str, Any], 
                          polymarket_orderbook: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a paper trade."""
        try:
            trade_id = str(uuid.uuid4())[:8]
            action = signal['action']
            
            # Determine side
            if action == 'BUY':
                side = 'YES'
            elif action == 'SELL':
                side = 'NO'
            else:
                return None
            
            # Calculate execution details
            execution = self._simulate_fill(side, polymarket_orderbook)
            
            if not execution:
                self.logger.warning("Unable to simulate fill - insufficient liquidity")
                return None
            
            # Create trade record
            trade = {
                'trade_id': trade_id,
                'timestamp': datetime.now(),
                'action': action,
                'side': side,
                'quantity': execution['quantity'],
                'entry_price': execution['price'],
                'slippage': execution['slippage'],
                'expected_cost': config.POSITION_SIZE,
                'actual_cost': execution['total_cost'],
                'confidence': signal['confidence'],
                'predicted_roi': signal['predicted_roi'],
                'status': 'OPEN'
            }
            
            # Store position
            self.active_positions[trade_id] = trade
            self.trade_history.append(trade)
            
            self.logger.info(f"Paper trade executed: {action} {execution['quantity']} shares at ${execution['price']:.4f}")
            self.logger.info(f"  Slippage: {execution['slippage']:.2%}, Total cost: ${execution['total_cost']:.2f}")
            
            return trade
        
        except Exception as e:
            self.logger.error(f"Error executing paper trade: {e}", exc_info=True)
            return None
    
    def _simulate_fill(self, side: str, orderbook: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate order fill based on orderbook liquidity."""
        try:
            # Select appropriate side of orderbook
            if side == 'YES':
                orders = orderbook.get('asks', [])
            else:
                orders = orderbook.get('bids', [])
            
            if not orders:
                return None
            
            # Calculate how many shares we can buy with POSITION_SIZE
            remaining_capital = config.POSITION_SIZE
            total_shares = 0
            total_cost = 0
            weighted_price = 0
            
            for order in orders:
                price = order['price']
                available_size = order['size']
                
                # Calculate how many shares we can buy at this price level
                max_shares_at_level = remaining_capital / price
                shares_to_buy = min(max_shares_at_level, available_size)
                
                cost = shares_to_buy * price
                total_shares += shares_to_buy
                total_cost += cost
                weighted_price += price * shares_to_buy
                remaining_capital -= cost
                
                if remaining_capital <= 0:
                    break
            
            if total_shares == 0:
                return None
            
            # Calculate average execution price
            avg_price = weighted_price / total_shares
            
            # Calculate slippage
            best_price = orders[0]['price']
            slippage = (avg_price - best_price) / best_price if side == 'YES' else (best_price - avg_price) / best_price
            
            # Check slippage tolerance
            if abs(slippage) > config.MAX_SLIPPAGE:
                self.logger.warning(f"Slippage {slippage:.2%} exceeds maximum {config.MAX_SLIPPAGE:.2%}")
                return None
            
            return {
                'quantity': total_shares,
                'price': avg_price,
                'slippage': slippage,
                'total_cost': total_cost
            }
        
        except Exception as e:
            self.logger.error(f"Error simulating fill: {e}")
            return None
    
    def close_position(self, trade_id: str, current_price: float) -> Dict[str, Any]:
        """Close an open position."""
        if trade_id not in self.active_positions:
            return None
        
        position = self.active_positions[trade_id]
        
        # Calculate P&L
        if position['side'] == 'YES':
            pnl = (current_price - position['entry_price']) * position['quantity']
        else:
            pnl = (position['entry_price'] - current_price) * position['quantity']
        
        # Update position
        position['exit_price'] = current_price
        position['exit_timestamp'] = datetime.now()
        position['pnl'] = pnl
        position['roi'] = pnl / position['actual_cost']
        position['status'] = 'CLOSED'
        
        # Update totals
        self.total_pnl += pnl
        
        # Remove from active positions
        del self.active_positions[trade_id]
        
        self.logger.info(f"Position closed: {trade_id}, P&L: ${pnl:.2f} ({position['roi']:.2%})")
        
        return position
    
    def get_stats(self) -> Dict[str, Any]:
        """Get trading statistics."""
        closed_trades = [t for t in self.trade_history if t['status'] == 'CLOSED']
        
        if not closed_trades:
            return {
                'total_trades': len(self.trade_history),
                'closed_trades': 0,
                'open_positions': len(self.active_positions),
                'total_pnl': self.total_pnl,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0
            }
        
        winning_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in closed_trades if t.get('pnl', 0) < 0]
        
        return {
            'total_trades': len(self.trade_history),
            'closed_trades': len(closed_trades),
            'open_positions': len(self.active_positions),
            'total_pnl': self.total_pnl,
            'win_rate': len(winning_trades) / len(closed_trades) if closed_trades else 0,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'avg_win': sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0,
            'avg_loss': sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
        }
