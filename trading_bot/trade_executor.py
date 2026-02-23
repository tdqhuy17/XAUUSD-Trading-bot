"""
Trade Execution Module
- Paper trading for simulation
- MT5 integration for live trading
"""

import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any


class PaperTrader:
    """
    Paper trading simulator
    Simulates trades without real execution
    """
    
    def __init__(
        self, 
        initial_cash: float = 10000,
        transaction_fee: float = 0.0002,
    ):
        self.initial_cash = initial_cash
        self.balance = initial_cash
        self.transaction_fee = transaction_fee
        
        # Position tracking
        self.position = 0.0  # Positive = long, negative = short
        self.position_price = 0.0
        self.position_value = 0.0
        
        # Statistics
        self.trade_count = 0
        self.total_pnl = 0.0
        self.trades = []
        
    def has_position(self) -> bool:
        """Check if currently in a position"""
        return abs(self.position) > 0.001
    
    def execute(self, action: str, price: float, size_pct: float = 1.0) -> Dict[str, Any]:
        """
        Execute a paper trade
        
        Args:
            action: "BUY", "SELL", or "CLOSE"
            price: Current market price
            size_pct: Position size as percentage of balance
            
        Returns:
            Dictionary with trade result
        """
        result = {
            'action': action,
            'price': price,
            'success': False,
            'message': '',
            'pnl': 0.0,
        }
        
        if action == "BUY":
            # Close short position if any
            if self.position < 0:
                pnl = (self.position_price - price) * abs(self.position)
                self.balance += abs(self.position) * price * (1 - self.transaction_fee)
                self.total_pnl += pnl
                result['pnl'] = pnl
                result['message'] = f"Covered short at ${price:.2f}, PnL: ${pnl:+.2f}"
                self.position = 0.0
                self.position_price = 0.0
            
            # Open long position
            if self.balance > 10:
                trade_value = self.balance * size_pct
                shares = trade_value / price
                cost = trade_value * (1 + self.transaction_fee)
                self.balance -= cost
                
                if self.position > 0:
                    # Add to existing position
                    total_shares = self.position + shares
                    self.position_price = (
                        self.position * self.position_price + shares * price
                    ) / total_shares
                    self.position = total_shares
                else:
                    self.position = shares
                    self.position_price = price
                
                result['success'] = True
                result['message'] = f"Bought {shares:.4f} shares at ${price:.2f}"
                self.trade_count += 1
            else:
                result['message'] = "Insufficient balance"
                
        elif action == "SELL":
            # Close long position if any
            if self.position > 0:
                pnl = (price - self.position_price) * self.position
                self.balance += self.position * price * (1 - self.transaction_fee)
                self.total_pnl += pnl
                result['pnl'] = pnl
                result['message'] = f"Sold at ${price:.2f}, PnL: ${pnl:+.2f}"
                self.position = 0.0
                self.position_price = 0.0
            
            # Open short position
            if self.balance > 10:
                trade_value = self.balance * size_pct
                shares = trade_value / price
                self.balance += trade_value * (1 - self.transaction_fee)
                
                if self.position < 0:
                    # Add to existing short
                    total_shares = abs(self.position) + shares
                    self.position_price = (
                        abs(self.position) * self.position_price + shares * price
                    ) / total_shares
                    self.position = -total_shares
                else:
                    self.position = -shares
                    self.position_price = price
                
                result['success'] = True
                result['message'] = f"Shorted {shares:.4f} shares at ${price:.2f}"
                self.trade_count += 1
            else:
                result['message'] = "Insufficient balance"
                
        elif action == "CLOSE":
            if self.position > 0:
                pnl = (price - self.position_price) * self.position
                self.balance += self.position * price * (1 - self.transaction_fee)
                self.total_pnl += pnl
                result['pnl'] = pnl
                result['success'] = True
                result['message'] = f"Closed long at ${price:.2f}, PnL: ${pnl:+.2f}"
                self.position = 0.0
                self.position_price = 0.0
                self.trade_count += 1
            elif self.position < 0:
                pnl = (self.position_price - price) * abs(self.position)
                self.balance += abs(self.position) * price * (1 - self.transaction_fee)
                self.total_pnl += pnl
                result['pnl'] = pnl
                result['success'] = True
                result['message'] = f"Closed short at ${price:.2f}, PnL: ${pnl:+.2f}"
                self.position = 0.0
                self.position_price = 0.0
                self.trade_count += 1
            else:
                result['message'] = "No position to close"
        
        # Record trade
        if result['success']:
            self.trades.append({
                'time': datetime.now().isoformat(),
                'action': action,
                'price': price,
                'pnl': result['pnl'],
                'balance': self.balance,
                'position': self.position,
            })
        
        return result
    
    def get_equity(self, current_price: float) -> float:
        """Calculate current equity"""
        return self.balance + (self.position * current_price)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get trading summary"""
        return {
            'initial_cash': self.initial_cash,
            'balance': self.balance,
            'position': self.position,
            'position_price': self.position_price,
            'total_pnl': self.total_pnl,
            'trade_count': self.trade_count,
            'return_pct': (self.balance - self.initial_cash) / self.initial_cash * 100,
        }


class MT5Trader:
    """
    Live trading via MetaTrader5
    """
    
    def __init__(
        self,
        symbol: str = "XAUUSD",
        lot_size: float = 0.01,
        magic_number: int = 234567,
        deviation: int = 20,
    ):
        self.symbol = symbol
        self.lot_size = lot_size
        self.magic_number = magic_number
        self.deviation = deviation
        
        self.mt5 = None
        self.trade_count = 0
        
    def initialize(self) -> bool:
        """Initialize MT5 connection"""
        try:
            import MetaTrader5 as mt5
            self.mt5 = mt5
            
            if not mt5.initialize():
                print(f"MT5 initialization failed: {mt5.last_error()}")
                return False
            
            # Check symbol
            if not mt5.symbol_select(self.symbol, True):
                print(f"Failed to select symbol: {self.symbol}")
                return False
            
            return True
            
        except ImportError:
            print("MetaTrader5 package not installed")
            return False
    
    def has_position(self) -> bool:
        """Check if currently in a position"""
        if self.mt5 is None:
            return False
        
        positions = self.mt5.positions_get(symbol=self.symbol)
        if positions is None or len(positions) == 0:
            return False
        
        # Filter by magic number
        for pos in positions:
            if pos.magic == self.magic_number:
                return True
        
        return False
    
    def execute(self, action: str, price: float, size_pct: float = 1.0) -> Dict[str, Any]:
        """
        Execute a live trade
        
        Args:
            action: "BUY", "SELL", or "CLOSE"
            price: Current market price (used for reference)
            size_pct: Position size multiplier
            
        Returns:
            Dictionary with trade result
        """
        result = {
            'action': action,
            'price': price,
            'success': False,
            'message': '',
        }
        
        if self.mt5 is None:
            result['message'] = "MT5 not initialized"
            return result
        
        tick = self.mt5.symbol_info_tick(self.symbol)
        if tick is None:
            result['message'] = "Failed to get tick"
            return result
        
        # Calculate lot size - ensure minimum of lot_size and round to valid steps
        lot = self.lot_size * size_pct
        lot = max(lot, self.lot_size)  # Minimum lot size
        lot = round(lot, 2)  # Round to 2 decimal places (MT5 standard)
        
        if action == "BUY":
            request = {
                "action": self.mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": lot,
                "type": self.mt5.ORDER_TYPE_BUY,
                "price": tick.ask,
                "deviation": self.deviation,
                "magic": self.magic_number,
                "comment": "SAC Bot Buy",
                "type_time": self.mt5.ORDER_TIME_GTC,
                "type_filling": self.mt5.ORDER_FILLING_IOC,
            }
            
            res = self.mt5.order_send(request)
            if res.retcode == self.mt5.TRADE_RETCODE_DONE:
                result['success'] = True
                result['message'] = f"Buy order executed at {tick.ask}"
                self.trade_count += 1
            else:
                result['message'] = f"Order failed: {res.comment}"
                
        elif action == "SELL":
            request = {
                "action": self.mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": lot,
                "type": self.mt5.ORDER_TYPE_SELL,
                "price": tick.bid,
                "deviation": self.deviation,
                "magic": self.magic_number,
                "comment": "SAC Bot Sell",
                "type_time": self.mt5.ORDER_TIME_GTC,
                "type_filling": self.mt5.ORDER_FILLING_IOC,
            }
            
            res = self.mt5.order_send(request)
            if res.retcode == self.mt5.TRADE_RETCODE_DONE:
                result['success'] = True
                result['message'] = f"Sell order executed at {tick.bid}"
                self.trade_count += 1
            else:
                result['message'] = f"Order failed: {res.comment}"
                
        elif action == "CLOSE":
            positions = self.mt5.positions_get(symbol=self.symbol)
            if positions is None or len(positions) == 0:
                result['message'] = "No positions to close"
                return result
            
            for pos in positions:
                if pos.magic != self.magic_number:
                    continue
                
                close_type = self.mt5.ORDER_TYPE_SELL if pos.type == 0 else self.mt5.ORDER_TYPE_BUY
                close_price = tick.bid if pos.type == 0 else tick.ask
                
                request = {
                    "action": self.mt5.TRADE_ACTION_DEAL,
                    "symbol": self.symbol,
                    "volume": pos.volume,
                    "position": pos.ticket,
                    "type": close_type,
                    "price": close_price,
                    "deviation": self.deviation,
                    "magic": self.magic_number,
                    "comment": "SAC Bot Close",
                    "type_time": self.mt5.ORDER_TIME_GTC,
                    "type_filling": self.mt5.ORDER_FILLING_IOC,
                }
                
                res = self.mt5.order_send(request)
                if res.retcode == self.mt5.TRADE_RETCODE_DONE:
                    result['success'] = True
                    result['message'] = f"Position {pos.ticket} closed"
                    self.trade_count += 1
                else:
                    result['message'] = f"Close failed: {res.comment}"
        
        return result
    
    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Get account information"""
        if self.mt5 is None:
            return None
        
        info = self.mt5.account_info()
        if info is None:
            return None
        
        return {
            'balance': info.balance,
            'equity': info.equity,
            'margin': info.margin,
            'free_margin': info.margin_free,
            'profit': info.profit,
        }
    
    def shutdown(self):
        """Shutdown MT5 connection"""
        if self.mt5:
            self.mt5.shutdown()