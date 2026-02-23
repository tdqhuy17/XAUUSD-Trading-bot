#!/usr/bin/env python
"""
Example: Paper Trading Script

This example demonstrates how to run paper trading (simulated trading)
with a trained SAC agent.

Usage:
    python examples/example_live.py
"""

import sys
import os
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from trading_bot.config import Config, ENV_CONFIG
from trading_bot.sac_agent import SACAgentMTF
from trading_bot.trade_executor import PaperTrader


def create_live_data(n_bars: int = 300) -> pd.DataFrame:
    """
    Create simulated live data for paper trading demo.
    In practice, you would fetch real-time data from MT5 or your broker.
    """
    np.random.seed(int(time.time()) % 10000)
    
    # Generate price data
    start_price = 2000.0 + np.random.randn() * 50
    returns = np.random.randn(n_bars) * 0.001
    prices = start_price * np.cumprod(1 + returns)
    
    # Create OHLCV
    df = pd.DataFrame({
        'open': prices + np.random.randn(n_bars) * 1,
        'high': prices + np.abs(np.random.randn(n_bars)) * 2,
        'low': prices - np.abs(np.random.randn(n_bars)) * 2,
        'close': prices,
        'volume': np.random.randint(100, 1000, n_bars),
    })
    
    # Set datetime index
    df.index = pd.date_range(end=datetime.now(), periods=n_bars, freq='1min')
    df.index.name = 'time'
    
    return df


class SimplePaperTradingBot:
    """
    Simplified paper trading bot for demonstration.
    """
    
    def __init__(self, model_path: str = None, initial_cash: float = 10000):
        """
        Initialize the paper trading bot.
        
        Args:
            model_path: Path to trained model
            initial_cash: Starting capital
        """
        self.initial_cash = initial_cash
        self.trader = PaperTrader(
            initial_cash=initial_cash,
            transaction_fee=ENV_CONFIG.get('transaction_fee', 0.0003),
        )
        
        # Load agent if available
        self.agent = None
        if model_path and os.path.exists(model_path):
            try:
                self.agent = SACAgentMTF.from_pretrained(model_path)
                print(f"[Agent] Loaded model from: {model_path}")
            except Exception as e:
                print(f"[Agent] Failed to load model: {e}")
        
        self.log_file = Config.LOGS_DIR / 'paper_trading.log'
        
    def log(self, message: str):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry + '\n')
    
    def get_action(self, df: pd.DataFrame) -> tuple:
        """
        Get trading action from the agent.
        
        Args:
            df: Recent price data with features
            
        Returns:
            Tuple of (action_value, interpretation)
        """
        if self.agent is None:
            # Random action if no agent
            return np.random.randn() * 0.3, "RANDOM"
        
        # Create observation (simplified)
        mtf_obs = {
            'M1': np.zeros((60, 30), dtype=np.float32),
            'M5': np.zeros((24, 30), dtype=np.float32),
            'M15': np.zeros((16, 30), dtype=np.float32),
            'M30': np.zeros((8, 30), dtype=np.float32),
            'H1': np.zeros((6, 30), dtype=np.float32),
        }
        
        equity = self.trader.get_equity(df['close'].iloc[-1])
        portfolio = np.array([
            self.trader.balance / self.initial_cash,
            self.trader.position * df['close'].iloc[-1] / self.initial_cash,
            abs(self.trader.position * df['close'].iloc[-1]) / self.initial_cash,
            0, 0, 0
        ], dtype=np.float32)
        
        try:
            action = self.agent.select_action_deterministic(mtf_obs, portfolio)
            action_value = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        except Exception as e:
            self.log(f"[Error] Failed to get action: {e}")
            action_value = 0.0
        
        # Interpret action
        if action_value > 0.3:
            interpretation = "BUY"
        elif action_value < -0.3:
            interpretation = "SELL"
        elif abs(action_value) < 0.1:
            interpretation = "CLOSE"
        else:
            interpretation = "HOLD"
        
        return action_value, interpretation
    
    def run_step(self, df: pd.DataFrame):
        """Execute one trading step."""
        current_price = df['close'].iloc[-1]
        
        # Get action
        action_value, interpretation = self.get_action(df)
        
        # Execute trade
        if interpretation == "BUY" and not self.trader.has_position():
            self.trader.execute("BUY", current_price, size_pct=0.5)
            self.log(f"[Trade] BUY @ ${current_price:.2f} | Action: {action_value:+.3f}")
        
        elif interpretation == "SELL" and not self.trader.has_position():
            self.trader.execute("SELL", current_price, size_pct=0.5)
            self.log(f"[Trade] SELL @ ${current_price:.2f} | Action: {action_value:+.3f}")
        
        elif interpretation == "CLOSE" and self.trader.has_position():
            result = self.trader.execute("CLOSE", current_price)
            pnl = result.get('pnl', 0)
            self.log(f"[Trade] CLOSE @ ${current_price:.2f} | PnL: ${pnl:+.2f}")
        
        else:
            equity = self.trader.get_equity(current_price)
            self.log(f"[Hold] Price: ${current_price:.2f} | Equity: ${equity:.2f} | Action: {action_value:+.3f}")
    
    def run(self, n_steps: int = 10, interval: float = 1.0):
        """
        Run paper trading loop.
        
        Args:
            n_steps: Number of trading steps
            interval: Time between steps in seconds
        """
        self.log("=" * 50)
        self.log("Starting Paper Trading Session")
        self.log("=" * 50)
        self.log(f"Initial Cash: ${self.initial_cash:,.2f}")
        self.log(f"Steps: {n_steps}")
        self.log("-" * 50)
        
        for step in range(n_steps):
            self.log(f"\n--- Step {step + 1}/{n_steps} ---")
            
            # Get "live" data
            df = create_live_data(n_bars=300)
            
            # Run trading step
            self.run_step(df)
            
            # Wait for next step
            if step < n_steps - 1:
                time.sleep(interval)
        
        # Final summary
        final_df = create_live_data(n_bars=300)
        final_price = final_df['close'].iloc[-1]
        final_equity = self.trader.get_equity(final_price)
        
        self.log("\n" + "=" * 50)
        self.log("Paper Trading Session Complete")
        self.log("=" * 50)
        summary = self.trader.get_summary()
        self.log(f"Initial Cash   : ${summary['initial_cash']:,.2f}")
        self.log(f"Final Balance  : ${summary['balance']:,.2f}")
        self.log(f"Total PnL      : ${summary['total_pnl']:,.2f}")
        self.log(f"Return         : {summary['return_pct']:+.2f}%")
        self.log(f"Total Trades   : {summary['trade_count']}")
        self.log("=" * 50)


def main():
    """Main entry point."""
    print("=" * 60)
    print("  Example: Paper Trading with SAC Agent")
    print("=" * 60)
    
    # Check for model
    model_path = Config.MODELS_DIR / 'sac_mtf_best.pt'
    
    if not model_path.exists():
        print(f"\n[Warning] No trained model found at {model_path}")
        print("[Info] Paper trading will use random actions")
        model_path = None
    else:
        print(f"\n[Info] Using model: {model_path}")
    
    # Create and run bot
    bot = SimplePaperTradingBot(
        model_path=str(model_path) if model_path else None,
        initial_cash=10000,
    )
    
    # Run paper trading
    bot.run(n_steps=10, interval=1.0)
    
    print("\n[Note] This is a simplified demonstration.")
    print("       Use scripts/run_live.py for full paper/live trading.")


if __name__ == "__main__":
    main()