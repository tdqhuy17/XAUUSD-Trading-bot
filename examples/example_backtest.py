#!/usr/bin/env python
"""
Example: Backtesting Script

This example demonstrates how to evaluate a trained SAC agent
on historical test data.

Usage:
    python examples/example_backtest.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from trading_bot.config import Config
from trading_bot.sac_agent import SACAgentMTF


def create_sample_data(n_bars: int = 1000) -> pd.DataFrame:
    """
    Create sample price data for demonstration.
    In practice, you would load real XAUUSD data.
    """
    np.random.seed(42)
    
    # Generate price data (random walk)
    start_price = 2000.0
    returns = np.random.randn(n_bars) * 0.002
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
    df.index = pd.date_range(start='2025-01-01', periods=n_bars, freq='1min')
    df.index.name = 'time'
    
    return df


def simple_backtest(agent, prices: np.ndarray, initial_cash: float = 10000) -> dict:
    """
    Simple backtest implementation.
    
    Args:
        agent: Trained SAC agent
        prices: Array of closing prices
        initial_cash: Starting capital
        
    Returns:
        Dictionary with backtest results
    """
    cash = initial_cash
    position = 0.0
    position_price = 0.0
    equity_curve = [initial_cash]
    trades = []
    
    for i in range(len(prices) - 1):
        price = prices[i]
        
        # Create simple observation (in practice, use full features)
        obs = np.zeros((60, 30), dtype=np.float32)  # Dummy observation
        portfolio = np.array([cash / initial_cash, 0, 0, 0, 0, 0], dtype=np.float32)
        
        # Get action from agent
        mtf_obs = {
            'M1': obs,
            'M5': np.zeros((24, 30), dtype=np.float32),
            'M15': np.zeros((16, 30), dtype=np.float32),
            'M30': np.zeros((8, 30), dtype=np.float32),
            'H1': np.zeros((6, 30), dtype=np.float32),
        }
        
        try:
            action = agent.select_action_deterministic(mtf_obs, portfolio)
            action_value = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        except Exception:
            action_value = 0.0
        
        # Execute action
        # Long
        if action_value > 0.3 and position <= 0:
            if position < 0:
                # Close short
                pnl = (position_price - price) * abs(position)
                cash += abs(position) * price + pnl
                trades.append({'type': 'close_short', 'price': price, 'pnl': pnl})
            
            # Open long
            shares = (cash * 0.5) / price
            cash -= shares * price
            position = shares
            position_price = price
            trades.append({'type': 'buy', 'price': price, 'shares': shares})
        
        # Short
        elif action_value < -0.3 and position >= 0:
            if position > 0:
                # Close long
                pnl = (price - position_price) * position
                cash += position * price + pnl
                trades.append({'type': 'close_long', 'price': price, 'pnl': pnl})
            
            # Open short
            shares = (cash * 0.5) / price
            position = -shares
            position_price = price
            trades.append({'type': 'short', 'price': price, 'shares': shares})
        
        # Close
        elif abs(action_value) < 0.1 and position != 0:
            if position > 0:
                pnl = (price - position_price) * position
                cash += position * price + pnl
                trades.append({'type': 'close_long', 'price': price, 'pnl': pnl})
            elif position < 0:
                pnl = (position_price - price) * abs(position)
                cash += abs(position) * price + pnl
                trades.append({'type': 'close_short', 'price': price, 'pnl': pnl})
            position = 0.0
        
        # Calculate equity
        equity = cash + position * price
        equity_curve.append(equity)
    
    # Close final position
    final_price = prices[-1]
    if position > 0:
        cash += position * final_price
    elif position < 0:
        cash += abs(position) * final_price
    
    final_equity = cash
    total_return = (final_equity - initial_cash) / initial_cash * 100
    
    # Calculate metrics
    equity_curve = np.array(equity_curve)
    returns = np.diff(equity_curve) / equity_curve[:-1]
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 24 * 60)  # Annualized
    
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak
    max_drawdown = np.max(drawdown) * 100
    
    winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
    total_trades = len([t for t in trades if 'pnl' in t])
    win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
    
    return {
        'initial_cash': initial_cash,
        'final_equity': final_equity,
        'total_return_pct': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown_pct': max_drawdown,
        'win_rate_pct': win_rate,
        'total_trades': total_trades,
        'equity_curve': equity_curve,
        'trades': trades,
    }


def main():
    """Main backtest example."""
    print("=" * 60)
    print("  Example: Backtesting SAC Agent")
    print("=" * 60)
    
    # ==========================================
    # Step 1: Load or Create Test Data
    # ==========================================
    print("\n[Step 1] Creating sample test data...")
    print("  (In practice, load real XAUUSD data)")
    
    df = create_sample_data(n_bars=1000)
    prices = df['close'].values
    
    print(f"  Test period: {df.index[0]} to {df.index[-1]}")
    print(f"  Bars: {len(df)}")
    print(f"  Price range: ${prices.min():.2f} - ${prices.max():.2f}")
    
    # ==========================================
    # Step 2: Load Trained Model
    # ==========================================
    print("\n[Step 2] Loading trained model...")
    
    model_path = Config.MODELS_DIR / 'sac_mtf_best.pt'
    
    if model_path.exists():
        agent = SACAgentMTF.from_pretrained(str(model_path))
        print(f"  Loaded: {model_path}")
    else:
        print(f"  Model not found at {model_path}")
        print("  Creating untrained agent for demonstration...")
        
        from trading_bot.config import MTF_CONFIG, SAC_CONFIG
        
        # Create minimal config
        config = {
            'timeframes': ['M1', 'M5', 'M15', 'M30', 'H1'],
            'bars_per_tf': MTF_CONFIG['bars_per_tf'],
            'encoder_dims': MTF_CONFIG['encoder_dims'],
            'feature_dims': {tf: 30 for tf in ['M1', 'M5', 'M15', 'M30', 'H1']},
            'attention_dim': 256,
            'attention_heads': 4,
            'hidden_dim': 128,
            'n_layers': 2,
            'dropout': 0.1,
            'rnn_type': 'gru',
            **{k: SAC_CONFIG[k] for k in ['learning_rate', 'actor_lr', 'critic_lr', 
                                          'alpha_lr', 'gamma', 'tau', 'batch_size']}
        }
        agent = SACAgentMTF(config, action_dim=1)
        print("  Created untrained agent (results will be random)")
    
    # ==========================================
    # Step 3: Run Backtest
    # ==========================================
    print("\n[Step 3] Running backtest...")
    
    results = simple_backtest(agent, prices)
    
    # ==========================================
    # Step 4: Display Results
    # ==========================================
    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Initial Capital : ${results['initial_cash']:,.2f}")
    print(f"  Final Equity    : ${results['final_equity']:,.2f}")
    print(f"  Total Return    : {results['total_return_pct']:+.2f}%")
    print("-" * 60)
    print(f"  Sharpe Ratio    : {results['sharpe_ratio']:.4f}")
    print(f"  Max Drawdown    : {results['max_drawdown_pct']:.2f}%")
    print(f"  Win Rate        : {results['win_rate_pct']:.1f}%")
    print(f"  Total Trades    : {results['total_trades']}")
    print("=" * 60)
    
    # ==========================================
    # Step 5: Interpretation
    # ==========================================
    print("\n[Interpretation]")
    ret = results['total_return_pct']
    sharpe = results['sharpe_ratio']
    
    if ret > 5 and sharpe > 0.5:
        verdict = "✅ GOOD - Agent shows positive performance"
    elif ret > 0:
        verdict = "⚠️  MARGINAL - Positive but weak performance"
    elif ret > -10:
        verdict = "⚠️  POOR - Agent underperforming"
    else:
        verdict = "❌ BAD - Agent needs more training"
    
    print(f"  {verdict}")
    print("\n  Note: This is a simplified example. Use the full scripts/backtest.py")
    print("        for comprehensive evaluation with real data.")


if __name__ == "__main__":
    main()