"""
SAC Model Training and Inference API
- Pure PyTorch implementation (no stable-baselines)
- CUDA GPU acceleration
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path

from .config import SAC_CONFIG, ENV_CONFIG
from .environment import TradingEnv
from .sac_agent import SACAgent
from .features import get_feature_columns


def train_sac(
    train_df,
    model_path: str = None,
    eval_df=None,
    total_timesteps: int = None,
    config: dict = None,
    verbose: int = 1,
    callback=None,
    device: str = 'auto',
) -> SACAgent:
    """
    Train SAC agent on trading environment
    
    Args:
        train_df: Training data DataFrame
        model_path: Path to save model (optional)
        eval_df: Evaluation data (optional, for logging)
        total_timesteps: Number of training steps
        config: SAC configuration dict
        verbose: Verbosity level (0=silent, 1=progress)
        callback: Optional callback function
        
    Returns:
        Trained SACAgent instance
    """
    # Use default config if not provided
    cfg = config or SAC_CONFIG.copy()
    total_timesteps = total_timesteps or cfg.get('train_steps', 50000)
    
    # Create environment
    env = TradingEnv(train_df)
    
    # Get dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    if verbose >= 1:
        print(f"\n{'='*60}")
        print(f"SAC Training Configuration")
        print(f"{'='*60}")
        print(f"  State dim: {state_dim}")
        print(f"  Action dim: {action_dim}")
        print(f"  Total steps: {total_timesteps:,}")
        print(f"  Batch size: {cfg.get('batch_size', 256)}")
        print(f"  Learning rate: {cfg.get('learning_rate', 3e-4)}")
        print(f"  Gamma: {cfg.get('gamma', 0.99)}")
        print(f"  Tau: {cfg.get('tau', 0.005)}")
        print(f"{'='*60}\n")
    
    # Create agent on the specified device
    agent = SACAgent(state_dim, action_dim, cfg, device=device)
    
    # Training callback wrapper
    episode_rewards = []
    episode_lengths = []
    best_reward = float('-inf')
    
    def training_callback(step, total_steps, episode_reward, episode_length, info):
        nonlocal best_reward
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if episode_reward > best_reward:
            best_reward = episode_reward

        # Per-episode verbose logging (every 10 episodes when verbose=2+)
        if verbose >= 2 and len(episode_rewards) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"  Episode {len(episode_rewards):4d} | "
                  f"Avg Reward (10): {avg_reward:+.2f} | "
                  f"Last Reward: {episode_reward:+.2f} | "
                  f"Steps: {step:,}/{total_steps:,}")

        if callback:
            callback(step, total_steps, episode_reward, episode_length, info)
    
    # Train
    if verbose >= 1:
        print(f"Starting training for {total_timesteps:,} steps...")
    
    history = agent.learn(
        total_timesteps,
        env,
        callback=training_callback,
        checkpoint_dir=os.path.dirname(model_path) if model_path and os.path.dirname(model_path) else 'models',
        best_model_path=(model_path + '.best') if model_path else None,
    )
    
    # Save final model
    if model_path:
        agent.save(model_path)
        if verbose >= 1:
            print(f"\nModel saved to: {model_path}")
    
    # Print summary
    if verbose >= 1:
        print(f"\n{'='*60}")
        print(f"Training Summary")
        print(f"{'='*60}")
        print(f"  Total episodes: {len(episode_rewards)}")
        print(f"  Best reward: {best_reward:+.2f}")
        print(f"  Avg reward (last 10): {np.mean(episode_rewards[-10:]):+.2f}")
        print(f"  Avg episode length: {np.mean(episode_lengths):.1f}")
        print(f"{'='*60}\n")
    
    return agent


def load_model(model_path: str, df=None, device: str = 'auto') -> SACAgent:
    """
    Load trained SAC model
    
    Args:
        model_path: Path to saved model
        df: Optional DataFrame to create environment
        device: Device to load model on
        
    Returns:
        SACAgent instance
    """
    # Add extension if not present
    if not model_path.endswith('.pt') and not model_path.endswith('.pth'):
        if os.path.exists(model_path + '.pt'):
            model_path = model_path + '.pt'
        elif os.path.exists(model_path + '.pth'):
            model_path = model_path + '.pth'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load agent
    agent = SACAgent.from_pretrained(model_path, device=device)
    
    print(f"Model loaded from: {model_path}")
    return agent


def predict(
    agent: SACAgent, 
    obs: np.ndarray, 
    deterministic: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get prediction from model
    
    Args:
        agent: SACAgent instance
        obs: Observation array
        deterministic: Use deterministic policy
        
    Returns:
        Tuple of (action, action_deterministic)
    """
    if deterministic:
        action = agent.select_action_deterministic(obs)
    else:
        action = agent.select_action(obs)
    
    action_det = agent.select_action_deterministic(obs)
    
    return action, action_det


def evaluate_model(
    agent: SACAgent, 
    test_df, 
    render: bool = False,
    verbose: int = 1,
) -> Tuple[pd.DataFrame, float]:
    """
    Evaluate model on test data
    
    Args:
        agent: SACAgent instance
        test_df: Test data DataFrame
        render: Print step-by-step details
        verbose: Verbosity level
        
    Returns:
        Tuple of (results DataFrame, profit percentage)
    """
    env = TradingEnv(test_df)
    
    obs, _ = env.reset()
    done = False
    history = []
    
    while not done:
        action = agent.select_action_deterministic(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        history.append(info)
        
        if render and info:
            print(f"Action: {info.get('action', 'N/A')}, "
                  f"Assets: ${info.get('assets', 0):.2f}")
    
    df_results = pd.DataFrame(history)
    
    # Calculate performance
    final_assets = df_results['assets'].iloc[-1]
    initial_assets = ENV_CONFIG.get('initial_cash', 10000)
    profit_pct = ((final_assets - initial_assets) / initial_assets) * 100
    
    if verbose >= 1:
        print(f"\n{'='*50}")
        print(f"Evaluation Results")
        print(f"{'='*50}")
        print(f"Initial: ${initial_assets:,.2f}")
        print(f"Final: ${final_assets:,.2f}")
        print(f"Profit: {profit_pct:+.2f}%")
        print(f"Trades: {df_results['trade_count'].iloc[-1] if 'trade_count' in df_results else 'N/A'}")
        print(f"{'='*50}")
    
    return df_results, profit_pct


def continuous_learning(
    model_path: str, 
    new_data_df, 
    fine_tune_steps: int = None,
    config: dict = None,
    verbose: int = 1,
) -> SACAgent:
    """
    Fine-tune existing model with new data
    
    Args:
        model_path: Path to saved model
        new_data_df: New training data
        fine_tune_steps: Number of fine-tuning steps
        config: Optional config override
        verbose: Verbosity level
        
    Returns:
        Updated SACAgent instance
    """
    cfg = config or SAC_CONFIG.copy()
    fine_tune_steps = fine_tune_steps or cfg.get('fine_tune_steps', 10000)
    
    # Load existing model
    agent = load_model(model_path, new_data_df)
    
    # Create environment with new data
    env = TradingEnv(new_data_df)
    
    if verbose >= 1:
        print(f"\nFine-tuning for {fine_tune_steps:,} steps...")
    
    # Fine-tune
    agent.learn(fine_tune_steps, env)
    
    # Save updated model
    agent.save(model_path)
    
    if verbose >= 1:
        print(f"Updated model saved to: {model_path}")
    
    return agent


def get_action_interpretation(action: float) -> str:
    """Interpret action value as trading signal"""
    if isinstance(action, np.ndarray):
        action = action[0]
    
    if action < -0.1:
        return "SHORT"
    elif action > 0.1:
        return "LONG"
    else:
        return "HOLD"


def backtest(
    agent: SACAgent,
    df,
    initial_cash: float = 10000,
    transaction_fee: float = 0.0002,
    verbose: int = 1,
) -> Dict[str, Any]:
    """
    Run backtest with detailed metrics
    
    Args:
        agent: SACAgent instance
        df: Price data DataFrame
        initial_cash: Starting capital
        transaction_fee: Fee per trade
        verbose: Verbosity level
        
    Returns:
        Dictionary with backtest results
    """
    env = TradingEnv(df)
    env.initial_cash = initial_cash
    env.transaction_fee = transaction_fee
    
    obs, _ = env.reset()
    done = False
    trades = []
    equity_curve = [initial_cash]
    
    step = 0
    while not done:
        action = agent.select_action_deterministic(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1
        
        equity_curve.append(info.get('assets', equity_curve[-1]))
        
        if info.get('action') != 'Hold':
            trades.append({
                'step': step,
                'action': info.get('action'),
                'price': info.get('price'),
                'position': info.get('position', 0),
                'cash': info.get('cash', 0),
                'assets': info.get('assets'),
                'pnl': info.get('realized_pnl', 0),
            })
    
    equity_curve = np.array(equity_curve)
    returns = np.diff(equity_curve) / equity_curve[:-1]
    
    # Calculate metrics
    final_value = equity_curve[-1]
    total_return = (final_value - initial_cash) / initial_cash * 100
    
    # Sharpe ratio (annualized, assuming 252 trading days)
    if len(returns) > 1 and returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(252 * 96)  # M15 bars
    else:
        sharpe = 0
    
    # Max drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak
    max_drawdown = drawdown.max() * 100
    
    # Win rate
    if trades:
        winning = sum(1 for t in trades if t['pnl'] > 0)
        win_rate = winning / len(trades) * 100
    else:
        win_rate = 0
    
    results = {
        'initial_cash': initial_cash,
        'final_value': final_value,
        'total_return_pct': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown_pct': max_drawdown,
        'win_rate_pct': win_rate,
        'total_trades': len(trades),
        'equity_curve': equity_curve,
        'trades': trades,
    }
    
    if verbose >= 1:
        print(f"\n{'='*50}")
        print(f"Backtest Results")
        print(f"{'='*50}")
        print(f"Initial: ${initial_cash:,.2f}")
        print(f"Final: ${final_value:,.2f}")
        print(f"Total Return: {total_return:+.2f}%")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2f}%")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Total Trades: {len(trades)}")
        print(f"{'='*50}")
    
    return results