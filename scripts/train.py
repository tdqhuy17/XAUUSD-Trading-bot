"""
Training Script for Multi-Timeframe SAC Agent
Usage: python scripts/train.py

Features:
- Loads M1 data (2015-2024 for training, 2025-2026 for testing)
- Creates multi-timeframe features (M1, M5, M15, M30, H1)
- Trains hierarchical MTF SAC agent
"""

import sys
import os
import argparse
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from trading_bot.config import (
    Config, SAC_CONFIG, MTF_CONFIG, MTF_FEATURE_CONFIG, 
    DATA_CONFIG, ENV_CONFIG, DEBUG_MODE
)
from trading_bot.data_loader import load_mtf_data, print_data_summary
from trading_bot.features import add_mtf_features, get_feature_columns, normalize_features
from trading_bot.environment import TradingEnvMTF
from trading_bot.sac_agent import SACAgentMTF


def prepare_mtf_config(mtf_data: dict) -> dict:
    """
    Prepare the MTF config with actual feature dimensions from data.
    
    Args:
        mtf_data: Dict mapping timeframe to DataFrame with features
        
    Returns:
        Complete MTF config for agent initialization
    """
    timeframes = MTF_CONFIG['timeframes']
    bars_per_tf = MTF_CONFIG['bars_per_tf']
    encoder_dims = MTF_CONFIG['encoder_dims']
    
    # Get feature dimensions from data
    feature_dims = {}
    feature_cols = {}
    
    for tf in timeframes:
        if tf in mtf_data:
            df = mtf_data[tf]
            cols = get_feature_columns(include_divergence=(tf == 'M1'))
            # Only use columns that exist
            existing_cols = [c for c in cols if c in df.columns]
            feature_cols[tf] = existing_cols
            feature_dims[tf] = len(existing_cols)
        else:
            feature_dims[tf] = 30
            feature_cols[tf] = []
    
    # Build complete config
    config = {
        'timeframes': timeframes,
        'bars_per_tf': bars_per_tf,
        'encoder_dims': encoder_dims,
        'feature_dims': feature_dims,
        'attention_dim': MTF_CONFIG.get('attention_dim', 256),
        'attention_heads': MTF_CONFIG.get('attention_heads', 4),
        
        # Network architecture
        'hidden_dim': SAC_CONFIG.get('hidden_dim', 384),
        'n_layers': SAC_CONFIG.get('n_layers', 3),
        'dropout': SAC_CONFIG.get('dropout', 0.1),
        'rnn_type': SAC_CONFIG.get('rnn_type', 'gru'),
        
        # Training parameters
        'learning_rate': SAC_CONFIG.get('learning_rate', 3e-4),
        'actor_lr': SAC_CONFIG.get('actor_lr', 3e-4),
        'critic_lr': SAC_CONFIG.get('critic_lr', 1e-4),
        'alpha_lr': SAC_CONFIG.get('alpha_lr', 3e-4),
        'gamma': SAC_CONFIG.get('gamma', 0.995),
        'tau': SAC_CONFIG.get('tau', 0.005),
        'batch_size': SAC_CONFIG.get('batch_size', 128),
        'buffer_size': SAC_CONFIG.get('buffer_size', 500_000),
        'learning_starts': SAC_CONFIG.get('learning_starts', 2000),
        'gradient_steps': SAC_CONFIG.get('gradient_steps', 1),
    }
    
    return config, feature_cols


def main():
    parser = argparse.ArgumentParser(description="Train Multi-Timeframe SAC Agent")
    parser.add_argument('--debug', action='store_true', help='Use debug config (faster)')
    parser.add_argument('--steps', type=int, default=None, help='Override training steps')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size')
    parser.add_argument('--learning_starts', type=int, default=None, help='Override learning starts')
    args = parser.parse_args()
    
    # Set debug mode
    if args.debug:
        Config.set_debug(True)
        print("\n[DEBUG MODE] Using smaller config for quick testing\n")
    
    print("=" * 70)
    print("  MULTI-TIMEFRAME SAC AGENT TRAINING")
    print("=" * 70)
    print(f"  Debug Mode    : {DEBUG_MODE}")
    print(f"  Timeframes    : {MTF_CONFIG['timeframes']}")
    print(f"  Lookback      : {MTF_CONFIG['lookback_hours']} hours")
    print(f"  Train Period  : {DATA_CONFIG['train_start']} to {DATA_CONFIG['train_end']}")
    print(f"  Test Period   : {DATA_CONFIG['test_start']} to {DATA_CONFIG['test_end']}")
    print("=" * 70)
    
    # ================================================================
    # STEP 1: Load Data
    # ================================================================
    print("\n[STEP 1] Loading M1 data and creating multi-timeframe data...")
    
    train_mtf, test_mtf = load_mtf_data()
    print_data_summary(train_mtf, test_mtf)
    
    # ================================================================
    # STEP 2: Add Features
    # ================================================================
    print("\n[STEP 2] Adding technical features to all timeframes...")
    
    train_mtf = add_mtf_features(train_mtf, MTF_FEATURE_CONFIG)
    test_mtf = add_mtf_features(test_mtf, MTF_FEATURE_CONFIG)
    
    # Normalize features
    print("\n  Normalizing features...")
    for tf in train_mtf:
        train_mtf[tf], _ = normalize_features(train_mtf[tf])
        test_mtf[tf], _ = normalize_features(test_mtf[tf])
    
    # Drop NaN rows
    print("  Dropping NaN rows...")
    for tf in train_mtf:
        train_mtf[tf] = train_mtf[tf].dropna()
        test_mtf[tf] = test_mtf[tf].dropna()
    
    print("\n  Feature summary:")
    for tf in train_mtf:
        n_features = len([c for c in train_mtf[tf].columns if c not in ['open', 'high', 'low', 'close', 'volume']])
        print(f"    {tf}: {len(train_mtf[tf])} bars, {n_features} features")
    
    # ================================================================
    # STEP 3: Prepare Config
    # ================================================================
    print("\n[STEP 3] Preparing MTF configuration...")
    
    mtf_config, feature_cols = prepare_mtf_config(train_mtf)
    
    print(f"\n  Feature dimensions:")
    for tf, dim in mtf_config['feature_dims'].items():
        print(f"    {tf}: {dim} features")
    
    # Override with command line args
    if args.steps:
        mtf_config['train_steps'] = args.steps
    else:
        mtf_config['train_steps'] = SAC_CONFIG.get('train_steps', 200_000)
    
    if args.batch_size:
        mtf_config['batch_size'] = args.batch_size
    
    if args.learning_starts:
        mtf_config['learning_starts'] = args.learning_starts
    
    print(f"\n  Training config:")
    print(f"    Steps          : {mtf_config['train_steps']:,}")
    print(f"    Batch size     : {mtf_config['batch_size']}")
    print(f"    Learning starts: {mtf_config['learning_starts']:,}")
    print(f"    Hidden dim     : {mtf_config['hidden_dim']}")
    print(f"    RNN type       : {mtf_config['rnn_type']}")
    
    # ================================================================
    # STEP 4: Create Environment
    # ================================================================
    print("\n[STEP 4] Creating MTF trading environment...")
    
    env = TradingEnvMTF(train_mtf, feature_cols)
    
    print(f"  Observation space:")
    for tf, space in env.observation_space.spaces.items():
        if tf != 'portfolio':
            print(f"    {tf}: {space.shape}")
    print(f"  Portfolio: {env.observation_space['portfolio'].shape}")
    print(f"  Action space: {env.action_space.shape}")
    
    # ================================================================
    # STEP 5: Create Agent
    # ================================================================
    print("\n[STEP 5] Creating MTF SAC agent...")
    
    agent = SACAgentMTF(mtf_config, action_dim=1)
    
    # Count parameters
    actor_params = sum(p.numel() for p in agent.actor.parameters())
    critic_params = sum(p.numel() for p in agent.critic.parameters())
    print(f"  Actor parameters  : {actor_params:,}")
    print(f"  Critic parameters : {critic_params:,}")
    
    # ================================================================
    # STEP 6: Train
    # ================================================================
    print("\n[STEP 6] Starting training...")
    print("=" * 70)
    
    start_time = time.time()
    
    metrics = agent.learn(
        total_steps=mtf_config['train_steps'],
        env=env,
        checkpoint_dir=str(Config.MODELS_DIR),
        checkpoint_interval=SAC_CONFIG.get('checkpoint_interval', 10_000),
        best_model_path=str(Config.MODELS_DIR / 'sac_mtf_best.pt'),
    )
    
    elapsed = time.time() - start_time
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Total time     : {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"  Best reward    : {metrics.get('best_episode_reward', 0):.4f}")
    print(f"  Win rate       : {metrics.get('win_rate', 0)*100:.1f}%")
    print(f"  Model saved to : {Config.MODELS_DIR / 'sac_mtf_best.pt'}")
    print("=" * 70)
    
    # ================================================================
    # STEP 7: Detailed Test Evaluation
    # ================================================================
    print("\n[STEP 7] Detailed evaluation on test data...")
    
    test_env = TradingEnvMTF(test_mtf, feature_cols)
    
    # Run evaluation episodes
    eval_results = []
    all_trades = []
    
    for ep in range(5):
        obs, _ = test_env.reset()
        done = False
        ep_reward = 0
        ep_trades = 0
        ep_start_assets = test_env.total_assets
        
        while not done:
            mtf_obs = obs
            portfolio = obs.get('portfolio', np.zeros(6))
            action = agent.select_action_deterministic(mtf_obs, portfolio)
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_trades = info.get('trade_count', 0)
        
        final_assets = test_env.total_assets
        pnl = final_assets - test_env.initial_cash
        pnl_pct = (pnl / test_env.initial_cash) * 100
        
        result = {
            'episode': ep + 1,
            'reward': ep_reward,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'trades': ep_trades,
            'final_assets': final_assets,
        }
        eval_results.append(result)
        
        print(f"  Episode {ep+1}: reward={ep_reward:+.2f}, PnL=${pnl:+.2f} ({pnl_pct:+.2f}%), trades={ep_trades}")
    
    # Calculate summary
    avg_reward = np.mean([r['reward'] for r in eval_results])
    total_pnl = sum(r['pnl'] for r in eval_results)
    avg_pnl_pct = np.mean([r['pnl_pct'] for r in eval_results])
    total_trades = sum(r['trades'] for r in eval_results)
    winning_eps = sum(1 for r in eval_results if r['pnl'] > 0)
    
    print("\n" + "-" * 50)
    print("  EVALUATION SUMMARY")
    print("-" * 50)
    print(f"  Avg Reward     : {avg_reward:+.4f}")
    print(f"  Total PnL      : ${total_pnl:+.2f}")
    print(f"  Avg PnL %      : {avg_pnl_pct:+.2f}%")
    print(f"  Total Trades   : {total_trades}")
    print(f"  Winning Episodes: {winning_eps}/5 ({winning_eps/5*100:.0f}%)")
    print("-" * 50)
    
    # Save to CSV
    eval_df = pd.DataFrame(eval_results)
    eval_df['timestamp'] = pd.Timestamp.now()
    eval_df['train_steps'] = mtf_config['train_steps']
    eval_df['model'] = 'sac_mtf_best.pt'
    
    eval_file = Config.LOGS_DIR / 'evaluation_results.csv'
    
    # Append to existing file or create new
    if eval_file.exists():
        existing = pd.read_csv(eval_file)
        eval_df = pd.concat([existing, eval_df], ignore_index=True)
    
    eval_df.to_csv(eval_file, index=False)
    print(f"\n  Results saved to: {eval_file}")
    
    return metrics


if __name__ == "__main__":
    main()