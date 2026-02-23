#!/usr/bin/env python
"""
Example: Basic Training Script

This example demonstrates how to train a Multi-Timeframe SAC agent
from scratch using historical XAUUSD data.

Usage:
    python examples/example_train.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from trading_bot.config import Config, SAC_CONFIG, MTF_CONFIG, MTF_FEATURE_CONFIG
from trading_bot.data_loader import load_mtf_data, print_data_summary
from trading_bot.features import add_mtf_features, get_feature_columns, normalize_features
from trading_bot.environment import TradingEnvMTF
from trading_bot.sac_agent import SACAgentMTF


def prepare_mtf_config(mtf_data: dict) -> tuple:
    """Prepare MTF config with actual feature dimensions from data."""
    timeframes = MTF_CONFIG['timeframes']
    bars_per_tf = MTF_CONFIG['bars_per_tf']
    encoder_dims = MTF_CONFIG['encoder_dims']
    
    feature_dims = {}
    feature_cols = {}
    
    for tf in timeframes:
        if tf in mtf_data:
            df = mtf_data[tf]
            cols = get_feature_columns(include_divergence=(tf == 'M1'))
            existing_cols = [c for c in cols if c in df.columns]
            feature_cols[tf] = existing_cols
            feature_dims[tf] = len(existing_cols)
        else:
            feature_dims[tf] = 30
            feature_cols[tf] = []
    
    config = {
        'timeframes': timeframes,
        'bars_per_tf': bars_per_tf,
        'encoder_dims': encoder_dims,
        'feature_dims': feature_dims,
        'attention_dim': MTF_CONFIG.get('attention_dim', 256),
        'attention_heads': MTF_CONFIG.get('attention_heads', 4),
        'hidden_dim': SAC_CONFIG.get('hidden_dim', 384),
        'n_layers': SAC_CONFIG.get('n_layers', 3),
        'dropout': SAC_CONFIG.get('dropout', 0.1),
        'rnn_type': SAC_CONFIG.get('rnn_type', 'gru'),
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
        'train_steps': 10000,  # Example: 10k steps for demo
    }
    
    return config, feature_cols


def main():
    """Main training example."""
    print("=" * 60)
    print("  Example: Training Multi-Timeframe SAC Agent")
    print("=" * 60)
    
    # Use debug mode for faster training in this example
    Config.set_debug(True)
    print("\n[Config] Debug mode enabled for faster training")
    
    # ==========================================
    # Step 1: Load Data
    # ==========================================
    print("\n[Step 1] Loading multi-timeframe data...")
    
    try:
        train_mtf, test_mtf = load_mtf_data()
        print_data_summary(train_mtf, test_mtf)
    except FileNotFoundError as e:
        print(f"\n[Error] {e}")
        print("\n[Info] Please place XAUUSD M1 data files in the data/ directory.")
        print("       Expected format: DAT_ASCII_XAUUSD_M1_YYYY.csv")
        return
    
    # ==========================================
    # Step 2: Add Features
    # ==========================================
    print("\n[Step 2] Adding technical features...")
    
    train_mtf = add_mtf_features(train_mtf, MTF_FEATURE_CONFIG)
    test_mtf = add_mtf_features(test_mtf, MTF_FEATURE_CONFIG)
    
    # Normalize features
    print("  Normalizing features...")
    for tf in train_mtf:
        train_mtf[tf], _ = normalize_features(train_mtf[tf])
        test_mtf[tf], _ = normalize_features(test_mtf[tf])
    
    # Drop NaN rows
    for tf in train_mtf:
        train_mtf[tf] = train_mtf[tf].dropna()
        test_mtf[tf] = test_mtf[tf].dropna()
    
    print(f"  Feature count per timeframe:")
    for tf in train_mtf:
        n_features = len([c for c in train_mtf[tf].columns 
                         if c not in ['open', 'high', 'low', 'close', 'volume']])
        print(f"    {tf}: {n_features} features, {len(train_mtf[tf])} bars")
    
    # ==========================================
    # Step 3: Create Environment & Agent
    # ==========================================
    print("\n[Step 3] Creating environment and agent...")
    
    mtf_config, feature_cols = prepare_mtf_config(train_mtf)
    env = TradingEnvMTF(train_mtf, feature_cols)
    
    print(f"  Observation space:")
    for tf, space in env.observation_space.spaces.items():
        if tf != 'portfolio':
            print(f"    {tf}: {space.shape}")
    print(f"  Portfolio: {env.observation_space['portfolio'].shape}")
    
    agent = SACAgentMTF(mtf_config, action_dim=1)
    
    # ==========================================
    # Step 4: Train
    # ==========================================
    print("\n[Step 4] Training agent...")
    print(f"  Total steps: {mtf_config['train_steps']:,}")
    print("-" * 60)
    
    metrics = agent.learn(
        total_steps=mtf_config['train_steps'],
        env=env,
        checkpoint_dir=str(Config.MODELS_DIR),
        best_model_path=str(Config.MODELS_DIR / 'example_agent.pt'),
    )
    
    # ==========================================
    # Step 5: Summary
    # ==========================================
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"  Best episode reward: {metrics.get('best_episode_reward', 0):.4f}")
    print(f"  Win rate: {metrics.get('win_rate', 0) * 100:.1f}%")
    print(f"  Model saved to: {Config.MODELS_DIR / 'example_agent.pt'}")
    print("=" * 60)


if __name__ == "__main__":
    main()