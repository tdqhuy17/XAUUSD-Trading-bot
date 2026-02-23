"""
Weekly Training Script for SAC Agent
Usage: python scripts/weekly_train.py

Features:
- Fetches latest data from MT5
- Loads existing model checkpoint
- Fine-tunes on new data
- Saves updated model with replay buffer
- Can be scheduled with Windows Task Scheduler
"""

import sys
import os
import argparse
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_bot.config import Config, SAC_CONFIG
from trading_bot.data_loader import load_mt5_data, load_csv_data, prepare_data
from trading_bot.environment import TradingEnv
from trading_bot.sac_agent import SACAgent
from trading_bot.features import get_feature_columns


def get_latest_model_path():
    """Find the latest checkpoint or model"""
    models_dir = Config.MODELS_DIR
    
    # Check for checkpoints first
    checkpoints = list(models_dir.glob('sac_checkpoint_*.pt'))
    if checkpoints:
        # Sort by modification time, get latest
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return str(checkpoints[0])
    
    # Fall back to main model
    main_model = models_dir / 'sac_gold_agent.pt'
    if main_model.exists():
        return str(main_model)
    
    return None


def fetch_latest_data(days_back: int = 30):
    """Fetch latest data from MT5"""
    print("\n[Fetching Data from MT5]")
    print("  Make sure MT5 is running and logged in!")
    
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        print(f"  Fetching: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        df = load_mt5_data(
            symbol="XAUUSD",
            timeframe="M15",
            start_date=start_date,
            end_date=end_date
        )
        
        print(f"  Loaded: {len(df):,} bars from MT5")
        return df
        
    except Exception as e:
        print(f"\n[ERROR] Failed to load from MT5: {e}")
        print("\nFalling back to CSV data...")
        
        # Fall back to CSV
        csv_path = Config.DATA_DIR / 'XAUUSD_M15.csv'
        if csv_path.exists():
            df = load_csv_data(str(csv_path))
            print(f"  Loaded: {len(df):,} bars from CSV")
            return df
        
        return None


def main():
    parser = argparse.ArgumentParser(description="Weekly fine-tuning for SAC agent")
    parser.add_argument('--days', type=int, default=30, help='Days of data to fetch')
    parser.add_argument('--steps', type=int, default=20000, help='Training steps')
    parser.add_argument('--model', type=str, default=None, help='Model path (auto-detect if not specified)')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    parser.add_argument('--no-buffer', action='store_true', help='Do not load/save replay buffer')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("   SAC Agent Weekly Fine-Tuning")
    print("=" * 60)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Find model
    if args.model:
        model_path = args.model
    elif args.resume:
        model_path = get_latest_model_path()
    else:
        model_path = str(Config.MODELS_DIR / 'sac_gold_agent.pt')
    
    if not model_path or not os.path.exists(model_path):
        print(f"\n[ERROR] No model found!")
        print("  Run train.py first to create a base model")
        return
    
    print(f"\n[Model]")
    print(f"  Path: {model_path}")
    
    # Fetch data
    df = fetch_latest_data(args.days)
    if df is None or len(df) == 0:
        print("\n[ERROR] No data available!")
        return
    
    # Prepare features
    print(f"\n[Preparing Features]")
    df = prepare_data(df, add_indicators=True, add_divergence=True, normalize=True)
    features = get_feature_columns(df)
    print(f"  Features: {len(features)}")
    
    # Create environment
    env = TradingEnv(df, features=features)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"\n[Environment]")
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Data bars: {len(df):,}")
    
    # Load agent
    print(f"\n[Loading Agent]")
    try:
        agent = SACAgent.from_pretrained(model_path)
        print(f"  Loaded successfully!")
        print(f"  Previous training steps: {agent.total_steps:,}")
    except Exception as e:
        print(f"  [ERROR] Failed to load agent: {e}")
        return
    
    # Load replay buffer
    if not args.no_buffer:
        buffer_path = model_path.replace('.pt', '_buffer.pkl').replace('.zip', '_buffer.pkl')
        if os.path.exists(buffer_path):
            print(f"  Loading replay buffer...")
            agent.buffer.load(buffer_path)
            print(f"  Buffer size: {len(agent.buffer):,}")
    
    # Fine-tune
    print(f"\n[Fine-Tuning]")
    print(f"  Steps: {args.steps:,}")
    print("-" * 60)
    
    try:
        metrics = agent.learn(
            total_steps=args.steps,
            env=env,
            checkpoint_dir=str(Config.MODELS_DIR),
            checkpoint_interval=20000
        )
    except KeyboardInterrupt:
        print("\n\n[Training interrupted]")
    except Exception as e:
        print(f"\n[ERROR during training]: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("-" * 60)
    
    # Save updated model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = str(Config.MODELS_DIR / f'sac_gold_agent_{timestamp}.pt')
    
    print(f"\n[Saving Model]")
    agent.save(save_path, save_buffer=not args.no_buffer)
    
    # Also update main model
    main_path = str(Config.MODELS_DIR / 'sac_gold_agent.pt')
    agent.save(main_path, save_buffer=not args.no_buffer)
    print(f"  Updated main model: {main_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("   Weekly Training Complete!")
    print("=" * 60)
    
    if metrics.get('episode_rewards'):
        rewards = metrics['episode_rewards']
        print(f"\n[Summary]")
        print(f"  Episodes: {len(rewards)}")
        print(f"  Mean reward: {sum(rewards)/len(rewards):.2f}")
        print(f"  Total steps: {agent.total_steps:,}")
    
    print(f"\n[Files]")
    print(f"  Checkpoint: {save_path}")
    print(f"  Main model: {main_path}")


if __name__ == "__main__":
    main()