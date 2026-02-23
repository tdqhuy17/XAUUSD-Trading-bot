"""
Configuration Settings for Trading Bot
- Multi-Timeframe Support
- Custom GRU/LSTM Architecture
- CUDA GPU acceleration
"""

import os
from pathlib import Path

# ============== PATHS ==============
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ============== DEBUG MODE ==============
# Set to False for production training
DEBUG_MODE = True

# ============== MT5 CONFIGURATION ==============
MT5_CONFIG = {
    "symbol": "XAUUSD",
    "timeframe": "M1",  # Base timeframe for execution
    "initial_cash": 10000,
    "transaction_fee": 0.0003,
}

# ============== MULTI-TIMEFRAME CONFIG ==============
MTF_CONFIG = {
    # Timeframes to use (base is M1)
    "timeframes": ["M1", "M5", "M15", "M30", "H1"],
    
    # Lookback window: 6 hours at each timeframe
    "lookback_hours": 6,
    
    # Bars per timeframe (reduced for speed)
    "bars_per_tf": {
        "M1": 60,     # 1 hour of M1 (reduced from 360)
        "M5": 24,     # 2 hours of M5
        "M15": 16,    # 4 hours of M15
        "M30": 8,     # 4 hours of M30
        "H1": 6,      # 6 hours of H1
    },
    
    # Resampling map (minutes)
    "tf_minutes": {
        "M1": 1,
        "M5": 5,
        "M15": 15,
        "M30": 30,
        "H1": 60,
    },
    
    # Encoder dimensions per timeframe (reduced for speed)
    "encoder_dims": {
        "M1": {"hidden": 64, "layers": 1},
        "M5": {"hidden": 32, "layers": 1},
        "M15": {"hidden": 32, "layers": 1},
        "M30": {"hidden": 16, "layers": 1},
        "H1": {"hidden": 16, "layers": 1},
    },
    
    # Cross-timeframe attention
    "attention_dim": 256,
    "attention_heads": 4,
}

# ============== FEATURE CONFIG (PRUNED) ==============
# Core features computed at each timeframe
MTF_FEATURE_CONFIG = {
    # Momentum
    "rsi_period": 14,
    
    # Trend
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "ema_periods": [20, 50],  # Simplified from 20, 50, 100, 200, 256
    
    # Volatility
    "atr_period": 14,
    
    # Trend strength
    "adx_period": 14,
    
    # Structure
    "bos_window": 10,        # Break of Structure lookback
    "swing_window": 20,      # Swing high/low detection
    
    # Divergence (only on M1 for entry signals)
    "pivot_left": 3,
    "pivot_right": 3,
    "rsi_divergence_lookback": 5,
}

# Legacy feature config (for backward compatibility)
FEATURE_CONFIG = {
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "adx_period": 14,
    "ema_short": 20,
    "ema_long": 50,
    "atr_period": 14,
    "rsi_divergence_lookback": 5,
    "pivot_left": 3,
    "pivot_right": 3,
}

# ============== RL ENVIRONMENT CONFIG ==============
ENV_CONFIG = {
    "lookback_window": 20,         # Legacy - not used in MTF
    "initial_cash": 10000,
    "transaction_fee": 0.0003,
    "max_position_pct": 0.5,
    "max_episode_steps": 1000,     # ~16 hours of M1 trading
    "max_trades_per_episode": 50,
}

# ============== SAC MODEL CONFIG ==============
# Debug config (small values for quick testing)
SAC_CONFIG_DEBUG = {
    "learning_rate": 3e-4,
    "actor_lr": 3e-4,
    "critic_lr": 1e-4,
    "alpha_lr": 3e-4,
    "batch_size": 32,
    "gamma": 0.99,
    "tau": 0.005,
    "train_steps": 500,
    "learning_starts": 50,
    "eval_freq": 100,
    
    # Network architecture (debug - smaller)
    "hidden_dim": 128,
    "attention_heads": 2,
    "gru_hidden": 64,
    "n_layers": 2,
    "dropout": 0.0,
    
    # Replay buffer
    "buffer_size": 10_000,
    "buffer_type": "standard",
    
    # MTF specific
    "use_mtf": True,
    "rnn_type": "gru",  # "gru" or "lstm"
    
    # Checkpoint interval
    "checkpoint_interval": 10_000,
}

# Production config (for real training)
SAC_CONFIG_PROD = {
    "learning_rate": 3e-4,
    "actor_lr": 3e-4,
    "critic_lr": 1e-4,
    "alpha_lr": 3e-4,
    "batch_size": 256,  # Increased for better GPU utilization
    "gamma": 0.995,
    "tau": 0.005,
    "train_steps": 500_000,  # Extended for convergence
    "learning_starts": 1000,
    "eval_freq": 5000,
    
    # Network architecture
    "hidden_dim": 384,
    "attention_heads": 4,
    "gru_hidden": 128,
    "n_layers": 3,
    "dropout": 0.1,
    
    # Replay buffer
    "buffer_size": 500_000,
    "buffer_type": "standard",
    
    # MTF specific
    "use_mtf": True,
    "rnn_type": "gru",  # "gru" or "lstm"
    
    # Gradient steps (multiple updates per env step for efficiency)
    "gradient_steps": 4,
    
    # Checkpoint interval
    "checkpoint_interval": 10_000,
}

# Select config based on debug mode
SAC_CONFIG = SAC_CONFIG_DEBUG if DEBUG_MODE else SAC_CONFIG_PROD

# ============== TRADING CONFIG ==============
TRADE_CONFIG = {
    "lot_size": 0.01,
    "magic_number": 234567,
    "deviation": 20,
    "max_positions": 2,
    "use_trailing_stop": False,
    "trailing_stop_pct": 0.02,
    "tp_pct": 0.01,
    "sl_pct": 0.005,
}

# ============== DIVERGENCE STRATEGY CONFIG ==============
DIVERGENCE_CONFIG = {
    "rsi_period": 9,
    "lookback_left": 1,
    "lookback_right": 3,
    "range_upper": 60,
    "range_lower": 5,
    "tp_rsi_level": 80,
    "enable_regular_bullish": True,
    "enable_hidden_bullish": True,
    "enable_regular_bearish": True,
    "enable_hidden_bearish": False,
    "sl_type": "NONE",
    "stop_loss_pct": 5.0,
    "atr_length": 14,
    "atr_multiplier": 3.5,
}

# ============== CONTINUOUS LEARNING ==============
CONTINUOUS_LEARNING_CONFIG = {
    "enabled": True,
    "retrain_frequency": "weekly",
    "new_data_window": 30,
    "fine_tune_steps": 10000,
    "save_replay_buffer": True,
}

# ============== DATA CONFIG ==============
DATA_CONFIG = {
    "csv_date_format": "%Y%m%d %H%M%S",  # Format: 20150101 180100
    "train_start": "2015-01-01",
    "train_end": "2024-12-31",
    "test_start": "2025-01-01",
    "test_end": "2026-02-28",
    
    # M1 data files pattern
    "m1_pattern": "DAT_ASCII_XAUUSD_M1_*.csv",
}

# ============== COLUMNS ==============
COLUMNS = {
    "time": "time",
    "open": "open",
    "high": "high", 
    "low": "low",
    "close": "close",
    "volume": "volume",
}


class Config:
    MT5 = MT5_CONFIG
    ENV = ENV_CONFIG
    SAC = SAC_CONFIG
    FEATURES = FEATURE_CONFIG
    MTF = MTF_CONFIG
    MTF_FEATURES = MTF_FEATURE_CONFIG
    TRADE = TRADE_CONFIG
    DIVERGENCE = DIVERGENCE_CONFIG
    CONTINUOUS_LEARNING = CONTINUOUS_LEARNING_CONFIG
    DATA = DATA_CONFIG
    
    BASE_DIR = BASE_DIR
    DATA_DIR = DATA_DIR
    MODELS_DIR = MODELS_DIR
    LOGS_DIR = LOGS_DIR
    
    DEBUG = DEBUG_MODE
    
    @classmethod
    def get_model_path(cls, name="sac_gold_agent"):
        return cls.MODELS_DIR / name
    
    @classmethod
    def get_data_path(cls, filename):
        return cls.DATA_DIR / filename
    
    @classmethod
    def set_debug(cls, debug: bool):
        """Switch between debug and production mode"""
        global DEBUG_MODE, SAC_CONFIG
        DEBUG_MODE = debug
        SAC_CONFIG = SAC_CONFIG_DEBUG if debug else SAC_CONFIG_PROD
        cls.DEBUG = debug
        cls.SAC = SAC_CONFIG
        print(f"[Config] Debug mode: {'ON' if debug else 'OFF'}")
        if debug:
            print(f"  Train steps: {SAC_CONFIG['train_steps']}")
            print(f"  Batch size: {SAC_CONFIG['batch_size']}")
        else:
            print(f"  Train steps: {SAC_CONFIG['train_steps']}")
            print(f"  Batch size: {SAC_CONFIG['batch_size']}")


def get_device_info():
    """Get information about available compute devices."""
    devices = []
    
    # Check for CUDA (NVIDIA GPU)
    try:
        import torch
        if torch.cuda.is_available():
            devices.append({
                'name': f'CUDA ({torch.cuda.get_device_name(0)})',
                'device': 'cuda',
                'type': 'gpu'
            })
    except ImportError:
        pass
    
    # CPU fallback
    devices.append({
        'name': 'CPU',
        'device': 'cpu',
        'type': 'cpu'
    })
    
    return devices