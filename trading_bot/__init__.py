"""
Trading Bot Package
Multi-Timeframe SAC Reinforcement Learning trading system
"""

from .config import Config, SAC_CONFIG, ENV_CONFIG, MTF_CONFIG, MTF_FEATURE_CONFIG
from .features import (
    add_technical_indicators, 
    add_divergence_features, 
    add_mtf_features,
    add_core_features,
    normalize_features,
    get_feature_columns
)
from .data_loader import (
    load_mt5_data, 
    load_csv_data, 
    load_mtf_data,
    merge_m1_data,
    resample_ohlcv,
    create_mtf_data,
    print_data_summary
)
from .environment import TradingEnv, TradingEnvMTF
from .model import train_sac, load_model, predict, evaluate_model, backtest
from .trade_executor import PaperTrader, MT5Trader
from .sac_agent import SACAgent, SACAgentMTF
from .networks import (
    ActorNetwork, CriticNetwork, 
    MTFActorNetwork, MTFCriticNetwork,
    build_networks, build_mtf_networks,
    CustomGRUCell, CustomLSTMCell
)
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

__version__ = "3.0.0"
__all__ = [
    # Config
    "Config",
    "SAC_CONFIG",
    "ENV_CONFIG",
    "MTF_CONFIG",
    "MTF_FEATURE_CONFIG",
    # Features
    "add_technical_indicators", 
    "add_divergence_features",
    "add_mtf_features",
    "add_core_features",
    "normalize_features",
    "get_feature_columns",
    # Data
    "load_mt5_data",
    "load_csv_data",
    "load_mtf_data",
    "merge_m1_data",
    "resample_ohlcv",
    "create_mtf_data",
    "print_data_summary",
    # Environment
    "TradingEnv",
    "TradingEnvMTF",
    # Model
    "train_sac",
    "load_model", 
    "predict",
    "evaluate_model",
    "backtest",
    # Trading
    "PaperTrader",
    "MT5Trader",
    # SAC Agent
    "SACAgent",
    "SACAgentMTF",
    # Networks
    "ActorNetwork",
    "CriticNetwork",
    "MTFActorNetwork",
    "MTFCriticNetwork",
    "build_networks",
    "build_mtf_networks",
    "CustomGRUCell",
    "CustomLSTMCell",
    # Replay Buffer
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
]