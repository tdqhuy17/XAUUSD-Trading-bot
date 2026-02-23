# 🤖 Multi-Timeframe SAC Trading Bot

A sophisticated reinforcement learning trading system for gold (XAUUSD) using Soft Actor-Critic (SAC) algorithm with multi-timeframe analysis.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

- **Multi-Timeframe Analysis**: Processes M1, M5, M15, M30, and H1 timeframes simultaneously
- **Custom Neural Networks**: Handcrafted GRU/LSTM cells with cross-timeframe attention
- **Soft Actor-Critic (SAC)**: State-of-the-art reinforcement learning with automatic entropy tuning
- **GPU Acceleration**: Supports NVIDIA CUDA and AMD DirectML
- **Technical Indicators**: RSI, MACD, ATR, ADX, EMAs, Break of Structure, Fair Value Gaps
- **Divergence Detection**: Regular and hidden RSI divergences for entry signals
- **Paper Trading**: Built-in simulation mode for risk-free testing
- **MT5 Integration**: Live trading support via MetaTrader5

## 📋 Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Multi-Timeframe Data                          │
│  ┌─────┐  ┌─────┐  ┌──────┐  ┌──────┐  ┌────┐                  │
│  │ M1  │  │ M5  │  │ M15  │  │ M30  │  │ H1 │                  │
│  └──┬──┘  └──┬──┘  └──┬───┘  └──┬───┘  └─┬──┘                  │
│     │        │        │         │        │                      │
│     ▼        ▼        ▼         ▼        ▼                      │
│  ┌──────────────────────────────────────────────┐               │
│  │           Feature Engineering                 │               │
│  │  RSI, MACD, ATR, ADX, EMAs, BOS, FVG, Div.   │               │
│  └──────────────────────────────────────────────┘               │
│                         │                                        │
│                         ▼                                        │
│  ┌──────────────────────────────────────────────┐               │
│  │         Multi-Timeframe Encoder               │               │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐       │               │
│  │  │ GRU/LSTM │  │ GRU/LSTM │  │ GRU/LSTM │ ... │               │
│  │  └────┬────┘  └────┬────┘  └────┬────┘       │               │
│  │       └────────────┼────────────┘            │               │
│  │                    ▼                          │               │
│  │       ┌────────────────────┐                 │               │
│  │       │ Cross-TF Attention │                 │               │
│  │       └────────────────────┘                 │               │
│  └──────────────────────────────────────────────┘               │
│                         │                                        │
│                         ▼                                        │
│  ┌──────────────────────────────────────────────┐               │
│  │              SAC Agent                        │               │
│  │  ┌────────────┐     ┌────────────┐           │               │
│  │  │   Actor    │     │   Critic   │ (Twin Q)  │               │
│  │  │  Network   │     │  Networks  │           │               │
│  │  └────────────┘     └────────────┘           │               │
│  │         │                   │                 │               │
│  │         ▼                   ▼                 │               │
│  │    Action [-1,1]        Q-Values             │               │
│  │   Short ←───→ Long                          │               │
│  └──────────────────────────────────────────────┘               │
│                         │                                        │
│                         ▼                                        │
│  ┌──────────────────────────────────────────────┐               │
│  │          Trading Environment                  │               │
│  │  • Position Management                       │               │
│  │  • Reward Calculation                        │               │
│  │  • Risk Management                           │               │
│  └──────────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Description |
|-----------|-------------|
| **Timeframe Encoders** | Custom GRU/LSTM processes each timeframe independently |
| **Cross-TF Attention** | Attention mechanism to combine multi-timeframe context |
| **SAC Agent** | Soft Actor-Critic with automatic entropy tuning |
| **Trading Environment** | Gymnasium-based environment with realistic simulation |

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Quick Install

```bash
# Clone the repository
git clone https://github.com/yourusername/trading-bot.git
cd trading-bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### GPU Support

#### NVIDIA GPU (CUDA)
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

#### AMD GPU (DirectML)
```bash
pip install torch-directml
```

### Optional Dependencies

```bash
# For MetaTrader5 live trading
pip install MetaTrader5

# For technical analysis library
pip install ta-lib  # Requires separate installation
```

## 🎯 Quick Start

### 1. Prepare Data

Place your XAUUSD M1 data in the `data/` directory:
```
data/
├── DAT_ASCII_XAUUSD_M1_2015.csv
├── DAT_ASCII_XAUUSD_M1_2016.csv
├── ...
└── DAT_ASCII_XAUUSD_M1_2026.csv
```

Expected CSV format (semicolon-separated, no header):
```
20150101 180100;1184.130000;1184.440000;1184.040000;1184.130000;0
```

### 2. Train Model

```bash
# Debug mode (quick testing)
python scripts/train.py --debug

# Production training
python scripts/train.py --steps 500000
```

### 3. Evaluate Model

```bash
# Run backtest
python scripts/backtest.py --model sac_mtf_best --start 2025-01-01 --end 2026-02-28

# Evaluate on test data
python scripts/evaluate.py --model sac_mtf_best
```

### 4. Paper Trading

```bash
python scripts/run_live.py --model sac_mtf_best
```

### 5. Live Trading

```bash
# ⚠️ Use with caution - real money at risk
python scripts/run_live.py --model sac_mtf_best --live
```

## ⚙️ Configuration

### Debug vs Production Mode

Edit `trading_bot/config.py` or use the Config class:

```python
from trading_bot.config import Config

# Enable debug mode (smaller model, faster training)
Config.set_debug(True)

# Production mode (full model)
Config.set_debug(False)
```

### Key Configuration Parameters

| Parameter | Debug | Production | Description |
|-----------|-------|------------|-------------|
| `train_steps` | 500 | 500,000 | Total training steps |
| `batch_size` | 32 | 256 | Batch size for training |
| `hidden_dim` | 128 | 384 | Network hidden dimension |
| `learning_starts` | 50 | 1,000 | Steps before learning starts |
| `buffer_size` | 10,000 | 500,000 | Replay buffer size |

### Multi-Timeframe Settings

```python
MTF_CONFIG = {
    "timeframes": ["M1", "M5", "M15", "M30", "H1"],
    "bars_per_tf": {
        "M1": 60,    # 1 hour of M1
        "M5": 24,    # 2 hours of M5
        "M15": 16,   # 4 hours of M15
        "M30": 8,    # 4 hours of M30
        "H1": 6,     # 6 hours of H1
    },
    "attention_dim": 256,
    "attention_heads": 4,
}
```

### Trading Parameters

```python
ENV_CONFIG = {
    "initial_cash": 10000,
    "transaction_fee": 0.0003,  # 0.03%
    "max_position_pct": 0.5,    # Max 50% of capital
    "max_episode_steps": 1000,
    "max_trades_per_episode": 50,
}
```

## 📁 Project Structure

```
trading-bot/
├── trading_bot/               # Core package
│   ├── __init__.py           # Package exports
│   ├── config.py             # Configuration settings
│   ├── data_loader.py        # Data loading utilities
│   ├── features.py           # Technical indicators
│   ├── environment.py        # Trading environment
│   ├── networks.py           # Neural network architectures
│   ├── sac_agent.py          # SAC implementation
│   ├── replay_buffer.py      # Experience replay
│   ├── trade_executor.py     # Trade execution
│   ├── model.py              # Model utilities
│   └── directml_optim.py     # DirectML optimizer
│
├── scripts/                   # Executable scripts
│   ├── train.py              # Training script
│   ├── backtest.py           # Backtesting script
│   ├── evaluate.py           # Evaluation script
│   ├── run_live.py           # Live trading script
│   ├── weekly_train.py       # Continuous learning
│   └── visualize_results.py  # Results visualization
│
├── examples/                  # Example scripts
│   ├── example_train.py      # Basic training
│   ├── example_backtest.py   # Basic backtest
│   └── example_live.py       # Basic paper trading
│
├── data/                      # Data directory
├── models/                    # Saved models
├── logs/                      # Training logs
├── ref/                       # Reference materials
│
├── requirements.txt          # Dependencies
├── setup.py                  # Package setup
├── README.md                 # This file
├── CONTRIBUTING.md           # Contribution guide
└── LICENSE                   # MIT License
```

## 📖 API Reference

### Core Classes

#### SACAgentMTF

```python
from trading_bot.sac_agent import SACAgentMTF

# Create agent
agent = SACAgentMTF(mtf_config, action_dim=1)

# Train
agent.learn(total_steps=100000, env=env)

# Select action
action = agent.select_action(mtf_obs, portfolio)

# Save/Load
agent.save("models/my_agent.pt")
agent.load("models/my_agent.pt")

# Load pretrained
agent = SACAgentMTF.from_pretrained("models/my_agent.pt")
```

#### TradingEnvMTF

```python
from trading_bot.environment import TradingEnvMTF

# Create environment
env = TradingEnvMTF(mtf_data, feature_cols)

# Use like any Gymnasium environment
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

#### Feature Engineering

```python
from trading_bot.features import (
    add_core_features,
    add_mtf_features,
    get_feature_columns,
    normalize_features,
)

# Add features to single timeframe
df = add_core_features(df, include_divergence=True)

# Add features to multi-timeframe data
mtf_data = add_mtf_features(mtf_data)

# Normalize features
df, scaler = normalize_features(df)

# Get feature column names
features = get_feature_columns(include_divergence=True)
```

#### Data Loading

```python
from trading_bot.data_loader import (
    load_mtf_data,
    merge_m1_data,
    create_mtf_data,
)

# Load and split data
train_mtf, test_mtf = load_mtf_data(
    train_start="2015-01-01",
    train_end="2024-12-31",
    test_start="2025-01-01",
    test_end="2026-02-28",
)
```

### Neural Network Components

```python
from trading_bot.networks import (
    CustomGRUCell,
    CustomLSTMCell,
    MTFActorNetwork,
    MTFCriticNetwork,
    build_mtf_networks,
)

# Build networks
actor, critic = build_mtf_networks(mtf_config)

# Custom RNN cells
gru_cell = CustomGRUCell(input_dim=64, hidden_dim=128)
lstm_cell = CustomLSTMCell(input_dim=64, hidden_dim=128)
```

## 📊 Examples

### Basic Training Example

```python
from trading_bot.config import Config, MTF_CONFIG, SAC_CONFIG
from trading_bot.data_loader import load_mtf_data
from trading_bot.features import add_mtf_features, normalize_features
from trading_bot.environment import TradingEnvMTF
from trading_bot.sac_agent import SACAgentMTF

# Load data
train_mtf, test_mtf = load_mtf_data()

# Add features
train_mtf = add_mtf_features(train_mtf)
for tf in train_mtf:
    train_mtf[tf], _ = normalize_features(train_mtf[tf])

# Create environment
env = TradingEnvMTF(train_mtf)

# Create and train agent
agent = SACAgentMTF(MTF_CONFIG)
agent.learn(total_steps=100000, env=env)
```

### Backtest Example

```python
from trading_bot.model import load_model, backtest

# Load trained model
agent = load_model("models/sac_mtf_best")

# Run backtest
results = backtest(agent, test_df, verbose=1)

print(f"Total Return: {results['total_return_pct']:.2f}%")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}")
print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
print(f"Win Rate: {results['win_rate_pct']:.1f}%")
```

### Paper Trading Example

```python
from trading_bot.trade_executor import PaperTrader

trader = PaperTrader(initial_cash=10000)

# Execute trades
result = trader.execute("BUY", price=2000.0)
result = trader.execute("SELL", price=2010.0)

# Get summary
summary = trader.get_summary()
print(f"PnL: ${summary['total_pnl']:.2f}")
```

## 🧠 Technical Details

### Reward Structure

The environment uses a multi-component reward:

1. **Realized PnL**: Reward for closing profitable positions
2. **Unrealized PnL**: Small reward for holding winning positions
3. **Divergence Bonus**: Extra reward for trading with divergence signals
4. **Time Decay**: Small penalty for holding too long
5. **Drawdown Penalty**: Penalty for large drawdowns

### Action Space

- Continuous action in `[-1, 1]`
- `[-1, -0.1)`: Short position
- `[-0.1, 0.1]`: Close position / Hold
- `(0.1, 1]`: Long position

### Observation Space

Multi-timeframe observations:
- M1: 60 bars × 32 features
- M5: 24 bars × 32 features
- M15: 16 bars × 32 features
- M30: 8 bars × 32 features
- H1: 6 bars × 32 features
- Portfolio: 6 values (cash, position, unrealized PnL, divergences)

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone and install dev dependencies
git clone https://github.com/yourusername/trading-bot.git
cd trading-bot
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black trading_bot/ scripts/
```

## ⚠️ Disclaimer

**This software is for educational purposes only.** 

Trading financial instruments involves substantial risk of loss. Past performance is not indicative of future results. The authors and contributors are not responsible for any financial losses incurred through the use of this software.

- Always test thoroughly with paper trading before live trading
- Never trade with money you cannot afford to lose
- Understand the risks involved in algorithmic trading

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Soft Actor-Critic algorithm from [Haarnoja et al.](https://arxiv.org/abs/1801.01290)
- PyTorch team for the excellent deep learning framework
- OpenAI Gymnasium for the RL environment interface

## 📧 Contact

For questions and support, please open an issue on GitHub.

---

**Star ⭐ this repo if you find it useful!**