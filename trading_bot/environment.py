"""
Trading Environment for Multi-Timeframe RL
- Compatible with gymnasium API
- Supports M1 execution with HTF context
- Correct mark-to-market P&L for both long and short positions
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple

from .config import ENV_CONFIG, MTF_CONFIG, Config


class TradingEnvMTF(gym.Env):
    """
    Multi-Timeframe Trading Environment for SAC agent

    Observation space:
    - M1:  bars_per_tf['M1'] × features (price action + indicators)
    - M5:  bars_per_tf['M5'] × features
    - M15: bars_per_tf['M15'] × features
    - M30: bars_per_tf['M30'] × features
    - H1:  bars_per_tf['H1'] × features
    - Portfolio state (cash_ratio, position_value_ratio, abs_position_ratio, div_bull, div_bear)

    Action space:
    - Continuous in [-1, 1]
    - Negative → Short, Near-zero → Hold, Positive → Long

    The environment executes trades at M1 granularity but provides
    context from higher timeframes.
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, mtf_data: Dict[str, pd.DataFrame], 
                 feature_cols: Dict[str, List[str]] = None):
        """
        Initialize multi-timeframe environment.

        Args:
            mtf_data: Dict mapping timeframe to DataFrame with features
            feature_cols: Dict mapping timeframe to list of feature columns
        """
        super().__init__()

        self.mtf_data = mtf_data
        self.timeframes = MTF_CONFIG.get('timeframes', ['M1', 'M5', 'M15', 'M30', 'H1'])
        self.bars_per_tf = MTF_CONFIG.get('bars_per_tf', {
            'M1': 360, 'M5': 72, 'M15': 24, 'M30': 12, 'H1': 6
        })

        # Feature columns per timeframe
        if feature_cols is None:
            feature_cols = {tf: self._get_default_features(tf) for tf in self.timeframes}
        self.feature_cols = feature_cols

        # Calculate feature dimensions
        self.feature_dims = {}
        for tf in self.timeframes:
            if tf in mtf_data and len(mtf_data[tf].columns) > 0:
                self.feature_dims[tf] = len(feature_cols.get(tf, []))
            else:
                self.feature_dims[tf] = 30  # Default

        # M1 data reference for execution (needed before precompute)
        self.m1_df = mtf_data.get('M1')
        if self.m1_df is None:
            raise ValueError("M1 data is required for trading environment")
        
        # ===== OPTIMIZATION: Pre-convert DataFrames to numpy arrays =====
        self._precompute_arrays()

        # Environment parameters
        self.initial_cash = ENV_CONFIG.get('initial_cash', 10_000)
        self.transaction_fee = ENV_CONFIG.get('transaction_fee', 0.0003)
        self.max_position_pct = ENV_CONFIG.get('max_position_pct', 0.5)
        self.max_episode_steps = ENV_CONFIG.get('max_episode_steps', 1000)
        self.max_position_value = self.initial_cash * 1.5
        self.max_trades_per_episode = ENV_CONFIG.get('max_trades_per_episode', 50)

        # Reward clipping
        self.reward_clip = 1.0

        # Warmup bars for indicators
        self.warmup_bars = 100

        # Action space
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32
        )

        # Observation space (dict for MTF)
        self.observation_space = spaces.Dict({
            tf: spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.bars_per_tf[tf], self.feature_dims[tf]),
                dtype=np.float32
            )
            for tf in self.timeframes
        })
        # Add portfolio space (6 values now: cash, pos_val, abs_pos, unrealized_pnl_pct, div_bull, div_bear)
        self.observation_space['portfolio'] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        # State variables (reset in reset())
        self.current_step = 0
        self.episode_step = 0
        self.cash = float(self.initial_cash)
        self.position = 0.0
        self.position_price = 0.0
        self.total_assets = float(self.initial_cash)
        self.peak_assets = float(self.initial_cash)
        self.trade_count = 0
        self._episode_started = False

    def _get_default_features(self, tf: str) -> List[str]:
        """Get default feature columns for a timeframe."""
        from .features import get_feature_columns
        
        base_features = get_feature_columns(include_divergence=(tf == 'M1'))
        return base_features

    def _precompute_arrays(self):
        """Pre-convert DataFrames to numpy arrays for faster observation building."""
        print("[Env] Pre-computing arrays for faster training...")
        
        # Store feature arrays and timestamps for each timeframe
        self.tf_arrays = {}
        self.tf_times = {}
        
        for tf in self.timeframes:
            if tf in self.mtf_data:
                df = self.mtf_data[tf]
                features = self.feature_cols.get(tf, [])
                
                # Store as contiguous numpy array
                arr = df[features].values.astype(np.float32)
                arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
                arr = np.clip(arr, -10, 10)
                self.tf_arrays[tf] = arr
                self.tf_times[tf] = df.index.values
            else:
                self.tf_arrays[tf] = None
                self.tf_times[tf] = None
        
        # Pre-compute M1 price array
        self.m1_prices = self.m1_df['close'].values.astype(np.float32)
        self.m1_times = self.m1_df.index.values
        
        # Pre-compute divergence arrays for M1
        if 'bullish_divergence' in self.m1_df.columns:
            self.m1_bull_div = self.m1_df['bullish_divergence'].values.astype(np.float32)
        else:
            self.m1_bull_div = np.zeros(len(self.m1_df), dtype=np.float32)
            
        if 'bearish_divergence' in self.m1_df.columns:
            self.m1_bear_div = self.m1_df['bearish_divergence'].values.astype(np.float32)
        else:
            self.m1_bear_div = np.zeros(len(self.m1_df), dtype=np.float32)
        
        # Build timestamp index for fast lookup
        # Map each M1 timestamp to its index
        self.m1_time_to_idx = {t: i for i, t in enumerate(self.m1_times)}
        
        # For each TF, map TF timestamps to indices
        self.tf_time_to_idx = {}
        for tf in self.timeframes:
            if self.tf_times[tf] is not None:
                self.tf_time_to_idx[tf] = {t: i for i, t in enumerate(self.tf_times[tf])}
        
        print("[Env] Pre-computation complete")

    def _position_value(self, price: float) -> float:
        """Mark-to-market value of current position."""
        if self.position > 0:
            return self.position * price
        elif self.position < 0:
            return (self.position_price - price) * abs(self.position)
        return 0.0

    def _total_assets_calc(self, price: float) -> float:
        """Calculate total assets."""
        raw = self.cash + self._position_value(price)
        return float(np.clip(raw, 0.0, self.initial_cash * 100))

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Random start within M1 data
        safe_end = len(self.m1_df) - self.max_episode_steps - 1
        lo = self.warmup_bars + max(self.bars_per_tf.values())
        hi = max(lo + 1, safe_end)
        
        if hi > lo:
            self.current_step = int(self.np_random.integers(lo, hi))
        else:
            self.current_step = lo

        self.episode_step = 0
        self.cash = float(self.initial_cash)
        self.position = 0.0
        self.position_price = 0.0
        self.total_assets = float(self.initial_cash)
        self.peak_assets = float(self.initial_cash)
        self.trade_count = 0
        self._episode_started = True

        return self._get_observation(), {}

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Build multi-timeframe observation using pre-computed arrays."""
        obs = {}

        # Current timestamp from M1
        current_time = self.m1_times[self.current_step]

        for tf in self.timeframes:
            if self.tf_arrays[tf] is None:
                # Zero observation if timeframe missing
                obs[tf] = np.zeros(
                    (self.bars_per_tf[tf], self.feature_dims[tf]),
                    dtype=np.float32
                )
                continue

            n_bars = self.bars_per_tf[tf]
            arr = self.tf_arrays[tf]
            times = self.tf_times[tf]
            
            # Find index in TF array (binary search for speed)
            tf_idx = np.searchsorted(times, current_time, side='right')
            
            if tf_idx < n_bars:
                # Pad with zeros if not enough history
                padding = np.zeros((n_bars - tf_idx, arr.shape[1]), dtype=np.float32)
                if tf_idx > 0:
                    data = np.vstack([padding, arr[:tf_idx]])[-n_bars:]
                else:
                    data = padding
            else:
                # Get last n_bars
                data = arr[tf_idx - n_bars:tf_idx]

            obs[tf] = data.astype(np.float32)

        # Portfolio state - use pre-computed arrays
        price = self.m1_prices[self.current_step]
        pos_val = np.clip(self._position_value(price) / self.initial_cash, -5, 5)

        # Calculate unrealized PnL % (critical for knowing when to close)
        if self.position != 0 and self.position_price > 0:
            if self.position > 0:  # Long position
                unrealized_pnl_pct = (price - self.position_price) / self.position_price
            else:  # Short position
                unrealized_pnl_pct = (self.position_price - price) / self.position_price
        else:
            unrealized_pnl_pct = 0.0

        portfolio = np.array([
            np.clip(self.cash / self.initial_cash, 0, 10),
            pos_val,
            abs(pos_val),
            np.clip(unrealized_pnl_pct * 10, -1, 1),  # Unrealized PnL % (scaled)
            self.m1_bull_div[self.current_step],
            self.m1_bear_div[self.current_step],
        ], dtype=np.float32)

        obs['portfolio'] = portfolio

        return obs

    def step(self, action: np.ndarray):
        """Execute one trading step."""
        if not self._episode_started:
            self.reset()

        raw_action = float(np.clip(action[0], -1, 1))

        # Check termination
        if self.current_step >= len(self.m1_df) - 1:
            obs = self._get_observation()
            return obs, 0.0, True, False, self._get_info()

        price = self.m1_df['close'].iloc[self.current_step]
        prev_assets = self.total_assets

        row = self.m1_df.iloc[self.current_step]
        div_bullish = float(row.get('bullish_divergence', 0)) if 'bullish_divergence' in row else 0.0
        div_bearish = float(row.get('bearish_divergence', 0)) if 'bearish_divergence' in row else 0.0

        realized_pnl = 0.0
        trade_type = "Hold"

        max_lots = self.max_position_value / price

        # CLOSE action (near-zero): Close any open position
        if -0.1 <= raw_action <= 0.1:
            if self.position > 0:  # Close long
                realized_pnl = (price - self.position_price) * self.position
                fee = self.position * price * self.transaction_fee
                self.cash += self.position * price - fee
                self.position = 0.0
                self.position_price = 0.0
                trade_type = "Close Long"
                self.trade_count += 1
            elif self.position < 0:  # Close short
                realized_pnl = (self.position_price - price) * abs(self.position)
                fee = abs(self.position) * price * self.transaction_fee
                self.cash -= fee
                self.position = 0.0
                self.position_price = 0.0
                trade_type = "Close Short"
                self.trade_count += 1

        # LONG action
        elif raw_action > 0.1:
            position_pct = min(abs(raw_action), self.max_position_pct)

            # Close short first
            if self.position < 0:
                realized_pnl = (self.position_price - price) * abs(self.position)
                fee = abs(self.position) * price * self.transaction_fee
                self.cash -= fee
                self.position = 0.0
                self.position_price = 0.0
                trade_type = "Cover Short"
                self.trade_count += 1

            # Open/add to long
            if self.cash > 10:
                buy_cash = self.cash * position_pct * 0.95
                new_lots = buy_cash / price
                fee = buy_cash * self.transaction_fee

                total_lots = min(self.position + new_lots, max_lots)
                new_lots = total_lots - self.position

                if new_lots > 0:
                    cost = new_lots * price * (1 + self.transaction_fee)
                    if cost <= self.cash:
                        if self.position > 0:
                            self.position_price = (
                                self.position * self.position_price + new_lots * price
                            ) / (self.position + new_lots)
                        else:
                            self.position_price = price
                        self.position = total_lots
                        self.cash = np.clip(self.cash - cost, 0, self.initial_cash * 20)
                        if trade_type == "Hold":
                            trade_type = "Buy"
                            self.trade_count += 1

        # SHORT action
        elif raw_action < -0.1:
            position_pct = min(abs(raw_action), self.max_position_pct)

            # Close long first
            if self.position > 0:
                realized_pnl = (price - self.position_price) * self.position
                fee = self.position * price * self.transaction_fee
                self.cash += self.position * price - fee
                self.position = 0.0
                self.position_price = 0.0
                trade_type = "Sell"
                self.trade_count += 1

            # Open/add to short
            if self.cash > 10:
                notional = self.cash * position_pct * 0.95
                new_lots = notional / price
                fee = notional * self.transaction_fee

                total_lots_abs = min(abs(self.position) + new_lots, max_lots)
                new_lots = total_lots_abs - abs(self.position)

                if new_lots > 0:
                    cost = new_lots * price * self.transaction_fee
                    if cost <= self.cash:
                        if self.position < 0:
                            self.position_price = (
                                abs(self.position) * self.position_price + new_lots * price
                            ) / total_lots_abs
                        else:
                            self.position_price = price
                        self.position = -total_lots_abs
                        self.cash = np.clip(self.cash - cost, 0, self.initial_cash * 20)
                        if trade_type == "Hold":
                            trade_type = "Short"
                            self.trade_count += 1

        # Advance step
        self.current_step += 1
        self.episode_step += 1

        # Update assets
        next_idx = min(self.current_step, len(self.m1_df) - 1)
        next_price = self.m1_df['close'].iloc[next_idx]

        self.total_assets = self._total_assets_calc(next_price)
        self.peak_assets = max(self.peak_assets, self.total_assets)

        # Calculate reward
        step_reward = 0.0

        # Reward for realized PnL (stronger signal for closing trades)
        if realized_pnl != 0:
            pnl_ratio = realized_pnl / self.initial_cash
            
            # Base reward for realized PnL
            step_reward += np.clip(pnl_ratio, -1, 1) * 0.5
            
            # Bonus for profitable close (encourages taking profit)
            if realized_pnl > 0:
                # Extra reward scaled by profit size
                profit_bonus = min(0.3, pnl_ratio * 2)
                step_reward += profit_bonus
            else:
                # Small penalty for closing at loss (encourages better entries)
                step_reward -= 0.05

        unrealized = (self.total_assets - prev_assets) / self.initial_cash
        step_reward += np.clip(unrealized, -1, 1) * 0.3

        # Bonus for entry on divergence signals
        if trade_type == "Buy" and div_bullish == 1:
            step_reward += 0.3
        if trade_type == "Short" and div_bearish == 1:
            step_reward += 0.3

        # Small penalty for holding (encourages decisions)
        if trade_type == "Hold" and self.position == 0:
            step_reward -= 0.005
        elif trade_type == "Hold" and self.position != 0:
            # Holding a position is okay, but slight time decay
            step_reward -= 0.002

        if self.position != 0 and unrealized > 0:
            step_reward += 0.005

        if self.trade_count > self.max_trades_per_episode:
            step_reward -= 0.1

        drawdown = (self.peak_assets - self.total_assets) / max(self.peak_assets, 1)
        if drawdown > 0.25:
            step_reward -= 0.25
        elif drawdown > 0.15:
            step_reward -= 0.10
        elif drawdown > 0.08:
            step_reward -= 0.05

        step_reward = float(np.clip(step_reward, -self.reward_clip, self.reward_clip))

        # Termination
        terminated = self.current_step >= len(self.m1_df) - 1
        truncated = self.episode_step >= self.max_episode_steps

        return self._get_observation(), step_reward, terminated, truncated, self._get_info()

    def _get_info(self) -> dict:
        """Get environment info."""
        idx = min(self.current_step, len(self.m1_df) - 1)
        price = self.m1_df['close'].iloc[idx]
        return {
            'step': self.current_step,
            'episode_step': self.episode_step,
            'price': price,
            'position': self.position,
            'cash': self.cash,
            'assets': self.total_assets,
            'trade_count': self.trade_count,
        }

    def render(self):
        """Render current state."""
        idx = min(self.current_step, len(self.m1_df) - 1)
        price = self.m1_df['close'].iloc[idx]
        pv = self._position_value(price)
        print(
            f"Step {self.current_step}/{len(self.m1_df)} (ep={self.episode_step}) | "
            f"Price={price:.2f} | Pos={self.position:.4f} | "
            f"Cash=${self.cash:.2f} | PosVal=${pv:.2f} | "
            f"Assets=${self.total_assets:.2f}"
        )

    def get_current_price(self) -> float:
        """Get current price."""
        idx = min(self.current_step, len(self.m1_df) - 1)
        return self.m1_df['close'].iloc[idx]

    def get_current_time(self) -> pd.Timestamp:
        """Get current timestamp."""
        idx = min(self.current_step, len(self.m1_df) - 1)
        return self.m1_df.index[idx]


# ============================================================
# Legacy Environment (for backward compatibility)
# ============================================================

class TradingEnv(gym.Env):
    """
    Legacy single-timeframe Trading Environment for SAC agent.
    Kept for backward compatibility with existing trained models.
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, df: pd.DataFrame, features: List[str] = None):
        super().__init__()

        # Drop warmup rows
        self.df = df.iloc[50:].reset_index(drop=True)

        self.features = features or [
            col for col in self.df.columns
            if col not in ['open', 'high', 'low', 'close', 'volume', 'time']
        ]

        # Environment parameters
        self.lookback_window = ENV_CONFIG.get('lookback_window', 20)
        self.initial_cash = ENV_CONFIG.get('initial_cash', 10_000)
        self.transaction_fee = ENV_CONFIG.get('transaction_fee', 0.0002)
        self.max_position_pct = ENV_CONFIG.get('max_position_pct', 1.0)
        self.max_episode_steps = ENV_CONFIG.get('max_episode_steps', 500)
        self.max_position_value = self.initial_cash * 1.5
        self.max_trades_per_episode = ENV_CONFIG.get('max_trades_per_episode', 50)
        self.reward_clip = 1.0

        # Spaces
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32
        )
        obs_dim = self.lookback_window * len(self.features) + 5
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # State
        self.current_step = self.lookback_window
        self.episode_step = 0
        self.cash = float(self.initial_cash)
        self.position = 0.0
        self.position_price = 0.0
        self.total_assets = float(self.initial_cash)
        self.peak_assets = float(self.initial_cash)
        self.trade_count = 0
        self._episode_started = False

    def _position_value(self, price: float) -> float:
        if self.position > 0:
            return self.position * price
        elif self.position < 0:
            return (self.position_price - price) * abs(self.position)
        return 0.0

    def _total_assets(self, price: float) -> float:
        raw = self.cash + self._position_value(price)
        return float(np.clip(raw, 0.0, self.initial_cash * 100))

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        safe_end = len(self.df) - self.max_episode_steps - 1
        lo = self.lookback_window
        hi = max(lo + 1, safe_end)
        self.current_step = int(self.np_random.integers(lo, hi))

        self.episode_step = 0
        self.cash = float(self.initial_cash)
        self.position = 0.0
        self.position_price = 0.0
        self.total_assets = float(self.initial_cash)
        self.peak_assets = float(self.initial_cash)
        self.trade_count = 0
        self._episode_started = True

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        end = self.current_step + 1
        start = max(0, end - self.lookback_window)

        feature_data = self.df.iloc[start:end][self.features].values
        frame = feature_data.flatten()

        expected = self.lookback_window * len(self.features)
        if len(frame) < expected:
            frame = np.pad(frame, (expected - len(frame), 0), mode='constant')

        frame = np.clip(frame, -10, 10)
        frame = np.nan_to_num(frame, nan=0.0, posinf=1.0, neginf=-1.0)

        idx = min(self.current_step, len(self.df) - 1)
        price = self.df['close'].iloc[idx]
        row = self.df.iloc[idx]

        pos_val = np.clip(
            self._position_value(price) / self.initial_cash, -5, 5
        )
        portfolio = np.array([
            np.clip(self.cash / self.initial_cash, 0, 10),
            pos_val,
            abs(pos_val),
            float(row.get('bullish_divergence', 0)),
            float(row.get('bearish_divergence', 0)),
        ], dtype=np.float32)

        obs = np.concatenate([frame, portfolio]).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    def step(self, action: np.ndarray):
        if not self._episode_started:
            self.reset()

        raw_action = float(np.clip(action[0], -1, 1))

        if self.current_step >= len(self.df) - 1:
            obs = self._get_observation()
            return obs, 0.0, True, False, self._get_info()

        price = self.df['close'].iloc[self.current_step]
        prev_assets = self.total_assets

        row = self.df.iloc[self.current_step]
        div_bullish = float(row.get('bullish_divergence', 0))
        div_bearish = float(row.get('bearish_divergence', 0))

        realized_pnl = 0.0
        trade_type = "Hold"

        max_lots = self.max_position_value / price

        # LONG
        if raw_action > 0.1:
            position_pct = min(abs(raw_action), self.max_position_pct)

            if self.position < 0:
                realized_pnl = (self.position_price - price) * abs(self.position)
                fee = abs(self.position) * price * self.transaction_fee
                self.cash -= fee
                self.position = 0.0
                self.position_price = 0.0
                trade_type = "Cover Short"
                self.trade_count += 1

            if self.cash > 10:
                buy_cash = self.cash * position_pct * 0.95
                new_lots = buy_cash / price
                fee = buy_cash * self.transaction_fee

                total_lots = self.position + new_lots
                total_lots = min(total_lots, max_lots)
                new_lots = total_lots - self.position

                if new_lots > 0:
                    cost = new_lots * price * (1 + self.transaction_fee)
                    if cost <= self.cash:
                        if self.position > 0:
                            self.position_price = (
                                self.position * self.position_price + new_lots * price
                            ) / (self.position + new_lots)
                        else:
                            self.position_price = price
                        self.position = total_lots
                        self.cash = np.clip(self.cash - cost, 0, self.initial_cash * 20)
                        if trade_type == "Hold":
                            trade_type = "Buy"
                            self.trade_count += 1

        # SHORT
        elif raw_action < -0.1:
            position_pct = min(abs(raw_action), self.max_position_pct)

            if self.position > 0:
                realized_pnl = (price - self.position_price) * self.position
                fee = self.position * price * self.transaction_fee
                self.cash += self.position * price - fee
                self.position = 0.0
                self.position_price = 0.0
                trade_type = "Sell"
                self.trade_count += 1

            if self.cash > 10:
                notional = self.cash * position_pct * 0.95
                new_lots = notional / price
                fee = notional * self.transaction_fee

                total_lots_abs = abs(self.position) + new_lots
                total_lots_abs = min(total_lots_abs, max_lots)
                new_lots = total_lots_abs - abs(self.position)

                if new_lots > 0:
                    cost = new_lots * price * self.transaction_fee
                    if cost <= self.cash:
                        if self.position < 0:
                            self.position_price = (
                                abs(self.position) * self.position_price + new_lots * price
                            ) / total_lots_abs
                        else:
                            self.position_price = price
                        self.position = -total_lots_abs
                        self.cash = np.clip(self.cash - cost, 0, self.initial_cash * 20)
                        if trade_type == "Hold":
                            trade_type = "Short"
                            self.trade_count += 1

        self.current_step += 1
        self.episode_step += 1

        next_idx = min(self.current_step, len(self.df) - 1)
        next_price = self.df['close'].iloc[next_idx]

        self.total_assets = self._total_assets(next_price)
        self.peak_assets = max(self.peak_assets, self.total_assets)

        # Reward
        step_reward = 0.0

        if realized_pnl != 0:
            step_reward += np.clip(realized_pnl / self.initial_cash, -1, 1) * 0.3

        unrealized = (self.total_assets - prev_assets) / self.initial_cash
        step_reward += np.clip(unrealized, -1, 1) * 0.5

        if trade_type == "Buy" and div_bullish == 1:
            step_reward += 0.5
        if trade_type == "Short" and div_bearish == 1:
            step_reward += 0.5

        if trade_type == "Hold":
            step_reward -= 0.01

        if self.position != 0 and unrealized > 0:
            step_reward += 0.005

        if self.trade_count > self.max_trades_per_episode:
            step_reward -= 0.1

        drawdown = (self.peak_assets - self.total_assets) / max(self.peak_assets, 1)
        if drawdown > 0.25:
            step_reward -= 0.25
        elif drawdown > 0.15:
            step_reward -= 0.10
        elif drawdown > 0.08:
            step_reward -= 0.05

        step_reward = float(np.clip(step_reward, -self.reward_clip, self.reward_clip))

        terminated = self.current_step >= len(self.df) - 1
        truncated = self.episode_step >= self.max_episode_steps

        return self._get_observation(), step_reward, terminated, truncated, self._get_info()

    def _get_info(self) -> dict:
        idx = min(self.current_step, len(self.df) - 1)
        price = self.df['close'].iloc[idx]
        return {
            'step': self.current_step,
            'episode_step': self.episode_step,
            'price': price,
            'position': self.position,
            'cash': self.cash,
            'assets': self.total_assets,
            'trade_count': self.trade_count,
        }

    def render(self):
        idx = min(self.current_step, len(self.df) - 1)
        price = self.df['close'].iloc[idx]
        pv = self._position_value(price)
        print(
            f"Step {self.current_step}/{len(self.df)} (ep={self.episode_step}) | "
            f"Price={price:.2f} | Pos={self.position:.4f} | "
            f"Cash=${self.cash:.2f} | PosVal=${pv:.2f} | "
            f"Assets=${self.total_assets:.2f}"
        )

    def get_current_price(self) -> float:
        idx = min(self.current_step, len(self.df) - 1)
        return self.df['close'].iloc[idx]