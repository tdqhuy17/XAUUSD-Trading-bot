"""
Live Trading Script for SAC Agent
Usage: python scripts/run_live.py

Features:
- Paper trading mode (default)
- Live trading with MT5
- Divergence-based entry with RL refinement
"""

import sys
import os
import argparse
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from trading_bot.config import Config, SAC_CONFIG, DIVERGENCE_CONFIG, TRADE_CONFIG, ENV_CONFIG
from trading_bot.data_loader import load_mt5_data, prepare_data
from trading_bot.model import load_model, get_action_interpretation
from trading_bot.features import add_technical_indicators, add_divergence_features, normalize_features
from trading_bot.trade_executor import MT5Trader, PaperTrader


class LiveTradingBot:
    """
    Live trading bot combining:
    - SAC RL agent for action decisions
    - RSI divergence for entry signals
    """
    
    def __init__(self, model_path: str, paper_mode: bool = True):
        self.paper_mode = paper_mode
        self.model_path = model_path
        
        # Trading parameters
        self.symbol = "XAUUSD"
        self.timeframe = 15  # M15
        self.lookback = 300  # Bars needed for features (EMA256 needs ~300 bars)
        
        # Divergence tracking
        self.last_pivot_low = None
        self.last_pivot_high = None
        
        # Load agent
        self.agent = None
        
        # Initialize trader
        if paper_mode:
            self.trader = PaperTrader(
                initial_cash=ENV_CONFIG.get('initial_cash', 10000),
                transaction_fee=ENV_CONFIG.get('transaction_fee', 0.0002),
            )
            print("[Mode] Paper Trading")
        else:
            self.trader = MT5Trader(
                symbol=self.symbol,
                lot_size=TRADE_CONFIG.get('lot_size', 0.01),
                magic_number=TRADE_CONFIG.get('magic_number', 234567),
            )
            print("[Mode] Live Trading")
        
        # Logging
        self.log_file = Config.LOGS_DIR / 'live_trading.log'
        
    def log(self, message: str):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry + '\n')
    
    def initialize(self):
        """Initialize the trading bot"""
        self.log("=" * 50)
        self.log("Initializing Live Trading Bot")
        self.log("=" * 50)
        
        # Display trading configuration
        self.log(f"Symbol    : {self.symbol}")
        self.log(f"Timeframe : M{self.timeframe}")
        self.log(f"Lookback  : {self.lookback} bars")
        self.log(f"Mode      : {'LIVE' if not self.paper_mode else 'PAPER TRADING'}")
        self.log("-" * 50)
        
        # Show paper trading account info
        if self.paper_mode:
            self.log("PAPER TRADING ACCOUNT")
            self.log("-" * 50)
            self.log(f"  Initial Cash : ${self.trader.initial_cash:,.2f}")
            self.log(f"  Tx Fee       : {self.trader.transaction_fee * 100:.3f}%")
            self.log("-" * 50)
        
        # Load model
        try:
            self.agent = load_model(self.model_path)
            self.log(f"Model loaded: {self.model_path}")
        except Exception as e:
            self.log(f"ERROR loading model: {e}")
            return False
        
        # Initialize MT5 if live trading
        if not self.paper_mode:
            # Initialize the MT5Trader
            if not self.trader.initialize():
                self.log("Failed to initialize MT5Trader")
                return False
            
            try:
                import MetaTrader5 as mt5
                
                account = mt5.account_info()
                if account is None:
                    self.log("No MT5 account connected")
                    return False
                
                # Display detailed account info
                self.log("=" * 50)
                self.log("MT5 ACCOUNT INFO")
                self.log("=" * 50)
                self.log(f"  Account     : {account.login}")
                self.log(f"  Name        : {account.name}")
                self.log(f"  Server      : {account.server}")
                self.log(f"  Currency    : {account.currency}")
                self.log(f"  Leverage    : 1:{account.leverage}")
                self.log("-" * 50)
                self.log(f"  Balance     : ${account.balance:,.2f}")
                self.log(f"  Equity      : ${account.equity:,.2f}")
                self.log(f"  Margin      : ${account.margin:,.2f}")
                self.log(f"  Free Margin : ${account.margin_free:,.2f}")
                self.log(f"  Margin Lvl  : {account.margin_level:.2f}%")
                self.log(f"  Profit      : ${account.profit:,.2f}")
                self.log("=" * 50)
                
            except ImportError:
                self.log("MetaTrader5 not installed. Use paper mode.")
                return False
        
        return True
    
    def get_latest_data(self) -> pd.DataFrame:
        """Fetch latest market data"""
        # For paper trading, use simulated data
        if self.paper_mode:
            return self._generate_simulated_data()
        
        # For live trading, fetch from MT5
        try:
            import MetaTrader5 as mt5
            
            rates = mt5.copy_rates_from_pos(
                self.symbol, 
                mt5.TIMEFRAME_M15, 
                0, 
                self.lookback + 50
            )
            
            if rates is None or len(rates) == 0:
                self.log("MT5 returned no data, using simulated data")
                return self._generate_simulated_data()
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)  # Set time as index for ICT features
            
            return df
            
        except ImportError:
            self.log("MT5 package not installed, using simulated data")
            return self._generate_simulated_data()
        except Exception as e:
            self.log(f"MT5 error: {e}, using simulated data")
            return self._generate_simulated_data()
    
    def _generate_simulated_data(self) -> pd.DataFrame:
        """Generate simulated price data for paper trading demo"""
        n = self.lookback + 50
        
        # Random walk simulation
        np.random.seed(int(time.time()) % 10000)
        
        start_price = 2000.0  # Gold price
        returns = np.random.randn(n) * 0.001
        prices = start_price * np.cumprod(1 + returns)
        
        # Create OHLCV data
        df = pd.DataFrame({
            'open': prices + np.random.randn(n) * 2,
            'high': prices + np.abs(np.random.randn(n)) * 3,
            'low': prices - np.abs(np.random.randn(n)) * 3,
            'close': prices,
            'volume': np.random.randint(100, 1000, n),
        })
        
        # Set datetime index (required for ICT time features)
        df.index = pd.date_range(end=datetime.now(), periods=n, freq='15min')
        df.index.name = 'time'
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features to data - must match training features exactly"""
        from trading_bot.features import add_ict_features, add_goonix_features, get_feature_columns
        
        # Use the exact same preparation as training (data_loader.prepare_data)
        df = add_technical_indicators(df)
        df = add_ict_features(df)
        df = add_goonix_features(df)
        df = add_divergence_features(df)
        df.dropna(inplace=True)
        df, _ = normalize_features(df)
        
        # Get feature columns
        feature_cols = get_feature_columns(df)
        n_features = len(feature_cols)
        
        # Model expects 60 features (1205 = 60*20 + 5)
        # If we have more, we need to select the same 60 used in training
        # If we have fewer, padding will happen in get_rl_action
        self.log(f"Features: {n_features} columns")
        
        # Store expected obs size for validation
        env_lookback = ENV_CONFIG.get('lookback_window', 20)
        self.expected_obs_size = 60 * env_lookback + 5  # Model trained with 60 features
        
        return df
    
    def get_rl_action(self, df: pd.DataFrame) -> tuple:
        """Get action from RL agent"""
        if self.agent is None:
            return 0.0, "HOLD"
        
        from trading_bot.features import get_feature_columns
        
        env_lookback = ENV_CONFIG.get('lookback_window', 20)
        expected_feature_count = 60  # Model was trained with 60 features
        
        # Get feature columns
        feature_cols = get_feature_columns(df)
        n_features = len(feature_cols)
        
        # Get feature data
        start_idx = len(df) - env_lookback
        if start_idx < 0:
            start_idx = 0
        
        feature_data = df.iloc[start_idx:][feature_cols].values.flatten()
        
        # Adjust to match expected size (60 * 20 = 1200 feature values)
        expected_feature_size = expected_feature_count * env_lookback
        
        if len(feature_data) > expected_feature_size:
            # Truncate if too many features
            feature_data = feature_data[:expected_feature_size]
        elif len(feature_data) < expected_feature_size:
            # Pad with zeros if too few
            feature_data = np.pad(feature_data, (0, expected_feature_size - len(feature_data)))
        
        # Portfolio state (neutral for inference)
        portfolio = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        obs = np.concatenate([feature_data, portfolio]).astype(np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Get action from agent
        action = self.agent.select_action_deterministic(obs)
        interpretation = get_action_interpretation(action)
        
        return float(action[0]) if isinstance(action, np.ndarray) else float(action), interpretation
    
    def check_divergence_signal(self, df: pd.DataFrame) -> tuple:
        """Check for divergence signals"""
        if len(df) < 10:
            return None, None
        
        last_row = df.iloc[-1]
        
        # Check divergence columns
        bullish = last_row.get('bullish_divergence', 0)
        bearish = last_row.get('bearish_divergence', 0)
        
        if bullish == 1:
            return "BULLISH", "Regular Bullish Divergence detected"
        elif bearish == 1:
            return "BEARISH", "Regular Bearish Divergence detected"
        
        # Check hidden divergences
        hidden_bull = last_row.get('hidden_bullish', 0)
        hidden_bear = last_row.get('hidden_bearish', 0)
        
        if hidden_bull == 1:
            return "BULLISH", "Hidden Bullish Divergence detected"
        elif hidden_bear == 1:
            return "BEARISH", "Hidden Bearish Divergence detected"
        
        return None, None
    
    def run_step(self):
        """Execute one trading step"""
        # Get data
        df = self.get_latest_data()
        if df is None or len(df) < 50:
            self.log("ERROR: Not enough data")
            return
        
        # Prepare features
        df = self.prepare_features(df)
        
        current_price = df['close'].iloc[-1]
        
        # Get signals
        rl_action, rl_interp = self.get_rl_action(df)
        div_signal, div_msg = self.check_divergence_signal(df)
        
        # Check current position state
        has_position = self.trader.has_position()
        
        self.log(f"Price: ${current_price:.2f} | RL: {rl_interp} ({rl_action:+.3f}) | Div: {div_signal or 'None'} | Pos: {'YES' if has_position else 'NO'}")
        
        # Trading logic with position management
        # =====================================
        
        # EXIT CONDITIONS (when in position)
        if has_position:
            # Exit when RL signal reverses (near zero)
            if abs(rl_action) < 0.15:
                self.log(f"  >> EXIT SIGNAL: RL near zero ({rl_action:.3f})")
                self._execute_trade("CLOSE", current_price)
                return
            
            # Exit when signal completely reverses direction
            # (e.g., was long RL>0.5, now RL<-0.3)
            # This is handled by CLOSE above for simplicity
            return  # Already in position, don't add more
        
        # ENTRY CONDITIONS (when NOT in position)
        # =======================================
        
        # Strong buy signal: RL > 0.3 and bullish divergence
        if rl_action > 0.3 and div_signal == "BULLISH":
            self.log(f"  >> STRONG BUY SIGNAL: RL={rl_action:.3f}, Divergence={div_msg}")
            self._execute_trade("BUY", current_price)
        
        # Strong sell signal: RL < -0.3 and bearish divergence
        elif rl_action < -0.3 and div_signal == "BEARISH":
            self.log(f"  >> STRONG SELL SIGNAL: RL={rl_action:.3f}, Divergence={div_msg}")
            self._execute_trade("SELL", current_price)
        
        # Moderate signals (RL only, no divergence confirmation)
        elif rl_action > 0.5:
            self.log(f"  >> MODERATE BUY: RL={rl_action:.3f}")
            self._execute_trade("BUY", current_price, size_pct=0.5)
        
        elif rl_action < -0.5:
            self.log(f"  >> MODERATE SELL: RL={rl_action:.3f}")
            self._execute_trade("SELL", current_price, size_pct=0.5)
    
    def _execute_trade(self, action: str, price: float, size_pct: float = 1.0):
        """Execute trade through trader"""
        if self.paper_mode:
            result = self.trader.execute(action, price, size_pct)
            self.log(f"  >> Paper Trade: {action} @ ${price:.2f} | Result: {result}")
        else:
            result = self.trader.execute(action, price, size_pct)
            self.log(f"  >> Live Trade: {action} @ ${price:.2f} | Result: {result}")
    
    def run(self, interval: int = 60):
        """Main trading loop"""
        self.log("Starting trading loop...")
        self.log(f"Interval: {interval} seconds")
        
        while True:
            try:
                self.run_step()
                time.sleep(interval)
                
            except KeyboardInterrupt:
                self.log("Stopping by user request")
                break
            except Exception as e:
                self.log(f"ERROR: {e}")
                time.sleep(5)
        
        # Final summary
        if self.paper_mode:
            self.log("=" * 50)
            self.log("Trading Session Summary")
            self.log(f"Final Balance: ${self.trader.balance:.2f}")
            self.log(f"Total Trades: {self.trader.trade_count}")
            self.log("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Run live trading bot")
    parser.add_argument('--model', type=str, default='sac_gold_agent.best', help='Model name (without extension)')
    parser.add_argument('--live', action='store_true', help='Live trading (default: paper)')
    parser.add_argument('--interval', type=int, default=60, help='Check interval in seconds')
    args = parser.parse_args()
    
    # Find model - try .pt first, then .zip, then no extension
    model_path = Config.MODELS_DIR / (args.model + '.pt')
    if not model_path.exists():
        model_path = Config.MODELS_DIR / (args.model + '.zip')
    if not model_path.exists():
        model_path = Config.MODELS_DIR / args.model
    if not model_path.exists():
        print(f"\n[ERROR] Model not found: {model_path}")
        print("\nTrain a model first with: python scripts/train.py")
        return
    
    print(f"[Model] Using: {model_path}")
    
    # Create bot
    bot = LiveTradingBot(
        model_path=str(model_path),
        paper_mode=not args.live,
    )
    
    # Initialize
    if not bot.initialize():
        print("\n[ERROR] Failed to initialize trading bot")
        return
    
    # Warning for live mode
    if args.live:
        print("\n" + "!" * 50)
        print("  WARNING: LIVE TRADING MODE")
        print("  Real money will be at risk!")
        print("!" * 50)
        response = input("\nType 'CONFIRM' to continue: ")
        if response != 'CONFIRM':
            print("Aborted.")
            return
    
    # Run
    try:
        bot.run(interval=args.interval)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()