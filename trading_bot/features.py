"""
Technical Indicators and Features for Multi-Timeframe Trading
- Pruned feature set for efficiency
- Features computed at each timeframe
- Divergence detection (M1 only for entry signals)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from .config import MTF_FEATURE_CONFIG, MTF_CONFIG


# ============================================================
# Core Technical Indicators (computed at each timeframe)
# ============================================================

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def add_momentum_features(df: pd.DataFrame, cfg: dict = None) -> pd.DataFrame:
    """
    Add momentum features: RSI, MACD
    
    Args:
        df: OHLCV dataframe
        cfg: Feature config
        
    Returns:
        DataFrame with momentum features
    """
    df = df.copy()
    if cfg is None:
        cfg = MTF_FEATURE_CONFIG
    
    # RSI
    period = cfg.get('rsi_period', 14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    fast = cfg.get('macd_fast', 12)
    slow = cfg.get('macd_slow', 26)
    signal = cfg.get('macd_signal', 9)
    
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    return df


def add_trend_features(df: pd.DataFrame, cfg: dict = None) -> pd.DataFrame:
    """
    Add trend features: EMAs, ADX
    
    Args:
        df: OHLCV dataframe
        cfg: Feature config
        
    Returns:
        DataFrame with trend features
    """
    df = df.copy()
    if cfg is None:
        cfg = MTF_FEATURE_CONFIG
    
    # EMAs
    ema_periods = cfg.get('ema_periods', [20, 50])
    atr = calculate_atr(df, cfg.get('atr_period', 14))
    
    for period in ema_periods:
        ema = df['close'].ewm(span=period, adjust=False).mean()
        df[f'ema{period}'] = ema
        # Distance from EMA (normalized by ATR)
        df[f'dist_ema{period}'] = (df['close'] - ema) / atr
    
    # EMA alignment (trend direction)
    if len(ema_periods) >= 2:
        df['ema_alignment'] = (
            (df[f'ema{ema_periods[0]}'] > df[f'ema{ema_periods[1]}']).astype(int) * 2 - 1
        )
    
    # ADX (trend strength)
    adx_period = cfg.get('adx_period', 14)
    high_diff = df['high'].diff()
    low_diff = -df['low'].diff()
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
    
    plus_di = 100 * (plus_dm.rolling(window=adx_period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=adx_period).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
    df['adx'] = dx.rolling(window=adx_period).mean()
    df['adx_plus'] = plus_di
    df['adx_minus'] = minus_di
    
    # ATR normalized
    df['atr'] = atr
    df['atr_norm'] = atr / df['close']
    
    return df


def add_structure_features(df: pd.DataFrame, cfg: dict = None) -> pd.DataFrame:
    """
    Add price structure features: BOS, Swing levels, Range position
    
    Args:
        df: OHLCV dataframe
        cfg: Feature config
        
    Returns:
        DataFrame with structure features
    """
    df = df.copy()
    if cfg is None:
        cfg = MTF_FEATURE_CONFIG
    
    bos_window = cfg.get('bos_window', 10)
    swing_window = cfg.get('swing_window', 20)
    atr = df.get('atr', calculate_atr(df, 14))
    
    # Break of Structure (BOS)
    swing_high = df['high'].rolling(bos_window).max().shift(1)
    swing_low = df['low'].rolling(bos_window).min().shift(1)
    
    df['bos_bull'] = (df['high'] > swing_high).astype(int)
    df['bos_bear'] = (df['low'] < swing_low).astype(int)
    
    # Distance to swing levels (normalized by ATR)
    rolling_high = df['high'].rolling(swing_window).max()
    rolling_low = df['low'].rolling(swing_window).min()
    
    df['dist_swing_high'] = ((rolling_high - df['close']) / atr).clip(0, 20)
    df['dist_swing_low'] = ((df['close'] - rolling_low) / atr).clip(0, 20)
    
    # Range position (0 = at low, 1 = at high)
    price_range = rolling_high - rolling_low
    df['range_position'] = ((df['close'] - rolling_low) / price_range.replace(0, np.nan)).clip(0, 1).fillna(0.5)
    
    # Fair Value Gap (FVG)
    df['bull_fvg'] = (df['low'] > df['high'].shift(2)).astype(int)
    df['bear_fvg'] = (df['high'] < df['low'].shift(2)).astype(int)
    
    return df


def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add session/time features.
    
    Assumes broker server time (typically UTC+2/UTC+3).
    
    Sessions:
    - Asia: 01:00-08:00
    - London: 09:00-12:00 (kill zone)
    - NY: 14:00-17:00 (kill zone)
    - London/NY overlap: 14:00-16:00
    """
    df = df.copy()
    
    if not hasattr(df.index, 'hour'):
        return df
    
    hour = df.index.hour
    minute = df.index.minute
    time_frac = hour + minute / 60.0
    
    # Cyclical time encoding
    df['hour_sin'] = np.sin(2 * np.pi * time_frac / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * time_frac / 24.0)
    
    # Day of week cyclical
    dow = df.index.dayofweek
    df['dow_sin'] = np.sin(2 * np.pi * dow / 5.0)
    df['dow_cos'] = np.cos(2 * np.pi * dow / 5.0)
    
    # Session flags
    df['asia_session'] = ((hour >= 1) & (hour < 8)).astype(int)
    df['london_kz'] = ((hour >= 9) & (hour < 12)).astype(int)
    df['ny_kz'] = ((hour >= 14) & (hour < 17)).astype(int)
    df['london_ny_overlap'] = ((hour >= 14) & (hour < 16)).astype(int)
    
    return df


# ============================================================
# Divergence Features (M1 only for entry signals)
# ============================================================

def detect_pivot_low(df: pd.DataFrame, idx: int, left: int = 3, right: int = 3) -> bool:
    """Detect if current bar is a pivot low"""
    if idx + right >= len(df) or idx - left < 0:
        return False
    
    current_low = df['low'].iloc[idx]
    
    for j in range(1, left + 1):
        if df['low'].iloc[idx - j] <= current_low:
            return False
    
    for j in range(1, right + 1):
        if df['low'].iloc[idx + j] <= current_low:
            return False
    
    return True


def detect_pivot_high(df: pd.DataFrame, idx: int, left: int = 3, right: int = 3) -> bool:
    """Detect if current bar is a pivot high"""
    if idx + right >= len(df) or idx - left < 0:
        return False
    
    current_high = df['high'].iloc[idx]
    
    for j in range(1, left + 1):
        if df['high'].iloc[idx - j] >= current_high:
            return False
    
    for j in range(1, right + 1):
        if df['high'].iloc[idx + j] >= current_high:
            return False
    
    return True


def add_divergence_features(df: pd.DataFrame, cfg: dict = None) -> pd.DataFrame:
    """
    Add RSI divergence features (computationally expensive, use only on M1).
    
    Features:
    - bullish_divergence: Regular bullish (price LL, RSI HL)
    - hidden_bullish: Hidden bullish (price HL, RSI LL)
    - bearish_divergence: Regular bearish (price HH, RSI LH)
    - hidden_bearish: Hidden bearish (price LH, RSI HH)
    """
    df = df.copy()
    if cfg is None:
        cfg = MTF_FEATURE_CONFIG
    
    # Initialize columns
    df['bullish_divergence'] = 0
    df['hidden_bullish'] = 0
    df['bearish_divergence'] = 0
    df['hidden_bearish'] = 0
    
    pivot_left = cfg.get('pivot_left', 3)
    pivot_right = cfg.get('pivot_right', 3)
    lookback = cfg.get('rsi_divergence_lookback', 5)
    
    # Track last pivots
    last_pivot_low_rsi = None
    last_pivot_low_price = None
    last_pivot_low_idx = None
    
    last_pivot_high_rsi = None
    last_pivot_high_price = None
    last_pivot_high_idx = None
    
    start_idx = pivot_right + pivot_left + lookback
    
    for i in range(start_idx, len(df) - pivot_right):
        # Pivot low detection
        if detect_pivot_low(df, i, pivot_left, pivot_right):
            current_rsi = df['rsi'].iloc[i]
            current_price = df['low'].iloc[i]
            
            if last_pivot_low_idx is not None:
                bar_diff = i - last_pivot_low_idx
                
                # Regular Bullish: Price LL, RSI HL
                if (bar_diff >= lookback and 
                    current_price < last_pivot_low_price and 
                    current_rsi > last_pivot_low_rsi):
                    df.iloc[i, df.columns.get_loc('bullish_divergence')] = 1
                
                # Hidden Bullish: Price HL, RSI LL
                if (bar_diff >= lookback and 
                    current_price > last_pivot_low_price and 
                    current_rsi < last_pivot_low_rsi):
                    df.iloc[i, df.columns.get_loc('hidden_bullish')] = 1
            
            last_pivot_low_rsi = current_rsi
            last_pivot_low_price = current_price
            last_pivot_low_idx = i
        
        # Pivot high detection
        if detect_pivot_high(df, i, pivot_left, pivot_right):
            current_rsi = df['rsi'].iloc[i]
            current_price = df['high'].iloc[i]
            
            if last_pivot_high_idx is not None:
                bar_diff = i - last_pivot_high_idx
                
                # Regular Bearish: Price HH, RSI LH
                if (bar_diff >= lookback and 
                    current_price > last_pivot_high_price and 
                    current_rsi < last_pivot_high_rsi):
                    df.iloc[i, df.columns.get_loc('bearish_divergence')] = 1
                
                # Hidden Bearish: Price LH, RSI HH
                if (bar_diff >= lookback and 
                    current_price < last_pivot_high_price and 
                    current_rsi > last_pivot_high_rsi):
                    df.iloc[i, df.columns.get_loc('hidden_bearish')] = 1
            
            last_pivot_high_rsi = current_rsi
            last_pivot_high_price = current_price
            last_pivot_high_idx = i
    
    return df


# ============================================================
# Feature Pipeline
# ============================================================

def add_core_features(df: pd.DataFrame, 
                      include_divergence: bool = False,
                      cfg: dict = None) -> pd.DataFrame:
    """
    Add all core features to dataframe.
    
    Args:
        df: OHLCV dataframe
        include_divergence: Whether to add divergence features (expensive)
        cfg: Feature config
        
    Returns:
        DataFrame with all features
    """
    df = df.copy()
    if cfg is None:
        cfg = MTF_FEATURE_CONFIG
    
    # Add features in order (some depend on others)
    df = add_momentum_features(df, cfg)
    df = add_trend_features(df, cfg)
    df = add_structure_features(df, cfg)
    df = add_session_features(df)
    
    # Divergence only on M1 (expensive)
    if include_divergence:
        df = add_divergence_features(df, cfg)
    
    return df


def add_mtf_features(mtf_data: Dict[str, pd.DataFrame],
                     cfg: dict = None) -> Dict[str, pd.DataFrame]:
    """
    Add features to all timeframes.
    
    Args:
        mtf_data: Dictionary mapping timeframe to OHLCV dataframe
        cfg: Feature config
        
    Returns:
        Dictionary with features added to each timeframe
    """
    if cfg is None:
        cfg = MTF_FEATURE_CONFIG
    
    result = {}
    
    for tf, df in mtf_data.items():
        print(f"  Adding features to {tf}...")
        
        # Divergence only on M1
        include_div = (tf == "M1")
        
        result[tf] = add_core_features(df, include_divergence=include_div, cfg=cfg)
    
    return result


def get_feature_columns(include_divergence: bool = False) -> List[str]:
    """
    Get list of feature column names.
    
    Args:
        include_divergence: Whether to include divergence features
        
    Returns:
        List of feature column names
    """
    # Core features (computed at each timeframe)
    core_features = [
        # Momentum
        'rsi', 'macd', 'macd_signal', 'macd_hist',
        
        # Trend
        'ema20', 'ema50', 'dist_ema20', 'dist_ema50', 'ema_alignment',
        'adx', 'adx_plus', 'adx_minus',
        
        # Volatility
        'atr', 'atr_norm',
        
        # Structure
        'bos_bull', 'bos_bear', 'dist_swing_high', 'dist_swing_low',
        'range_position', 'bull_fvg', 'bear_fvg',
        
        # Session
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'asia_session', 'london_kz', 'ny_kz', 'london_ny_overlap',
    ]
    
    # Divergence features (M1 only)
    divergence_features = [
        'bullish_divergence', 'hidden_bullish',
        'bearish_divergence', 'hidden_bearish'
    ]
    
    if include_divergence:
        return core_features + divergence_features
    
    return core_features


def get_raw_columns() -> List[str]:
    """Get list of raw OHLCV columns"""
    return ['open', 'high', 'low', 'close', 'volume']


def normalize_features(df: pd.DataFrame, 
                       feature_cols: List[str] = None,
                       scaler=None) -> Tuple[pd.DataFrame, object]:
    """
    Normalize features using MinMaxScaler.
    
    Args:
        df: DataFrame with features
        feature_cols: Columns to normalize
        scaler: Pre-fitted scaler (optional)
        
    Returns:
        Normalized DataFrame and scaler
    """
    from sklearn.preprocessing import MinMaxScaler
    
    if feature_cols is None:
        feature_cols = get_feature_columns(include_divergence=True)
    
    # Only normalize columns that exist
    existing_cols = [c for c in feature_cols if c in df.columns]
    
    if not existing_cols:
        return df, scaler
    
    features = df[existing_cols].values.copy()
    
    # Handle inf/nan
    features = np.where(np.isinf(features), np.nan, features)
    col_means = np.nanmean(features, axis=0)
    col_means = np.where(np.isnan(col_means), 0, col_means)
    
    for i in range(features.shape[1]):
        mask = np.isnan(features[:, i])
        features[mask, i] = col_means[i]
    
    # Scale
    if scaler is None:
        scaler = MinMaxScaler()
        features = scaler.fit_transform(features)
    else:
        features = scaler.transform(features)
    
    # Final cleanup
    features = np.where(np.isnan(features), 0, features)
    features = np.where(np.isinf(features), 0, features)
    
    df = df.copy()
    df[existing_cols] = features
    
    return df, scaler


def drop_warmup_bars(df: pd.DataFrame, warmup_bars: int = 300) -> pd.DataFrame:
    """
    Drop initial bars needed for indicator warmup.
    
    EMA256 needs ~300 bars, but we simplified to EMA50,
    so 100 bars should be sufficient.
    """
    if len(df) > warmup_bars:
        return df.iloc[warmup_bars:].copy()
    return df


# ============================================================
# Legacy functions for backward compatibility
# ============================================================

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy function - add all technical indicators"""
    return add_core_features(df, include_divergence=False)


def add_ict_features(df: pd.DataFrame, swing_window: int = 20) -> pd.DataFrame:
    """Legacy function - ICT features are now in add_structure_features"""
    return add_structure_features(df)


def add_goonix_features(df: pd.DataFrame, bos_window: int = 10, swing_window: int = 20) -> pd.DataFrame:
    """Legacy function - Goonix features merged into core features"""
    return add_structure_features(df, {'bos_window': bos_window, 'swing_window': swing_window})


def prepare_data(df: pd.DataFrame, add_indicators: bool = True,
                 add_divergence: bool = True, add_ict: bool = True,
                 add_goonix: bool = True, normalize: bool = True):
    """Legacy function - prepare data with features"""
    df = df.copy()
    
    df = add_core_features(df, include_divergence=add_divergence)
    
    df.dropna(inplace=True)
    
    if normalize:
        df, _ = normalize_features(df)
    
    return df