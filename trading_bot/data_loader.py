"""
Data Loader for MT5 and CSV files
- M1 data merging (2015-2026)
- Multi-timeframe resampling
- Efficient chunked processing
"""

import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from .config import DATA_CONFIG, MTF_CONFIG, Config


def parse_m1_csv_format(file_path: str) -> pd.DataFrame:
    """
    Parse M1 CSV file with format: datetime;open;high;low;close;volume
    No header, semicolon-separated
    
    Example: 20150101 180100;1184.130000;1184.440000;1184.040000;1184.130000;0
    """
    # Read with no header, semicolon separator
    df = pd.read_csv(
        file_path, 
        sep=';', 
        header=None,
        names=['datetime', 'open', 'high', 'low', 'close', 'volume'],
        dtype={
            'datetime': str,
            'open': np.float64,
            'high': np.float64,
            'low': np.float64,
            'close': np.float64,
            'volume': np.float64
        }
    )
    
    # Parse datetime: YYYYMMDD HHMMSS
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%d %H%M%S')
    df.set_index('datetime', inplace=True)
    
    return df


def merge_m1_data(data_dir: str = None, 
                  train_start: str = "2015-01-01",
                  train_end: str = "2024-12-31",
                  test_start: str = "2025-01-01",
                  test_end: str = "2026-02-28") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge all M1 CSV files and split into train/test sets.
    
    Uses chunked reading to avoid loading all files at once into memory.
    
    Args:
        data_dir: Directory containing M1 CSV files
        train_start/end: Training period
        test_start/end: Testing period
        
    Returns:
        train_df, test_df: Merged M1 dataframes
    """
    if data_dir is None:
        data_dir = Config.DATA_DIR
    
    # Find all M1 files
    pattern = os.path.join(data_dir, "DAT_ASCII_XAUUSD_M1_*.csv")
    files = sorted(glob.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"No M1 files found matching: {pattern}")
    
    print(f"[Data] Found {len(files)} M1 files")
    
    # Collect dataframes for each period
    train_dfs = []
    test_dfs = []
    
    train_start_dt = pd.to_datetime(train_start)
    train_end_dt = pd.to_datetime(train_end)
    test_start_dt = pd.to_datetime(test_start)
    test_end_dt = pd.to_datetime(test_end)
    
    for file_path in files:
        filename = os.path.basename(file_path)
        print(f"  Processing: {filename}")
        
        # Parse file
        df = parse_m1_csv_format(file_path)
        
        # Check file date range
        file_start = df.index.min()
        file_end = df.index.max()
        
        # Determine if file belongs to train or test
        if file_end < train_start_dt:
            print(f"    Skipping (before train period)")
            continue
        if file_start > test_end_dt:
            print(f"    Skipping (after test period)")
            continue
        
        # Split data
        train_mask = (df.index >= train_start_dt) & (df.index <= train_end_dt)
        test_mask = (df.index >= test_start_dt) & (df.index <= test_end_dt)
        
        if train_mask.any():
            train_dfs.append(df[train_mask])
            print(f"    Train: {train_mask.sum()} bars")
        
        if test_mask.any():
            test_dfs.append(df[test_mask])
            print(f"    Test: {test_mask.sum()} bars")
    
    # Concatenate
    print("\n[Data] Merging dataframes...")
    train_df = pd.concat(train_dfs) if train_dfs else pd.DataFrame()
    test_df = pd.concat(test_dfs) if test_dfs else pd.DataFrame()
    
    # Sort by index
    if not train_df.empty:
        train_df.sort_index(inplace=True)
        print(f"  Train: {len(train_df):,} bars ({train_df.index.min()} to {train_df.index.max()})")
    
    if not test_df.empty:
        test_df.sort_index(inplace=True)
        print(f"  Test: {len(test_df):,} bars ({test_df.index.min()} to {test_df.index.max()})")
    
    return train_df, test_df


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample OHLCV data to a higher timeframe.
    
    Args:
        df: M1 OHLCV dataframe with datetime index
        timeframe: Target timeframe (M5, M15, M30, H1)
        
    Returns:
        Resampled dataframe
    """
    # Map timeframe to pandas offset string
    tf_map = {
        "M1": "1min",
        "M5": "5min",
        "M15": "15min",
        "M30": "30min",
        "H1": "1h",
        "H4": "4h",
        "D1": "1d",
    }
    
    offset = tf_map.get(timeframe)
    if offset is None:
        raise ValueError(f"Unknown timeframe: {timeframe}")
    
    # Resample
    resampled = df.resample(offset).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    # Drop incomplete bars
    resampled.dropna(inplace=True)
    
    return resampled


def create_mtf_data(m1_df: pd.DataFrame, 
                    timeframes: List[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Create multi-timeframe data from M1 data.
    
    Args:
        m1_df: M1 OHLCV dataframe
        timeframes: List of timeframes to create
        
    Returns:
        Dictionary mapping timeframe to dataframe
    """
    if timeframes is None:
        timeframes = MTF_CONFIG['timeframes']
    
    mtf_data = {}
    
    for tf in timeframes:
        if tf == "M1":
            mtf_data[tf] = m1_df.copy()
        else:
            print(f"  Resampling to {tf}...")
            mtf_data[tf] = resample_ohlcv(m1_df, tf)
    
    return mtf_data


def load_mtf_data(data_dir: str = None,
                  train_start: str = None,
                  train_end: str = None,
                  test_start: str = None,
                  test_end: str = None,
                  timeframes: List[str] = None) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Load and prepare multi-timeframe data.
    
    This is the main entry point for data loading.
    
    Args:
        data_dir: Directory containing M1 CSV files
        train_start/end: Training period
        test_start/end: Testing period
        timeframes: List of timeframes to create
        
    Returns:
        train_mtf, test_mtf: Dictionaries mapping timeframe to dataframe
    """
    # Use defaults from config
    if train_start is None:
        train_start = DATA_CONFIG['train_start']
    if train_end is None:
        train_end = DATA_CONFIG['train_end']
    if test_start is None:
        test_start = DATA_CONFIG['test_start']
    if test_end is None:
        test_end = DATA_CONFIG['test_end']
    if timeframes is None:
        timeframes = MTF_CONFIG['timeframes']
    
    # Merge M1 data
    train_m1, test_m1 = merge_m1_data(
        data_dir=data_dir,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end
    )
    
    # Create multi-timeframe data
    print("\n[Data] Creating multi-timeframe data...")
    print("  Training data:")
    train_mtf = create_mtf_data(train_m1, timeframes)
    
    print("  Test data:")
    test_mtf = create_mtf_data(test_m1, timeframes)
    
    return train_mtf, test_mtf


# ============================================================
# Legacy functions (for backward compatibility)
# ============================================================

def load_csv_data(file_path: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Load data from CSV file (MT5 format) - Legacy function
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path, sep='\t')
    except:
        try:
            df = pd.read_csv(file_path, sep=',')
        except Exception as e:
            raise ValueError(f"Could not read CSV file: {e}")
    
    # Clean column names
    df.columns = df.columns.str.replace('<', '').str.replace('>', '').str.strip()
    
    # Rename columns
    rename_map = {}
    for old, new in [('open', 'open'), ('high', 'high'), ('low', 'low'), 
                     ('close', 'close'), ('tickvol', 'volume'), ('vol', 'volume')]:
        if old in df.columns:
            rename_map[old] = new
    
    df.rename(columns=rename_map, inplace=True)
    
    # Parse datetime
    if 'date' in df.columns and 'time' in df.columns:
        df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str))
        df.set_index('datetime', inplace=True)
    elif 'time' in df.columns:
        df.index = pd.to_datetime(df['time'])
    elif 'date' in df.columns:
        df.index = pd.to_datetime(df['date'])
    
    # Keep only OHLCV
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    df = df[[c for c in required_cols if c in df.columns]].copy()
    
    df.sort_index(inplace=True)
    
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]
    
    df.dropna(inplace=True)
    
    return df


def load_mt5_data(symbol: str = None, timeframe: str = "M15", 
                   start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Load data from MT5 terminal - Legacy function
    """
    try:
        import MetaTrader5 as mt5
        
        if not mt5.initialize():
            raise RuntimeError("MT5 not initialized")
        
        timeframe_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }
        
        tf = timeframe_map.get(timeframe, mt5.TIMEFRAME_M15)
        
        if start_date:
            start = int(pd.to_datetime(start_date).timestamp())
        else:
            start = 0
            
        if end_date:
            end = int(pd.to_datetime(end_date).timestamp())
        else:
            end = int((datetime.now() + pd.Timedelta(days=365*10)).timestamp())
        
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, 100000)
        
        if rates is None or len(rates) == 0:
            rates = mt5.copy_rates_range(symbol, tf, start, end)
        
        if rates is None or len(rates) == 0:
            raise ValueError(f"No data for {symbol}")
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        df.rename(columns={
            'tick_volume': 'volume'
        }, inplace=True)
        
        mt5.shutdown()
        
        return df[['open', 'high', 'low', 'close', 'volume']]
    
    except ImportError:
        raise ImportError("MetaTrader5 package not installed")


def get_latest_data(df: pd.DataFrame, n_bars: int = 100) -> pd.DataFrame:
    """Get the latest N bars from dataframe"""
    return df.tail(n_bars).copy()


# ============================================================
# Data info utilities
# ============================================================

def get_data_info(df: pd.DataFrame) -> dict:
    """Get summary information about a dataframe"""
    return {
        'shape': df.shape,
        'start': df.index.min(),
        'end': df.index.max(),
        'columns': df.columns.tolist(),
        'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
    }


def print_data_summary(train_mtf: Dict[str, pd.DataFrame], 
                       test_mtf: Dict[str, pd.DataFrame]):
    """Print summary of multi-timeframe data"""
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    
    print("\nTraining Data:")
    for tf, df in train_mtf.items():
        print(f"  {tf}: {len(df):,} bars | {df.index.min()} to {df.index.max()}")
    
    print("\nTest Data:")
    for tf, df in test_mtf.items():
        print(f"  {tf}: {len(df):,} bars | {df.index.min()} to {df.index.max()}")
    
    print("=" * 60)