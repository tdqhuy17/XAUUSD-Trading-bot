"""Standalone test - no heavyweight imports (torch etc.)"""
import sys
import os
import glob
import pandas as pd
print("Imports OK", flush=True)

# Test _detect_csv_format logic inline
def detect(fp):
    with open(fp, 'r') as f:
        first = f.readline().strip()
    if ';' in first:
        return 'ascii'
    parts = first.split(',')
    if parts[0][0].isdigit():
        return 'mt'
    return 'header'

data_dir = 'data'

# Test format detection
files = sorted(glob.glob(os.path.join(data_dir, 'DAT_*XAUUSD_M1_*.csv')))
print(f"Found {len(files)} files", flush=True)
for fp in files:
    fmt = detect(fp)
    print(f"  {os.path.basename(fp)}: {fmt}", flush=True)

# Test loading ASCII file
print("\nLoading ASCII file...", flush=True)
fp = os.path.join(data_dir, 'DAT_ASCII_XAUUSD_M1_2016.csv')
df = pd.read_csv(fp, sep=';', header=None,
                 names=['datetime_str', 'open', 'high', 'low', 'close', 'volume'])
df.index = pd.to_datetime(df['datetime_str'], format='%Y%m%d %H%M%S')
df.drop(columns=['datetime_str'], inplace=True)
print(f"  Rows: {len(df):,}", flush=True)
print(f"  Range: {df.index[0]} to {df.index[-1]}", flush=True)
print(df.head(2), flush=True)

# Test loading MT file
print("\nLoading MT file...", flush=True)
fp2 = os.path.join(data_dir, 'DAT_MT_XAUUSD_M1_2025.csv')
df2 = pd.read_csv(fp2, sep=',', header=None,
                  names=['date_str', 'time_str', 'open', 'high', 'low', 'close', 'volume'])
df2.index = pd.to_datetime(
    df2['date_str'].str.replace('.', '-', regex=False) + ' ' + df2['time_str'])
df2.drop(columns=['date_str', 'time_str'], inplace=True)
print(f"  Rows: {len(df2):,}", flush=True)
print(f"  Range: {df2.index[0]} to {df2.index[-1]}", flush=True)
print(df2.head(2), flush=True)

# Merge all
print("\nMerging all files...", flush=True)
frames = []
for fp in files:
    fmt = detect(fp)
    if fmt == 'ascii':
        d = pd.read_csv(fp, sep=';', header=None,
                       names=['datetime_str','open','high','low','close','volume'])
        d.index = pd.to_datetime(d['datetime_str'], format='%Y%m%d %H%M%S')
        d.drop(columns=['datetime_str'], inplace=True)
    elif fmt == 'mt':
        d = pd.read_csv(fp, sep=',', header=None,
                       names=['date_str','time_str','open','high','low','close','volume'])
        d.index = pd.to_datetime(d['date_str'].str.replace('.', '-', regex=False) + ' ' + d['time_str'])
        d.drop(columns=['date_str', 'time_str'], inplace=True)
    d.dropna(inplace=True)
    print(f"  {os.path.basename(fp)}: {len(d):>10,} bars", flush=True)
    frames.append(d)

merged = pd.concat(frames, axis=0)
merged.sort_index(inplace=True)
before = len(merged)
merged = merged[~merged.index.duplicated(keep='first')]
dupes = before - len(merged)

print(f"\nTotal: {len(merged):,} bars ({dupes:,} dupes removed)", flush=True)
print(f"Range: {merged.index[0]} to {merged.index[-1]}", flush=True)
print(f"Columns: {list(merged.columns)}", flush=True)
print(f"NaN: {merged.isna().sum().sum()}", flush=True)
print("\nSUCCESS!", flush=True)
