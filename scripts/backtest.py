"""
Out-of-Distribution Backtest Script
Tests a trained SAC agent on data outside the training period to
measure generalization.

Default: train on 2021-2026, backtest on 2017-2019.

Usage:
    # Use existing CSV (must have date range covering target period)
    python scripts/backtest.py --data XAUUSD_M15.csv --start 2017-01-01 --end 2019-12-31

    # Pull fresh data from MT5 for exact date range
    python scripts/backtest.py --use-mt5 --start 2017-01-01 --end 2019-12-31

    # Specify which saved model to evaluate
    python scripts/backtest.py --model sac_gold_agent.best --start 2017-01-01 --end 2019-12-31
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from trading_bot.config import Config
from trading_bot.data_loader import load_csv_data, load_mt5_data, prepare_data
from trading_bot.model import load_model, backtest
from trading_bot.features import get_feature_columns


# ─── helpers ───────────────────────────────────────────────────────────────────

def _filter_dates(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Slice a DataFrame by date range (inclusive)."""
    if 'time' not in df.columns:
        return df
    ts = pd.to_datetime(df['time'], errors='coerce')
    mask = (ts >= pd.Timestamp(start)) & (ts <= pd.Timestamp(end))
    result = df.loc[mask].reset_index(drop=True)
    if len(result) == 0:
        raise ValueError(
            f"No data found between {start} and {end}.  "
            f"CSV covers {df['time'].iloc[0]} -> {df['time'].iloc[-1]}"
        )
    return result


def _print_regime_summary(df: pd.DataFrame, label: str):
    """Print a short price-regime summary for the test period."""
    prices = df['close']
    print("  Period     :", label)
    print("  Bars       : {:,}".format(len(df)))
    print("  Price open : ${:.2f}".format(prices.iloc[0]))
    print("  Price close: ${:.2f}".format(prices.iloc[-1]))
    print("  Price range: ${:.2f} – ${:.2f}".format(prices.min(), prices.max()))
    change = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0] * 100
    print("  Net move   : {:+.1f}%".format(change))
    if 'atr' in df.columns:
        print("  Avg ATR    : {:.2f}".format(df['atr'].mean()))
    print()


# ─── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Out-of-distribution backtest")
    parser.add_argument('--data',    type=str,  default='XAUUSD_M15.csv',
                        help='CSV file in data/ folder')
    parser.add_argument('--use-mt5', action='store_true',
                        help='Download data directly from MT5')
    parser.add_argument('--start',   type=str,  default='2017-01-01',
                        help='Backtest start date  (YYYY-MM-DD)')
    parser.add_argument('--end',     type=str,  default='2019-12-31',
                        help='Backtest end date    (YYYY-MM-DD)')
    parser.add_argument('--model',   type=str,  default='sac_gold_agent.best',
                        help='Model filename inside models/')
    parser.add_argument('--device',  type=str,  default='auto',
                        choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--save-csv', action='store_true',
                        help='Save equity curve to logs/backtest_results.csv')
    args = parser.parse_args()

    print("=" * 65)
    print("  Out-of-Distribution Backtest")
    print("=" * 65)
    print("  Test period : {} -> {}".format(args.start, args.end))
    print("  Model       : {}".format(args.model))
    print()

    # ── 1. Load price data ──────────────────────────────────────────────────
    print("[1] Loading price data...")

    if args.use_mt5:
        print("  Connecting to MT5...")
        try:
            df_raw = load_mt5_data(
                symbol="XAUUSD",
                timeframe="M15",
                start_date=args.start,
                end_date=args.end,
            )
            print("  Loaded {:,} bars from MT5".format(len(df_raw)))
        except Exception as e:
            print("  ERROR: {}".format(e))
            print("  Make sure MT5 is open and logged in.")
            return
    else:
        data_path = Config.DATA_DIR / args.data
        if not data_path.exists():
            print("  ERROR: {} not found".format(data_path))
            return
        df_raw = load_csv_data(str(data_path))
        print("  Loaded {:,} bars from {}".format(len(df_raw), args.data))

        # Slice to requested date range
        try:
            df_raw = _filter_dates(df_raw, args.start, args.end)
        except ValueError as e:
            print("  ERROR:", e)
            return
        print("  Sliced to {:,} bars for period".format(len(df_raw)))

    if len(df_raw) < 200:
        print("  ERROR: Not enough data ({} bars). Need at least 200.".format(len(df_raw)))
        return

    # ── 2. Prepare features ─────────────────────────────────────────────────
    print("\n[2] Computing indicators...")
    df = prepare_data(df_raw, add_indicators=True, add_divergence=True, normalize=True)
    features = get_feature_columns(df)
    print("  Features: {}  |  Bars after indicator warmup: {:,}".format(
        len(features), len(df)))

    _print_regime_summary(df, "{} -> {}".format(args.start, args.end))

    # ── 3. Load model ───────────────────────────────────────────────────────
    print("[3] Loading model...")
    model_path = str(Config.MODELS_DIR / args.model)
    # strip .pt suffix for load_model (it adds it)
    if model_path.endswith('.pt'):
        model_path_no_ext = model_path[:-3]
    else:
        model_path_no_ext = model_path

    try:
        agent = load_model(model_path_no_ext, device=args.device)
    except FileNotFoundError:
        # try with extension
        try:
            agent = load_model(model_path, device=args.device)
        except FileNotFoundError as e:
            print("  ERROR:", e)
            print("  Available models:")
            for f in Config.MODELS_DIR.iterdir():
                print("    -", f.name)
            return

    # ── 4. Run backtest ─────────────────────────────────────────────────────
    print("\n[4] Running backtest on {} bars...".format(len(df)))
    results = backtest(agent, df, verbose=0)

    # ── 5. Detailed report ──────────────────────────────────────────────────
    eq   = results['equity_curve']
    rets = np.diff(eq) / eq[:-1]

    print("\n" + "=" * 65)
    print("  BACKTEST RESULTS  ({} -> {})".format(args.start, args.end))
    print("=" * 65)
    print("  Initial capital  : ${:>12,.2f}".format(results['initial_cash']))
    print("  Final capital    : ${:>12,.2f}".format(results['final_value']))
    print("  Total return     : {:>+12.2f}%".format(results['total_return_pct']))
    print("-" * 65)
    print("  Sharpe ratio     : {:>12.4f}".format(results['sharpe_ratio']))
    print("  Max drawdown     : {:>12.2f}%".format(results['max_drawdown_pct']))
    print("  Win rate         : {:>12.1f}%".format(results['win_rate_pct']))
    print("  Total trades     : {:>12,}".format(results['total_trades']))
    print("-" * 65)

    # Monthly-ish breakdown (approximate: split equity curve into ~12 chunks)
    if len(eq) > 50:
        n_chunks = min(12, len(eq) // 50)
        chunk_size = len(eq) // n_chunks
        print("  Approximate period breakdown:")
        for i in range(n_chunks):
            s = i * chunk_size
            e = min((i + 1) * chunk_size, len(eq) - 1)
            seg_ret = (eq[e] - eq[s]) / eq[s] * 100
            bar = "#" * int(abs(seg_ret) / 2) if abs(seg_ret) < 40 else "#" * 20
            sign = "+" if seg_ret >= 0 else "-"
            print("    Seg {:>2}/{:>2}  {:>+7.2f}%  {}{}".format(
                i + 1, n_chunks, seg_ret, sign, bar))

    print("=" * 65)

    # ── 6. Comparison note ──────────────────────────────────────────────────
    print("\n  INTERPRETATION:")
    ret = results['total_return_pct']
    sharpe = results['sharpe_ratio']
    if ret > 5 and sharpe > 0.5:
        verdict = "GOOD - Agent generalizes to this regime"
    elif ret > 0:
        verdict = "MARGINAL - Positive but weak out-of-distribution performance"
    elif ret > -10:
        verdict = "POOR - Agent struggles outside training distribution"
    else:
        verdict = "BAD - Agent likely overfit to training regime"
    print("  {}".format(verdict))
    print()

    # ── 7. Optional CSV save ────────────────────────────────────────────────
    if args.save_csv:
        out_path = Config.LOGS_DIR / 'backtest_results.csv'
        trades_df = pd.DataFrame(results['trades']) if results['trades'] else pd.DataFrame()
        trades_df.to_csv(out_path, index=False)
        eq_path = Config.LOGS_DIR / 'backtest_equity.csv'
        pd.DataFrame({'equity': results['equity_curve']}).to_csv(eq_path, index=False)
        print("  Saved to {} and {}".format(out_path, eq_path))


if __name__ == "__main__":
    main()
