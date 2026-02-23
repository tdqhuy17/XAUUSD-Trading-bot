"""
Evaluation Script for SAC Agent
Usage: python scripts/evaluate.py

Features:
- Backtest with detailed metrics
- Equity curve visualization
- Trade analysis with best/worst highlights
- Performance breakdown by position type
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

from trading_bot.config import Config, SAC_CONFIG, ENV_CONFIG
from trading_bot.data_loader import load_csv_data, load_mt5_data, prepare_data
from trading_bot.model import load_model, backtest
from trading_bot.features import get_feature_columns


def plot_enhanced_analysis(equity_curve, trades, df_test, save_path=None):
    """Plot comprehensive analysis with best/worst trade markers"""
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Equity Curve with Trade Markers
    ax1 = plt.subplot(3, 2, (1, 2))
    ax1.plot(equity_curve, label='Portfolio Value', color='blue', linewidth=1.5, alpha=0.8)
    ax1.axhline(y=equity_curve[0], color='gray', linestyle='--', alpha=0.5, label='Initial')
    
    # Find best and worst trades
    if trades:
        pnls = [(i, t) for i, t in enumerate(trades) if t.get('pnl', 0) != 0]
        if pnls:
            best_idx, best_trade = max(pnls, key=lambda x: x[1]['pnl'])
            worst_idx, worst_trade = min(pnls, key=lambda x: x[1]['pnl'])
            
            # Mark best trade
            if 'step' in best_trade:
                ax1.axvline(x=best_trade['step'], color='green', linestyle='-', alpha=0.7, linewidth=2)
                ax1.annotate(f"Best: +${best_trade['pnl']:.2f}", 
                           xy=(best_trade['step'], equity_curve[min(best_trade['step'], len(equity_curve)-1)]),
                           xytext=(10, 20), textcoords='offset points',
                           fontsize=9, color='green', fontweight='bold')
            
            # Mark worst trade
            if 'step' in worst_trade:
                ax1.axvline(x=worst_trade['step'], color='red', linestyle='-', alpha=0.7, linewidth=2)
                ax1.annotate(f"Worst: ${worst_trade['pnl']:.2f}", 
                           xy=(worst_trade['step'], equity_curve[min(worst_trade['step'], len(equity_curve)-1)]),
                           xytext=(10, -20), textcoords='offset points',
                           fontsize=9, color='red', fontweight='bold')
    
    ax1.set_title('Portfolio Equity Curve with Best/Worst Trades', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Drawdown
    ax2 = plt.subplot(3, 2, 3)
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak * 100
    ax2.fill_between(range(len(drawdown)), 0, -drawdown, color='red', alpha=0.3)
    ax2.plot(-drawdown, color='red', linewidth=1)
    ax2.set_title('Drawdown (%)', fontsize=11)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Trade P&L Distribution
    ax3 = plt.subplot(3, 2, 4)
    if trades:
        pnls = [t['pnl'] for t in trades if t.get('pnl', 0) != 0]
        if pnls:
            colors = ['green' if p > 0 else 'red' for p in pnls]
            ax3.bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
            ax3.axhline(y=0, color='black', linewidth=0.5)
            ax3.set_title('Trade P&L Distribution', fontsize=11)
            ax3.set_xlabel('Trade #')
            ax3.set_ylabel('P&L ($)')
            ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Long vs Short Performance
    ax4 = plt.subplot(3, 2, 5)
    if trades:
        long_pnl = sum(t['pnl'] for t in trades if t['action'] in ['Buy', 'Cover Short'] and t.get('pnl'))
        short_pnl = sum(t['pnl'] for t in trades if t['action'] in ['Sell', 'Short'] and t.get('pnl'))
        
        categories = ['Long', 'Short']
        values = [long_pnl, short_pnl]
        colors = ['green' if v >= 0 else 'red' for v in values]
        bars = ax4.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
        ax4.axhline(y=0, color='black', linewidth=0.5)
        ax4.set_title('Long vs Short Performance', fontsize=11)
        ax4.set_ylabel('Total P&L ($)')
        for bar, val in zip(bars, values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'${val:.2f}', 
                    ha='center', va='bottom' if val >= 0 else 'top', fontsize=10)
    
    # 5. Win Rate Over Time
    ax5 = plt.subplot(3, 2, 6)
    if trades:
        cumulative_wins = []
        wins = 0
        total = 0
        for t in trades:
            if t.get('pnl', 0) != 0:
                total += 1
                if t['pnl'] > 0:
                    wins += 1
                cumulative_wins.append(wins / total * 100 if total > 0 else 0)
        
        ax5.plot(cumulative_wins, color='purple', linewidth=1.5)
        ax5.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50%')
        ax5.set_title('Win Rate Over Time (%)', fontsize=11)
        ax5.set_xlabel('Trade #')
        ax5.set_ylabel('Win Rate (%)')
        ax5.set_ylim(0, 100)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Chart saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_detailed_analysis(trades, equity_curve):
    """Print comprehensive trade analysis"""
    print("\n" + "=" * 60)
    print("   DETAILED TRADE ANALYSIS")
    print("=" * 60)
    
    if not trades:
        print("\n[No trades executed]")
        return
    
    pnls = [t['pnl'] for t in trades if t.get('pnl', 0) != 0]
    winning = [p for p in pnls if p > 0]
    losing = [p for p in pnls if p < 0]
    
    # Trade Highlights
    print("\n=== Trade Highlights ===")
    if pnls:
        best_trade = max(trades, key=lambda t: t.get('pnl', 0))
        worst_trade = min(trades, key=lambda t: t.get('pnl', 0))
        
        print(f"  Best Trade:  +${best_trade['pnl']:.2f} ({best_trade['action']})")
        print(f"  Worst Trade: ${worst_trade['pnl']:.2f} ({worst_trade['action']})")
        print(f"  Avg Win:     ${np.mean(winning):.2f}" if winning else "  Avg Win:     N/A")
        print(f"  Avg Loss:    ${np.mean(losing):.2f}" if losing else "  Avg Loss:    N/A")
        
        # Consecutive wins/losses
        max_consec_wins = max_consec_losses = curr_wins = curr_losses = 0
        for t in trades:
            if t.get('pnl', 0) > 0:
                curr_wins += 1
                curr_losses = 0
                max_consec_wins = max(max_consec_wins, curr_wins)
            elif t.get('pnl', 0) < 0:
                curr_losses += 1
                curr_wins = 0
                max_consec_losses = max(max_consec_losses, curr_losses)
        
        print(f"  Max Consecutive Wins:   {max_consec_wins}")
        print(f"  Max Consecutive Losses: {max_consec_losses}")
    
    # Position Analysis
    print("\n=== Position Analysis ===")
    long_trades = [t for t in trades if t['action'] in ['Buy', 'Cover Short']]
    short_trades = [t for t in trades if t['action'] in ['Sell', 'Short']]
    
    long_pnl = sum(t['pnl'] for t in long_trades if t.get('pnl'))
    short_pnl = sum(t['pnl'] for t in short_trades if t.get('pnl'))
    
    long_wins = len([t for t in long_trades if t.get('pnl', 0) > 0])
    short_wins = len([t for t in short_trades if t.get('pnl', 0) > 0])
    
    print(f"  Long Trades:  {len(long_trades):3d} trades | P&L: ${long_pnl:+.2f} | Win Rate: {long_wins/len(long_trades)*100 if long_trades else 0:.1f}%")
    print(f"  Short Trades: {len(short_trades):3d} trades | P&L: ${short_pnl:+.2f} | Win Rate: {short_wins/len(short_trades)*100 if short_trades else 0:.1f}%")
    
    # Risk Metrics
    print("\n=== Risk Metrics ===")
    if len(equity_curve) > 1:
        returns = np.diff(equity_curve) / equity_curve[:-1]
        returns = returns[returns != 0]
        
        if len(returns) > 1:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 96) if np.std(returns) > 0 else 0
            downside = returns[returns < 0]
            sortino = np.mean(returns) / np.std(downside) * np.sqrt(252 * 96) if len(downside) > 0 and np.std(downside) > 0 else 0
        else:
            sharpe = sortino = 0
        
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_dd = drawdown.max() * 100
        
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0] * 100
        
        print(f"  Total Return:     {total_return:+.2f}%")
        print(f"  Max Drawdown:     {max_dd:.2f}%")
        print(f"  Sharpe Ratio:     {sharpe:.2f}")
        print(f"  Sortino Ratio:    {sortino:.2f}")
        
        if max_dd > 0:
            recovery = total_return / max_dd
            print(f"  Recovery Factor:  {recovery:.2f}")
        
        if winning and losing:
            profit_factor = sum(winning) / abs(sum(losing))
            print(f"  Profit Factor:    {profit_factor:.2f}")
            print(f"  Risk-Reward:      {np.mean(winning)/abs(np.mean(losing)):.2f}" if np.mean(losing) != 0 else "  Risk-Reward:      N/A")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAC trading agent")
    parser.add_argument('--model', type=str, default='sac_gold_agent', help='Model name')
    parser.add_argument('--data', type=str, default='XAUUSD_M15.csv', help='Data file')
    parser.add_argument('--visualize', action='store_true', help='Show plots')
    parser.add_argument('--initial-cash', type=float, default=10000, help='Initial capital')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("   SAC Agent Evaluation")
    print("=" * 60)
    
    # Find model
    model_path = Config.MODELS_DIR / args.model
    if not model_path.exists():
        model_path_pt = str(model_path) + '.pt'
        if os.path.exists(model_path_pt):
            model_path = model_path_pt
        else:
            print(f"\n[ERROR] Model not found: {model_path}")
            return
    
    print(f"\n[Model] Path: {model_path}")
    
    # Load data
    data_path = Config.DATA_DIR / args.data
    if not data_path.exists():
        print(f"[ERROR] Data not found: {data_path}")
        return
    
    df = load_csv_data(str(data_path))
    print(f"[Data] Loaded: {len(df):,} bars")
    
    # Prepare
    df = prepare_data(df, add_indicators=True, add_divergence=True, normalize=True)
    test_df = df.iloc[int(len(df) * 0.8):].reset_index(drop=True)
    print(f"[Test] {len(test_df):,} bars")
    
    # Load and evaluate
    agent = load_model(str(model_path), df=test_df)
    results = backtest(agent, test_df, args.initial_cash, verbose=1)
    
    # Detailed analysis
    print_detailed_analysis(results['trades'], results['equity_curve'])
    
    # Visualize
    if args.visualize:
        chart_path = Config.LOGS_DIR / 'evaluation_analysis.png'
        plot_enhanced_analysis(results['equity_curve'], results['trades'], test_df, chart_path)
    
    # Save trades
    if results['trades']:
        trades_df = pd.DataFrame(results['trades'])
        trades_file = Config.LOGS_DIR / 'trade_log.csv'
        trades_df.to_csv(trades_file, index=False)
        print(f"\n[Saved] Trade log: {trades_file}")
    
    print("\n" + "=" * 60)
    print("   Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()