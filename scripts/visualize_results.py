"""
Visualization Script for Evaluation Results
Usage: python scripts/visualize_results.py

Generates charts from evaluation_results.csv:
- Price chart with position markers
- Equity curve
- Position over time
- Drawdown chart
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_results(file_path: str = None) -> pd.DataFrame:
    """Load evaluation results from CSV"""
    if file_path is None:
        file_path = Path(__file__).parent.parent / 'logs' / 'evaluation_results.csv'
    
    df = pd.read_csv(file_path)
    return df


def calculate_metrics(df: pd.DataFrame) -> dict:
    """Calculate trading metrics"""
    initial_cash = 10000  # Default initial cash
    
    # Final values
    final_assets = df['assets'].iloc[-1]
    total_return = (final_assets - initial_cash) / initial_cash * 100
    
    # Drawdown
    peak = df['assets'].expanding().max()
    drawdown = (df['assets'] - peak) / peak * 100
    max_drawdown = drawdown.min()
    
    # Trade stats
    total_trades = df['trade_count'].iloc[-1]
    
    # Position changes (count actual trades)
    position_changes = df['position'].diff().abs()
    num_trades = (position_changes > 0.01).sum()
    
    # Max position
    max_long = df['position'].max()
    max_short = df['position'].min()
    
    return {
        'initial_cash': initial_cash,
        'final_assets': final_assets,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'total_trades': total_trades,
        'num_trades': num_trades,
        'max_long': max_long,
        'max_short': max_short,
    }


def plot_results(df: pd.DataFrame, save_path: str = None):
    """Generate visualization plots"""
    # Calculate metrics
    metrics = calculate_metrics(df)
    
    # Calculate drawdown
    peak = df['assets'].expanding().max()
    drawdown = (df['assets'] - peak) / peak * 100
    
    # Create figure with subplots
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('Trading Bot Evaluation Results', fontsize=14, fontweight='bold')
    
    # Plot 1: Price
    ax1 = axes[0]
    ax1.plot(df.index, df['price'], 'b-', linewidth=0.8, label='Price')
    
    # Mark long positions (green)
    long_mask = df['position'] > 0.01
    ax1.scatter(df.index[long_mask], df['price'][long_mask], 
                c='green', s=5, alpha=0.3, label='Long')
    
    # Mark short positions (red)
    short_mask = df['position'] < -0.01
    ax1.scatter(df.index[short_mask], df['price'][short_mask], 
                c='red', s=5, alpha=0.3, label='Short')
    
    ax1.set_ylabel('Price ($)')
    ax1.legend(loc='upper right')
    ax1.set_title(f'Gold Price | Max Long: {metrics["max_long"]:.2f} | Max Short: {metrics["max_short"]:.2f}')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Equity
    ax2 = axes[1]
    ax2.plot(df.index, df['assets'], 'g-', linewidth=1, label='Assets')
    ax2.axhline(y=metrics['initial_cash'], color='gray', linestyle='--', alpha=0.5, label='Initial')
    ax2.fill_between(df.index, metrics['initial_cash'], df['assets'], 
                     where=df['assets'] >= metrics['initial_cash'], 
                     color='green', alpha=0.3)
    ax2.fill_between(df.index, metrics['initial_cash'], df['assets'], 
                     where=df['assets'] < metrics['initial_cash'], 
                     color='red', alpha=0.3)
    ax2.set_ylabel('Assets ($)')
    ax2.legend(loc='upper right')
    ax2.set_title(f'Equity Curve | Return: {metrics["total_return"]:.2f}%')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Position
    ax3 = axes[2]
    ax3.fill_between(df.index, 0, df['position'], 
                     where=df['position'] > 0, color='green', alpha=0.5, label='Long')
    ax3.fill_between(df.index, 0, df['position'], 
                     where=df['position'] < 0, color='red', alpha=0.5, label='Short')
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.set_ylabel('Position Size')
    ax3.legend(loc='upper right')
    ax3.set_title(f'Position Over Time | Trades: {metrics["total_trades"]}')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Drawdown
    ax4 = axes[3]
    ax4.fill_between(df.index, 0, drawdown, color='red', alpha=0.5)
    ax4.axhline(y=metrics['max_drawdown'], color='darkred', linestyle='--', 
                label=f'Max DD: {metrics["max_drawdown"]:.2f}%')
    ax4.set_ylabel('Drawdown (%)')
    ax4.set_xlabel('Step')
    ax4.legend(loc='lower right')
    ax4.set_title('Drawdown from Peak')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    if save_path is None:
        save_path = Path(__file__).parent.parent / 'logs' / 'evaluation_visualization.png'
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {save_path}")
    
    return fig


def print_summary(metrics: dict):
    """Print summary statistics"""
    print("\n" + "=" * 50)
    print("TRADING SUMMARY")
    print("=" * 50)
    print(f"Initial Cash    : ${metrics['initial_cash']:,.2f}")
    print(f"Final Assets    : ${metrics['final_assets']:,.2f}")
    print(f"Total Return    : {metrics['total_return']:+.2f}%")
    print("-" * 50)
    print(f"Max Drawdown    : {metrics['max_drawdown']:.2f}%")
    print(f"Total Trades    : {metrics['total_trades']}")
    print(f"Max Long Pos    : {metrics['max_long']:.2f}")
    print(f"Max Short Pos   : {metrics['max_short']:.2f}")
    print("=" * 50)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize evaluation results")
    parser.add_argument('--file', type=str, default=None, help='Path to CSV file')
    args = parser.parse_args()
    
    # Load data
    file_path = args.file if args.file else None
    df = load_results(file_path)
    
    print(f"Loaded {len(df)} rows from evaluation results")
    
    # Calculate metrics
    metrics = calculate_metrics(df)
    
    # Print summary
    print_summary(metrics)
    
    # Generate plots
    plot_results(df)
    
    print("\nDone!")


if __name__ == "__main__":
    main()