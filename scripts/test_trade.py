"""
MT5 Trade Test Script
Tests a simple BUY then SELL to verify MT5 connection and order execution
Usage: python scripts/test_trade.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_bot.trade_executor import MT5Trader
from trading_bot.config import TRADE_CONFIG


def test_mt5_trade():
    """Test MT5 connection and execute a test trade"""
    print("=" * 50)
    print("MT5 Trade Test Script")
    print("=" * 50)
    
    # Create trader
    trader = MT5Trader(
        symbol="XAUUSD",
        lot_size=0.01,  # Minimum lot size
        magic_number=999999,  # Unique magic number for test
        deviation=20,
    )
    
    # Initialize
    print("\n[1] Initializing MT5...")
    if not trader.initialize():
        print("FAILED: Could not initialize MT5")
        return False
    
    print("SUCCESS: MT5 initialized")
    
    # Check for existing positions
    print("\n[2] Checking existing positions...")
    if trader.has_position():
        print("WARNING: Already in a position, closing first...")
        import MetaTrader5 as mt5
        tick = mt5.symbol_info_tick("XAUUSD")
        if tick:
            result = trader.execute("CLOSE", tick.bid)
            print(f"Close result: {result}")
            time.sleep(1)
    else:
        print("No existing positions")
    
    # Get current price
    print("\n[3] Getting current price...")
    try:
        import MetaTrader5 as mt5
        tick = mt5.symbol_info_tick("XAUUSD")
        if tick is None:
            print("FAILED: Could not get tick data")
            return False
        
        print(f"Current price: Bid={tick.bid}, Ask={tick.ask}")
        current_price = tick.ask
    except Exception as e:
        print(f"FAILED: {e}")
        return False
    
    # Execute BUY
    print("\n[4] Executing BUY order (0.01 lot)...")
    result = trader.execute("BUY", current_price)
    print(f"Result: {result}")
    
    if not result['success']:
        print("FAILED: Buy order failed")
        return False
    
    print("SUCCESS: Buy order executed!")
    
    # Wait a moment
    print("\n[5] Waiting 2 seconds...")
    time.sleep(2)
    
    # Get updated price and close
    print("\n[6] Executing SELL to close position...")
    tick = mt5.symbol_info_tick("XAUUSD")
    if tick is None:
        print("FAILED: Could not get tick for close")
        return False
    
    result = trader.execute("SELL", tick.bid)
    print(f"Result: {result}")
    
    if result['success']:
        print("\n" + "=" * 50)
        print("TEST PASSED: Successfully opened and closed a trade!")
        print("=" * 50)
        return True
    else:
        print("\nFAILED: Could not close position")
        return False


if __name__ == "__main__":
    try:
        success = test_mt5_trade()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)