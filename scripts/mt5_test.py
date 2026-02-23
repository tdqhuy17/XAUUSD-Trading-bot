"""
MT5 Connection Test Script
Tests connection to MetaTrader 5 and displays account info
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_mt5_connection():
    """Test MT5 connection and display account info"""
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("[ERROR] MetaTrader5 package not installed")
        print("Install with: pip install MetaTrader5")
        return False
    
    print("=" * 50)
    print("MT5 Connection Test")
    print("=" * 50)
    
    # Initialize MT5
    if not mt5.initialize():
        print(f"[ERROR] MT5 initialization failed: {mt5.last_error()}")
        print("\nMake sure:")
        print("  1. MetaTrader 5 terminal is running")
        print("  2. You are logged into your account")
        return False
    
    print("[OK] MT5 initialized successfully")
    
    # Get account info
    account = mt5.account_info()
    if account is None:
        print("[ERROR] No account connected")
        mt5.shutdown()
        return False
    
    print("\n" + "=" * 50)
    print("ACCOUNT INFO")
    print("=" * 50)
    print(f"  Account     : {account.login}")
    print(f"  Name        : {account.name}")
    print(f"  Server      : {account.server}")
    print(f"  Currency    : {account.currency}")
    print(f"  Leverage    : 1:{account.leverage}")
    print("-" * 50)
    print(f"  Balance     : ${account.balance:,.2f}")
    print(f"  Equity      : ${account.equity:,.2f}")
    print(f"  Margin      : ${account.margin:,.2f}")
    print(f"  Free Margin : ${account.margin_free:,.2f}")
    print(f"  Margin Lvl  : {account.margin_level:.2f}%")
    print(f"  Profit      : ${account.profit:,.2f}")
    print("=" * 50)
    
    # Test symbol
    symbol = "XAUUSD"
    print(f"\n[TEST] Checking symbol: {symbol}")
    
    if not mt5.symbol_select(symbol, True):
        print(f"[ERROR] Failed to select symbol: {symbol}")
        mt5.shutdown()
        return False
    
    print(f"[OK] Symbol {symbol} selected")
    
    # Get tick
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"[ERROR] Failed to get tick for {symbol}")
        mt5.shutdown()
        return False
    
    print(f"[OK] Current prices:")
    print(f"      Bid: {tick.bid:.2f}")
    print(f"      Ask: {tick.ask:.2f}")
    print(f"      Spread: {(tick.ask - tick.bid):.2f}")
    
    # Get historical data
    print(f"\n[TEST] Fetching M15 data...")
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 100)
    
    if rates is None or len(rates) == 0:
        print(f"[ERROR] Failed to get historical data")
        mt5.shutdown()
        return False
    
    print(f"[OK] Fetched {len(rates)} bars")
    print(f"      Latest: {rates[-1]['close']:.2f} at {rates[-1]['time']}")
    
    # Check for open positions
    print(f"\n[TEST] Checking positions...")
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        print(f"[OK] No positions for {symbol}")
    else:
        print(f"[OK] {len(positions)} open positions for {symbol}")
        for pos in positions:
            print(f"      Ticket: {pos.ticket}, Type: {'BUY' if pos.type == 0 else 'SELL'}, "
                  f"Vol: {pos.volume}, PnL: ${pos.profit:.2f}")
    
    # Shutdown
    mt5.shutdown()
    print("\n" + "=" * 50)
    print("[SUCCESS] All tests passed!")
    print("=" * 50)
    return True


def test_mt5_trade_operations():
    """Test MT5 trade operations (no actual trades)"""
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("[ERROR] MetaTrader5 package not installed")
        return False
    
    print("\n" + "=" * 50)
    print("MT5 Trade Operations Test")
    print("=" * 50)
    
    if not mt5.initialize():
        print(f"[ERROR] MT5 initialization failed")
        return False
    
    symbol = "XAUUSD"
    
    # Check symbol specification
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"[ERROR] Cannot get symbol info for {symbol}")
        mt5.shutdown()
        return False
    
    print(f"\nSymbol: {symbol}")
    print(f"  Digits: {symbol_info.digits}")
    print(f"  Point: {symbol_info.point}")
    print(f"  Min Lot: {symbol_info.volume_min}")
    print(f"  Max Lot: {symbol_info.volume_max}")
    print(f"  Lot Step: {symbol_info.volume_step}")
    print(f"  Contract Size: {symbol_info.trade_contract_size}")
    
    # Check filling modes
    print(f"\nFilling Modes:")
    print(f"  ORDER_FILLING_FOK: {bool(symbol_info.filling_mode & mt5.ORDER_FILLING_FOK)}")
    print(f"  ORDER_FILLING_IOC: {bool(symbol_info.filling_mode & mt5.ORDER_FILLING_IOC)}")
    print(f"  ORDER_FILLING_RETURN: {bool(symbol_info.filling_mode & mt5.ORDER_FILLING_RETURN)}")
    
    mt5.shutdown()
    print("\n[OK] Trade operations test complete")
    return True


if __name__ == "__main__":
    success = test_mt5_connection()
    if success:
        test_mt5_trade_operations()