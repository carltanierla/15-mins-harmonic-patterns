import ccxt
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import time
import os
import requests
import json

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
TIMEFRAMES = ['30m']
ERR_ALLOWED = 0.07
TOP_N_PAIRS = 200  # Reduced slightly to speed up 30m run
SL_BUFFER   = 0.002

# Get Webhook from GitHub Secrets (Environment Variable)
DISCORD_WEBHOOK_URL = os.environ.get('DISCORD_WEBHOOK_URL')

# -----------------------------------------------------------------------------
# DISCORD ALERT FUNCTION
# -----------------------------------------------------------------------------
def send_discord_alert(symbol, timeframe, pattern, direction, entry, sl, tp1, tp2):
    if not DISCORD_WEBHOOK_URL:
        return

    # Color: Green for Bullish, Red for Bearish
    color = 5763719 if direction == "BULLISH" else 15548997
    
    data = {
        "username": "Harmonic Scanner Bot",
        "embeds": [
            {
                "title": f"🚨 {pattern} Pattern Detected!",
                "description": f"**{symbol}** on the **{timeframe}** timeframe.",
                "color": color,
                "fields": [
                    {"name": "Direction", "value": direction, "inline": True},
                    {"name": "Entry Price", "value": f"${entry:.4f}", "inline": True},
                    {"name": "\u200b", "value": "\u200b", "inline": True}, # Empty spacer
                    {"name": "🛑 Stop Loss", "value": f"${sl:.4f}", "inline": True},
                    {"name": "💰 Target 1", "value": f"${tp1:.4f}", "inline": True},
                    {"name": "💰 Target 2", "value": f"${tp2:.4f}", "inline": True}
                ],
                "footer": {
                    "text": "Market data via MEXC • Automated by GitHub Actions"
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        ]
    }

    try:
        requests.post(DISCORD_WEBHOOK_URL, json=data)
    except Exception as e:
        print(f"Failed to send alert: {e}")

# -----------------------------------------------------------------------------
# (EXISTING LOGIC BELOW - ABBREVIATED FOR CLARITY)
# -----------------------------------------------------------------------------
# ... [Paste the STD_PATTERNS dictionary here] ...
STD_PATTERNS = {
    'Gartley':    {'XB': 0.618, 'AC': [0.382, 0.886], 'BD': [1.272, 1.618], 'XD': 0.786},
    'Bat':        {'XB': [0.382, 0.50], 'AC': [0.382, 0.886], 'BD': [1.618, 2.618], 'XD': 0.886},
    'Butterfly':  {'XB': 0.786, 'AC': [0.382, 0.886], 'BD': [1.618, 2.618], 'XD': [1.27, 1.618]},
    'Crab':       {'XB': [0.382, 0.618], 'AC': [0.382, 0.886], 'BD': [2.24, 3.618], 'XD': 1.618},
}

def get_top_pairs(limit=100):
    exchange = ccxt.mexc()
    try:
        tickers = exchange.fetch_tickers()
        pairs = [{'symbol': s, 'volume': d['quoteVolume']} for s, d in tickers.items() if s.endswith('/USDT')]
        df = pd.DataFrame(pairs).sort_values('volume', ascending=False).head(limit)
        return df['symbol'].tolist()
    except: return []

def get_mexc_data(symbol, timeframe):
    exchange = ccxt.mexc()
    try:
        return pd.DataFrame(exchange.fetch_ohlcv(symbol, timeframe, limit=100), 
                          columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    except: return pd.DataFrame()

def find_peaks(df, order=3):
    max_idx = argrelextrema(df['high'].values, np.greater, order=order)[0]
    min_idx = argrelextrema(df['low'].values, np.less, order=order)[0]
    peaks = [{'idx': i, 'price': df['high'].iloc[i], 'type': 'high'} for i in max_idx] + \
            [{'idx': i, 'price': df['low'].iloc[i], 'type': 'low'} for i in min_idx]
    peaks.sort(key=lambda x: x['idx'])
    return peaks

def check_ratio(actual, target, tol=ERR_ALLOWED):
    if isinstance(target, list): return any(abs(actual - t) <= tol * t for t in target)
    return abs(actual - target) <= tol * target

def identify_patterns(points):
    if len(points) < 5: return []
    X, A, B, C, D = points[-5:]
    px, pa, pb, pc, pd_pt = X['price'], A['price'], B['price'], C['price'], D['price']
    XA, AB, BC, CD, XD = abs(px-pa), abs(pa-pb), abs(pb-pc), abs(pc-pd_pt), abs(px-pd_pt)
    if XA==0 or AB==0 or BC==0: return []
    
    XB_r, AC_r, BD_r, XD_r = AB/XA, BC/AB, CD/BC, XD/XA
    bullish, bearish = (X['type']=='low' and D['type']=='low'), (X['type']=='high' and D['type']=='high')
    if not (bullish or bearish): return []
    
    found = []
    trend = "BULLISH" if bullish else "BEARISH"
    for name, r in STD_PATTERNS.items():
        if check_ratio(XB_r, r['XB']) and check_ratio(AC_r, r['AC']) and check_ratio(BD_r, r['BD']) and check_ratio(XD_r, r['XD']):
            found.append((name, trend))
    return found

def calculate_trade_setup(pattern_name, direction, points):
    X, A, B, C, D = points[-5:]
    pX, pD = X['price'], D['price']
    sl_price = (pD * 0.98) if direction == "BULLISH" else (pD * 1.02) # Simple 2% SL for safety
    cd_len = abs(C['price'] - pD)
    tp1 = pD + (cd_len * 0.382) if direction == "BULLISH" else pD - (cd_len * 0.382)
    tp2 = pD + (cd_len * 0.618) if direction == "BULLISH" else pD - (cd_len * 0.618)
    return sl_price, tp1, tp2

# -----------------------------------------------------------------------------
# MAIN SCANNER LOOP
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    from datetime import datetime
    symbols = get_top_pairs(TOP_N_PAIRS)
    print(f"Scanning {len(symbols)} pairs...")
    
    for symbol in symbols:
        for tf in TIMEFRAMES:
            df = get_mexc_data(symbol, tf)
            if df.empty or len(df) < 50: continue
            
            peaks = find_peaks(df, order=3)
            matches = identify_patterns(peaks)
            
            if matches:
                current_price = df['close'].iloc[-1]
                entry_price = peaks[-1]['price']
                
                # Check if price is near entry (3% tolerance)
                if abs(current_price - entry_price) / entry_price < 0.03:
                    for name, direction in matches:
                        sl, tp1, tp2 = calculate_trade_setup(name, direction, peaks)
                        
                        # PRINT TO LOGS
                        print(f"MATCH: {symbol} {tf} - {name}")
                        
                        # SEND TO DISCORD
                        send_discord_alert(symbol, tf, name, direction, entry_price, sl, tp1, tp2)
                        
            time.sleep(0.1) # Rate limit
