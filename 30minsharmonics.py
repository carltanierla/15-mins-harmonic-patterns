import ccxt
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from discord_webhook import DiscordWebhook, DiscordEmbed
import os
import io
import time
import sys

# ==========================================
# CONFIGURATION
# ==========================================
# Use Top 50 to keep run-time under control for GitHub Actions
PAIRS_TO_SCAN = 100 
TIMEFRAMES = ['30m'] 
ERR_TOLERANCE = 0.10
WEBHOOK_URL = os.environ.get('DISCORD_WEBHOOK_URL_15M')

if not WEBHOOK_URL:
    print("❌ Error: DISCORD_WEBHOOK_URL_15M not found in environment variables.")
    sys.exit(1)

# ==========================================
# PATTERN RULES
# ==========================================
PATTERNS = {
    'Gartley':  {'XB': 0.618, 'AC': [0.382, 0.886], 'BD': [1.272, 1.618], 'XD': 0.786},
    'Bat':      {'XB': [0.382, 0.50], 'AC': [0.382, 0.886], 'BD': [1.618, 2.618], 'XD': 0.886},
    'Butterfly':{'XB': 0.786, 'AC': [0.382, 0.886], 'BD': [1.618, 2.618], 'XD': [1.27, 1.618]},
    'Crab':     {'XB': [0.382, 0.618], 'AC': [0.382, 0.886], 'BD': [2.618, 3.618], 'XD': 1.618},
    'Cypher':   {'XB': [0.382, 0.618], 'AC': [1.13, 1.414], 'BD': [1.272, 1.618], 'XC_D': 0.786},
}

def get_top_pairs():
    exchange = ccxt.mexc()
    try:
        tickers = exchange.fetch_tickers()
        # Filter for USDT pairs and sort by volume
        usdt_pairs = [
            {'symbol': symbol, 'volume': data['quoteVolume']} 
            for symbol, data in tickers.items() 
            if '/USDT' in symbol and data['quoteVolume'] is not None
        ]
        sorted_pairs = sorted(usdt_pairs, key=lambda x: x['volume'], reverse=True)
        return [p['symbol'] for p in sorted_pairs[:PAIRS_TO_SCAN]]
    except Exception as e:
        print(f"Error fetching tickers: {e}")
        return []

def get_data(symbol, timeframe):
    exchange = ccxt.mexc()
    try:
        # Fetch slightly more candles to ensure we catch the pattern
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except:
        return None

def find_pivots(df, order=4):
    high_idx = argrelextrema(df['high'].values, np.greater, order=order)[0]
    low_idx = argrelextrema(df['low'].values, np.less, order=order)[0]
    pivots = []
    for i in high_idx: pivots.append({'idx': i, 'price': df['high'].iloc[i], 'type': 'high', 'time': df['timestamp'].iloc[i]})
    for i in low_idx: pivots.append({'idx': i, 'price': df['low'].iloc[i], 'type': 'low', 'time': df['timestamp'].iloc[i]})
    pivots.sort(key=lambda x: x['idx'])
    
    clean = []
    if pivots:
        last = pivots[0]
        clean.append(last)
        for curr in pivots[1:]:
            if curr['type'] != last['type']:
                clean.append(curr)
                last = curr
            elif (curr['type'] == 'high' and curr['price'] > last['price']) or \
                 (curr['type'] == 'low' and curr['price'] < last['price']):
                clean[-1] = curr
                last = curr
    return clean

def check_ratio(val, target, tol=0.1):
    if isinstance(target, list):
        if len(target) == 2 and target[0] < target[1]: 
            return target[0]*(1-tol) <= val <= target[1]*(1+tol)
        else: 
            return any(abs(val - t) <= t*tol for t in target)
    return abs(val - target) <= target*tol

def send_discord_alert(symbol, timeframe, pattern, direction, entry_price, pivots, df):
    webhook = DiscordWebhook(url=WEBHOOK_URL)
    
    # 1. Create the Embed
    color = "00ff00" if direction == "BULLISH" else "ff0000" # Green or Red
    embed = DiscordEmbed(
        title=f"Harmonic Pattern Detected: {symbol}",
        description=f"**{pattern}** ({direction}) on **{timeframe}** timeframe.",
        color=color
    )
    embed.add_embed_field(name="Entry Price (D)", value=str(entry_price), inline=True)
    embed.add_embed_field(name="Time", value=str(pivots[-1]['time']), inline=True)
    embed.set_footer(text="MEXC Harmonic Screener")
    webhook.add_embed(embed)

    # 2. Generate Chart Image in Memory
    plt.figure(figsize=(10, 5))
    plt.plot(df['timestamp'], df['close'], color='gray', alpha=0.5)
    
    X, A, B, C, D = pivots[-5:]
    x_coords = [p['time'] for p in [X, A, B, C, D]]
    y_coords = [p['price'] for p in [X, A, B, C, D]]
    
    line_color = 'green' if direction == 'BULLISH' else 'red'
    plt.plot(x_coords, y_coords, color=line_color, linewidth=2, marker='o')
    plt.fill(x_coords, y_coords, color=line_color, alpha=0.1)
    
    for p, label in zip([X, A, B, C, D], ['X', 'A', 'B', 'C', 'D']):
        plt.text(p['time'], p['price'], f" {label}", fontweight='bold')
    
    plt.title(f"{pattern} ({direction}) - {symbol}")
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # 3. Attach Image to Webhook
    webhook.add_file(file=buf.read(), filename=f'{symbol}_{pattern}.png')
    
    response = webhook.execute()
    plt.close() # Close memory to prevent leaks

def run_screener():
    print("Starting Scan...")
    pairs = get_top_pairs()
    print(f"Scanning Top {len(pairs)} pairs...")
    
    for symbol in pairs:
        for tf in TIMEFRAMES:
            time.sleep(0.1) # Respect API limits
            df = get_data(symbol, tf)
            if df is None or len(df) < 20: continue
            
            pivots = find_pivots(df)
            if len(pivots) < 5: continue
            
            pts = pivots[-5:]
            X, A, B, C, D = pts
            
            XA = abs(A['price'] - X['price'])
            AB = abs(B['price'] - A['price'])
            BC = abs(C['price'] - B['price'])
            CD = abs(D['price'] - C['price'])
            XC = abs(C['price'] - X['price'])
            
            if XA == 0 or AB == 0 or BC == 0: continue
            
            XB_r = AB / XA
            AC_r = BC / AB
            BD_r = CD / BC
            XD_r = abs(D['price'] - A['price']) / XA
            XC_D_r = CD / XC 
            
            for name, rule in PATTERNS.items():
                if not check_ratio(XB_r, rule['XB'], ERR_TOLERANCE): continue
                if not check_ratio(AC_r, rule['AC'], ERR_TOLERANCE): continue
                if not check_ratio(BD_r, rule['BD'], ERR_TOLERANCE): continue
                
                match = False
                if name == 'Cypher':
                     if check_ratio(XC_D_r, rule['XC_D'], ERR_TOLERANCE): match = True
                else:
                     if check_ratio(XD_r, rule['XD'], ERR_TOLERANCE): match = True
                
                if match:
                    # Check if pattern is fresh (D point is recent)
                    # For simplicity, we alert if it exists. 
                    # You can add logic here to only alert if D.time is within last 2 candles
                    direction = "BULLISH" if D['type'] == 'low' else "BEARISH"
                    print(f"✅ Found {name} on {symbol}")
                    send_discord_alert(symbol, tf, name, direction, D['price'], pivots, df)

if __name__ == "__main__":
    run_screener()
