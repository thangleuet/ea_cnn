import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import timedelta
import mplfinance as mpf

def calculate_zigzag(df, depth=12, deviation=5, backstep=2):
    high = df['high']
    low = df['low']
    n = len(high)
    zz_price = np.full(n, np.nan)
    direction = np.zeros(n, dtype=int)
    label = [None] * n

    last_high = high[0]
    last_low = low[0]
    last_pivot_index = 0
    last_pivot_type = 0  # 1 for high, -1 for low

    for i in range(1, n):
        if (high[i] - last_low) >= deviation and (i - last_pivot_index) >= depth:
            zz_price[i] = high[i]
            direction[i] = 1
            label[i] = "HH" if last_pivot_type == 1 and high[i] > last_high else "LH"
            last_high = high[i]
            last_pivot_index = i
            last_pivot_type = 1
        elif (last_high - low[i]) >= deviation and (i - last_pivot_index) >= depth:
            zz_price[i] = low[i]
            direction[i] = -1
            label[i] = "LL" if last_pivot_type == -1 and low[i] < last_low else "HL"
            last_low = low[i]
            last_pivot_index = i
            last_pivot_type = -1

    for i in range(n - backstep - 1, -1, -1):
        if not np.isnan(zz_price[i]) and not np.isnan(zz_price[i + backstep]):
            zz_price[i + backstep] = np.nan
            direction[i + backstep] = 0
            label[i + backstep] = None

    result = pd.DataFrame({'zz_price': zz_price, 'direction': direction, 'label': label}, index=df.index)
    return pd.concat([df, result], axis=1)

def plot_zigzag_segment(df, start, end, folder='zigzag'):
    segment = df.loc[start:end].copy()  # Create a copy to avoid modifying original
    
    # Calculate EMAs
    segment['EMA34'] = segment['close'].ewm(span=34, adjust=False).mean()
    segment['EMA89'] = segment['close'].ewm(span=89, adjust=False).mean()
    
    # Ensure data is properly formatted for mplfinance
    ohlc_data = segment[['open', 'high', 'low', 'close']].dropna()
    
    # Prepare EMA plots
    ema34 = mpf.make_addplot(segment['EMA34'].reindex(ohlc_data.index), 
                           color='orange', 
                           width=1.5, 
                           label='EMA34')
    ema89 = mpf.make_addplot(segment['EMA89'].reindex(ohlc_data.index), 
                           color='purple', 
                           width=1.5, 
                           label='EMA89')
    
    # Prepare ZigZag points
    zz_points = segment.dropna(subset=['zz_price'])
    if not zz_points.empty:
        zz_high = zz_points[zz_points['direction'] == 1]
        zz_low = zz_points[zz_points['direction'] == -1]
        
        zz_high_plot = mpf.make_addplot(zz_high['zz_price'],
                                      type='scatter',
                                      markersize=50,
                                      marker='^',
                                      color='green',
                                      label='ZigZag High')
        zz_low_plot = mpf.make_addplot(zz_low['zz_price'],
                                     type='scatter',
                                     markersize=50,
                                     marker='v',
                                     color='red',
                                     label='ZigZag Low')
        
        add_plots = [ema34, ema89, zz_high_plot, zz_low_plot]
    else:
        add_plots = [ema34, ema89]

    # Plot using mplfinance
    mpf.plot(ohlc_data,
            type='candle',
            style='yahoo',
            addplot=add_plots,
            title=f'ZigZag with EMA34 & EMA89 from {start} to {end}',
            ylabel='Price',
            figsize=(14, 7),
            savefig=dict(fname=os.path.join(folder, 
                                          f"zigzag_{start.strftime('%Y%m%d_%H%M')}_{end.strftime('%Y%m%d_%H%M')}.png"),
                        dpi=100,
                        bbox_inches='tight'))
    
    os.makedirs(folder, exist_ok=True)

# Load data
csv_path = r"indicator_data_xau_table_m15_2024_7.csv"
df_raw = pd.read_csv(csv_path)
df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
df_raw = df_raw.set_index('timestamp')

# Calculate zigzag
df_raw = calculate_zigzag(df_raw)

# Plot segments (e.g., every 7 days)
segment_days = 7
start_date = df_raw.index.min()
end_date = df_raw.index.max()

current_start = start_date
while current_start < end_date:
    current_end = current_start + timedelta(days=segment_days)
    if current_end > end_date:
        current_end = end_date
    plot_zigzag_segment(df_raw, current_start, current_end)
    current_start = current_end

print(f"ZigZag plots with EMA34, EMA89, and candlesticks saved to folder: 'zigzag'")