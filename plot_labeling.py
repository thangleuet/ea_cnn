import os
import sys
import joblib
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import tqdm

CANDLE_PATTERN = {"Hammer": 1, "Inverted Hammer": 2, "Hanging Man": 3, "Shooting Star": 4, "Marubozu" : 5, "Doji": 6, "Dragonfly Doji": 7, "Gravestone Doji": 8, "Spinning Top": 9}

CANDLE_DOUBLE = {"Bullish Engulfing": 10, "Bearish Engulfing": 11, "Piercing": 12, 
                  "Dark Cloud Cover": 13, "Bullish Harami": 14, "Bearish Harami": 15}
CANDLE_TRIPPLE_PATTERN = {"Morning Star": 16, "Evening Star": 17, "Three Outside Up": 18, 
                  "Three Outside Down": 19, "Three Inside Up": 20, "Three Inside Down": 21}

def plot_candlestick_predict(df_raw, index, label):
    start_index = max(0, index - 100)
    index_db_len = df_raw.index[len(df_raw)-1]
    end_index = min(index_db_len, index + 100)
    df_candle = df_raw.loc[start_index:end_index][['timestamp', 'open', 'high', 'low', 'close']]
    df_candle['timestamp'] = pd.to_datetime(df_candle['timestamp'])
    df_candle.set_index('timestamp', inplace=True)
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(1, 2, width_ratios=[10, 1], figure=fig)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    mc= mpf.make_marketcolors(up='green', down='red', edge={'up': 'green', 'down': 'red'}, wick={'up': 'green', 'down': 'red'}, volume={'up': 'green', 'down': 'red'})
    s = mpf.make_mpf_style(marketcolors=mc)
    
    mpf.plot(
        df_candle, type='candle', style='charles', ax = ax1,
        volume=False, ylabel='Price'
    )
    ax1.set_title('Predict Price')
    ax1.set_ylabel('Price')
    
    ema34_data = df_raw.loc[start_index:end_index]['ema_34'].values
    ema89_data = df_raw.loc[start_index:end_index]['ema_89'].values
    price = df_raw.loc[index]['close']
    
    ax1.plot(ema34_data, label='EMA 34', color='orange', linewidth=1.5)
    ax1.plot(ema89_data, label='EMA 89', color='purple', linewidth=1.5)
    
    color = 'green' if label == 1 else 'red'
    if label in [0, 1]:
        ax1.scatter([index-start_index], [price], color=color, s=100)
    
    time_stamp = df_raw.loc[start_index]['timestamp'].replace('-', '').replace(':', '').replace(' ', '')
    output_folder_image = f"output_label"
    os.makedirs(output_folder_image, exist_ok=True)
    output_image_path = os.path.join(
        output_folder_image, f"{index}_{time_stamp}.png"
    )

    # Save the figure
    plt.savefig(output_image_path)
    plt.clf()
    

def plot_candlestick_by_time(df_raw, start_index, end_index, list_predict):
    df_candle = df_raw.iloc[start_index:end_index][['timestamp', 'open', 'high', 'low', 'close']]
    df_candle['timestamp'] = pd.to_datetime(df_candle['timestamp'])
    df_candle.set_index('timestamp', inplace=True)
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(1, 2, width_ratios=[10, 1], figure=fig)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    mc= mpf.make_marketcolors(up='green', down='red', edge={'up': 'green', 'down': 'red'}, wick={'up': 'green', 'down': 'red'}, volume={'up': 'green', 'down': 'red'})
    s = mpf.make_mpf_style(marketcolors=mc)
    
    mpf.plot(
        df_candle, type='candle', style='charles', ax = ax1,
        volume=False, ylabel='Price'
    )
    ax1.set_title('Predict Price')
    ax1.set_ylabel('Price')
    list_ema_7 = []
    list_ema_34 = []
    list_ema_89 = []
    for index, predicted_class, ema_34, ema_89, rsi, candlestick_pattern in list_predict:
        price = df_raw.iloc[index]['close']
        if candlestick_pattern is not None:
            candlestick_pattern_data = eval(candlestick_pattern.replace("null", "2").replace("false", "0").replace("true", "1"))
            candletype = candlestick_pattern_data["candle_type"]
        else:
            candletype = ""
            
        list_ema_34.append(ema_34)
        list_ema_89.append(ema_89)
        
        color = 'green' if predicted_class == 1 else 'red'
        if predicted_class in [0, 1]:
            ax1.scatter([index-start_index], [price], color=color, s=100)
        # ax1.text(index-start_index, price, f"{rsi}", fontsize=10, color=color)
               
    ax1.plot(list_ema_34, color='pink', label='ema_34')   
    ax1.plot(list_ema_89, color='yellow', label='ema_89') 
    ax1.legend()
    
    time_stamp = df_raw.iloc[start_index]['timestamp'].replace('-', '').replace(':', '').replace(' ', '')
        
    output_folder_image = f"output_label"
    os.makedirs(output_folder_image, exist_ok=True)
    output_image_path = os.path.join(
        output_folder_image, f"{time_stamp}_{index}.png"
    )

    # Save the figure
    plt.savefig(output_image_path)
    plt.clf()
    
list_label = []  
start_date = None
current_year = None
end_date = None 
csv_path = r"indicator_data_xau_table_m5_2022.csv"
df_raw = pd.read_csv(csv_path)
for index in tqdm.tqdm(range(len(df_raw))):
    label = df_raw.iloc[index]['labels']
    current_date = pd.to_datetime(df_raw.iloc[index]['timestamp']).dayofyear
    year = pd.to_datetime(df_raw.iloc[index]['timestamp']).year
    rsi = df_raw.iloc[index]['rsi_14']
    if current_year is None or current_year != year:
        current_year = year
        start_date = None
    if start_date is None:
        start_date = current_date
        start_index = index
    ema_89 = df_raw.iloc[index]['ema_89']
    ema_34 = df_raw.iloc[index]['ema_34']
    candlestick_pattern = None
    index_db = df_raw.index[index]
    # if label in [0,1]:
    #     plot_candlestick_predict(df_raw, index_db, label)
    
    feature_plot = (index, label, ema_34, ema_89, rsi, candlestick_pattern)
    list_label.append(feature_plot)
    if current_date - start_date > 7:
        end_date = current_date
        end_index = index
        plot_candlestick_by_time(df_raw, start_index, end_index, list_label)
        list_label = []
        start_date = current_date
        start_index = index
    
    
    