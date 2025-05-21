import os
import sys
import joblib
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import tqdm

CANDLE_PATTERN = {"Hammer": 1, "Inverted Hammer": 2, "Hanging Man": 3, "Shooting Star": 4, "Marubozu": 5, "Doji": 6, "Dragonfly Doji": 7, "Gravestone Doji": 8, "Spinning Top": 9}

CANDLE_DOUBLE = {"Bullish Engulfing": 10, "Bearish Engulfing": 11, "Piercing": 12, 
                 "Dark Cloud Cover": 13, "Bullish Harami": 14, "Bearish Harami": 15}
CANDLE_TRIPLE_PATTERN = {"Morning Star": 16, "Evening Star": 17, "Three Outside Up": 18, 
                        "Three Outside Down": 19, "Three Inside Up": 20, "Three Inside Down": 21}

def plot_candlestick_by_time(df_raw, start_index, end_index, list_predict):
    df_candle = df_raw.iloc[start_index:end_index][['timestamp', 'open', 'high', 'low', 'close']]
    df_candle['timestamp'] = pd.to_datetime(df_candle['timestamp'])
    df_candle.set_index('timestamp', inplace=True)
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(1, 2, width_ratios=[10, 1], figure=fig)
    ax1 = fig.add_subplot(gs[0])
    
    mc = mpf.make_marketcolors(up='green', down='red', edge={'up': 'green', 'down': 'red'}, 
                               wick={'up': 'green', 'down': 'red'}, volume={'up': 'green', 'down': 'red'})
    s = mpf.make_mpf_style(marketcolors=mc)
    
    mpf.plot(
        df_candle, type='candle', style='charles', ax=ax1,
        volume=False, ylabel='Price'
    )
    ax1.set_title('Predict Price')
    ax1.set_ylabel('Price')
    
    list_ema_34 = []
    list_ema_89 = []
    list_ema_5 = []
    list_support = []
    list_resistance = []
    
    for index, predicted_class, ema_34, ema_89, ema_5, support_1, support_2, resistance_1, resistance_2 in list_predict:
        price = df_raw.iloc[index]['close']
        
        list_ema_34.append(ema_34)
        list_ema_89.append(ema_89)
        list_ema_5.append(ema_5)
        
        color = 'green' if predicted_class == 1 else 'red'
        if predicted_class in [0, 1]:
            ax1.scatter([index - start_index], [price], color=color, s=100)
        
        # Plot key levels as horizontal lines with x-limits
        x_start = 0  # Start of the plot
        x_end = index - start_index  # Up to the current index
        
        # if not np.isnan(support_1):
        #     list_support.append(support_1)
        #     ax1.hlines(y=support_1, xmin=x_start, xmax=x_end, color='blue', linestyle='--', alpha=0.5)
        # if not np.isnan(support_2):
        #     list_support.append(support_2)
        #     ax1.hlines(y=support_2, xmin=x_start, xmax=x_end, color='blue', linestyle='--', alpha=0.5)
        # if not np.isnan(resistance_1):
        #     list_resistance.append(resistance_1)
        #     ax1.hlines(y=resistance_1, xmin=x_start, xmax=x_end, color='purple', linestyle='--', alpha=0.5)
        # if not np.isnan(resistance_2):
        #     list_resistance.append(resistance_2)
        #     ax1.hlines(y=resistance_2, xmin=x_start, xmax=x_end, color='purple', linestyle='--', alpha=0.5)
    
    ax1.plot(list_ema_34, color='pink', label='EMA 34')   
    ax1.plot(list_ema_89, color='yellow', label='EMA 89') 
    ax1.plot(list_ema_5, color='green', label='EMA 5')
    ax1.legend()
    
    time_stamp = df_raw.iloc[start_index]['timestamp'].replace('-', '').replace(':', '').replace(' ', '')
        
    output_folder_image = "output_label"
    os.makedirs(output_folder_image, exist_ok=True)
    output_image_path = os.path.join(
        output_folder_image, f"{time_stamp}_{index}.png"
    )

    # Save the figure
    plt.savefig(output_image_path)
    plt.close(fig)  # Close the figure to free memory
    
list_label = []  
start_date = None
current_year = None
end_date = None 
csv_path = r"indicator_data_xau_table_h1_2024.csv"
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
    ema_5 = df_raw.iloc[index]['ema_5']
    support_1 = df_raw.iloc[index]['support1'] + df_raw.iloc[index]['close']
    support_2 = df_raw.iloc[index]['support2'] + df_raw.iloc[index]['close']
    resistance_1 = df_raw.iloc[index]['resistance1'] + df_raw.iloc[index]['close']
    resistance_2 = df_raw.iloc[index]['resistance2'] + df_raw.iloc[index]['close']
    
    feature_plot = (index, label, ema_34, ema_89, ema_5, support_1, support_2, resistance_1, resistance_2)
    list_label.append(feature_plot)
    if current_date - start_date > 5:
        end_date = current_date
        end_index = index
        plot_candlestick_by_time(df_raw, start_index, end_index, list_label)
        list_label = []
        start_date = current_date
        start_index = index