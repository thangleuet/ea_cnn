import os
import sys
import joblib
from keras.src.layers.rnn.lstm import lstm_with_backend_selection
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import mplfinance as mpf
import tensorflow as tf
from tensorflow.python import keras
from keras.models import load_model
import tqdm

import matplotlib.pyplot as plt
from models.model_cnn_utils import create_model_cnn

CANDLE_PATTERN = {"Hammer": 1, "Inverted Hammer": 2, "Hanging Man": 3, "Shooting Star": 4, "Marubozu" : 5, "Doji": 6, "Dragonfly Doji": 7, "Gravestone Doji": 8, "SpinningTop": 9}

CANDLE_DOUBLE = {"Bullish Engulfing": 10, "Bearish Engulfing": 11, "Piercing": 12, 
                  "Dark Cloud Cover": 13, "Bullish Harami": 14, "Bearish Harami": 15}
CANDLE_TRIPPLE_PATTERN = {"Morning Star": 16, "Evening Star": 17, "Three Outside Up": 18, 
                  "Three Outside Down": 19, "Three Inside Up": 20, "Three Inside Down": 21}

def plot_candlestick(df_raw, index, predicted_class, confidence): 
    time_stamp = df_raw.iloc[index]['timestamp'].replace('-', '').replace(':', '').replace(' ', '')
    df_raw_init = df_raw.iloc[index - 100:index + 100]
    df_candle = df_raw.iloc[index - 100:index + 100][['timestamp', 'open', 'high', 'low', 'close']]
    df_candle['timestamp'] = pd.to_datetime(df_candle['timestamp'])
    df_candle.set_index('timestamp', inplace=True)   
    
    labels = df_raw.iloc[index]['labels']
    
    # output_ta = df_raw['output_ta'].values[index-100:index+100]

    # ha_candle = []
    # for ta in output_ta:
    #     ha = eval(ta)["ha_candle"]
    #     ha_candle.append(ha)

    # ha_candle = pd.DataFrame(ha_candle, columns=['Open', 'High', 'Low', 'Close'], index=df_candle.index)
    # ha_candle = ha_candle - 50
    # ha_candle.index = df_candle.index

    # Define scatter plot data aligned with timestamps
    marker = '^' if predicted_class == 1 else 'v'
    color = 'green' if predicted_class == 1 else 'red'
    label = 'Buy' if predicted_class == 1 else 'Sell'

    # Initialize scatter data with NaNs
    scatter_data = np.full(len(df_candle), np.nan)
    scatter_data[100] = df_candle['close'].iloc[100]  # Target point for the scatter plot

    # ha_scatter = np.full(len(ha_candle), np.nan)
    # ha_scatter[100] = ha_candle['Close'].iloc[100]  # Adjusted point in ha_candle for visibility

    # First subplot for candlestick plot with scatter_data

    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(1, 2, width_ratios=[5, 1], figure=fig)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Add text in ax3
    ax2.set_axis_off() 
    plt.subplots_adjust(
                left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.15
            )
    mc= mpf.make_marketcolors(up='green', down='red', edge={'up': 'green', 'down': 'red'}, wick={'up': 'green', 'down': 'red'}, volume={'up': 'green', 'down': 'red'})
    s = mpf.make_mpf_style(marketcolors=mc)
    
    mpf.plot(
        df_candle, type='candle', style='charles', ax = ax1,
        addplot=[mpf.make_addplot(scatter_data, type='scatter', markersize=100, marker=marker, color=color, label=label, ax=ax1)],
        volume=False)
    ax1.set_title('Predict Price')
    ax1.set_ylabel('Price')
    
    tp = 10
    sl = 10
    entry_price = df_candle['close'].iloc[100]
    
    if predicted_class == 1:
        tp_price = entry_price + tp
        sl_price = entry_price - sl
        ax1.axhline(y= df_candle['close'].iloc[100] + tp, color='green', linestyle='--', alpha=0.5)
        ax1.axhline(y= df_candle['close'].iloc[100] - sl, color='red', linestyle='--', alpha=0.5)
    else:
        tp_price = entry_price - tp
        sl_price = entry_price + sl
        ax1.axhline(y= df_candle['close'].iloc[100] - tp, color='green', linestyle='--', alpha=0.5)
        ax1.axhline(y= df_candle['close'].iloc[100] + sl, color='red', linestyle='--', alpha=0.5)
    
    status = "In progress"
    count = 0
    for i,row in df_candle.iterrows():
        # Kiểm tra TP/SL
        if count > 100:
            if predicted_class == 1:  # Giao dịch mua
                if row['low'] <= sl_price:
                    status = "SL"
                    break  # Nếu đã chạm SL thì không cần kiểm tra nữa
                elif row['high'] >= tp_price:
                    status = "TP"
                    break  # Nếu đã chạm TP thì không cần kiểm tra nữa
                
            else:  # Giao dịch bán
                if row['high'] >= sl_price:
                    status = "SL"
                    break  # Nếu đã chạm SL thì không cần kiểm tra nữa
                elif row['low'] <= tp_price:
                    status = "TP"
                    break  # Nếu đã chạm TP thì không cần kiểm tra nữa
        count += 1   

    # main_trend_start, main_trend_en
    # ha_scatter_plot = mpf.make_addplot(
    #     ha_scatter, type='scatter', markersize=100, marker=marker, color=color, label=f"{label} HA", ax=ax1
    # )
    ax2.text(0.5, 0.5, f"Status: {status}", fontsize=15, ha='center', va='center')
    ax2.text(0.5, 0.4, f"Confidence: {confidence}", fontsize=15, ha='center', va='center')
    ax2.text(0.5, 0.3, f"Label: {labels}", fontsize=15, ha='center', va='center')
    # mpf.plot(
    #     ha_candle, type='candle', style='charles', ax=ax1,
    #     addplot=[ha_scatter_plot],  # Correctly associate this addplot with ax1
    #     volume=False, ylabel='Heikin-Ashi Close'
    # )
    
    output_folder_image = f"output/buy/{status}" if predicted_class == 1 else f"output/sell/{status}"
    os.makedirs(output_folder_image, exist_ok=True)
    output_image_path = os.path.join(
        output_folder_image, f"{time_stamp}_{index}_{round(confidence, 2)}.png"
    )

    # Save the figure
    plt.savefig(output_image_path)
    plt.clf()

    return number_ha_candle, ha_status, status

def plot_predict(df_raw, start_index, end_index, list_predict, list_ema_7, list_ema_25):
    df_candle = df_raw.iloc[start_index:end_index][['timestamp', 'open', 'high', 'low', 'close']]
    df_candle['timestamp'] = pd.to_datetime(df_candle['timestamp'])
    df_candle.set_index('timestamp', inplace=True)
    
    # output_ta = df_raw['output_ta'].values[start_index:end_index]

    # ha_candle = []
    # for ta in output_ta:
    #     ha = eval(ta)["ha_candle"]
    #     ha_candle.append(ha)

    # ha_candle = pd.DataFrame(ha_candle, columns=['Open', 'High', 'Low', 'Close'], index=df_candle.index)
    # ha_candle = ha_candle - 30
    # ha_candle.index = df_candle.index
    
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
    # mpf.plot(
    #     ha_candle, type='candle', style='charles', ax=ax1,
    #     volume=False, ylabel='Heikin-Ashi Close'
    # )
    ax1.set_title('Predict Price')
    ax1.set_ylabel('Price')
    
    for index, confidence, predicted_class, candlestick_pattern, count_ha in list_predict:
        candle_type = candlestick_pattern["candle_type"]
        candle_pattern = candlestick_pattern["candle_pattern"]
        next_trend = candlestick_pattern["next_trend"]
        
        price = df_raw.iloc[index]['close']
        if predicted_class == 1:
            color = 'green'
        elif predicted_class == 0:
            color = 'red'
        else:
            color = 'blue'
        ax1.scatter([index-start_index], [price], color=color, s=100)
        ax1.text(index-start_index, df_raw.iloc[index]['low']-2, f"{round(count_ha, 2)}", fontsize=12, ha='center', va='center', color=color)
        if candle_type in CANDLE_PATTERN or candle_type in CANDLE_DOUBLE or candle_type in CANDLE_TRIPPLE_PATTERN:
            ax1.text(index-start_index, price, f"{candle_type}_{next_trend}", fontsize=10, color=color)
    time_stamp = df_raw.iloc[start_index]['timestamp'].replace('-', '').replace(':', '').replace(' ', '')
        
    ax1.plot(list_ema_7, color='blue')
    ax1.plot(list_ema_25, color='green')
    output_folder_image = f"output/date"
    os.makedirs(output_folder_image, exist_ok=True)
    output_image_path = os.path.join(
        output_folder_image, f"{time_stamp}_{start_index}.png"
    )

    # Save the figure
    plt.savefig(output_image_path)
    plt.clf()
    

csv_path = r"indicator_data_xau_table_h1_2024_0.005.csv"
df_raw = pd.read_csv(csv_path)

list_features = np.load('weights_lstm/list_features.npy', allow_pickle='TRUE')

# Load scaler
scaler = np.load('weights_lstm/scaler.npy', allow_pickle='TRUE').item()
lstm_scaler = np.load('weights_lstm/lstm_scaler.npy', allow_pickle='TRUE').item()

# Load feat_indx
feat_indx = np.load('weights_lstm/feat_idx.npy', allow_pickle='TRUE')

best_model_path = os.path.join('weights_lstm', 'best_model.h5') 
model = load_model(best_model_path)

count_tp = 0
count_sl = 0
count_fail = 0

current_entry = 0
current_status = 0
total_sl = 0
total_tp = 0
profit = 100
count_equal = 0

number_ha_candle = 0
ha_status = None
ha_turn_list = []
list_predict = []
start_date = None
end_date = None
list_ema_7 = []
list_ema_25 = []
list_predict_class = []
current_index = 0
timestep = 12
for index in tqdm.tqdm(range(len(df_raw))):
    if  index >= len(df_raw) - 100 or index < 100:
        continue
    filter_data = [df_raw.iloc[index].get(feat) for feat in list_features]
    input_data = np.array(filter_data)
    input_data = scaler.transform(input_data.reshape(1, -1))
    input_data = input_data[:, feat_indx]
    
    feature_lstm = ['close', 'open', 'high', 'low', 'volume']
    lstm_data = df_raw[feature_lstm].iloc[index-timestep:index].values
    lstm_data = lstm_scaler.transform(lstm_data.reshape(-1, len(feature_lstm))).reshape(-1, timestep, len(feature_lstm))
    
    pred = model.predict([lstm_data, input_data])
    predicted_class = np.argmax(pred, axis=1)[0]
    confidence = float(np.max(pred, axis=1)[0])
    
    candlestick_pattern = df_raw.iloc[index]['candlestick_pattern']
    ema_7 = df_raw.iloc[index]['ema_34']
    ema_25 = df_raw.iloc[index]['ema_89']
    list_ema_7.append(ema_7)
    list_ema_25.append(ema_25)
    
    count_ema_7 = df_raw.iloc[index]['count_ema_7']
    current_price = df_raw.iloc[index]['close']

    ha_candle_status = df_raw.iloc[index]['streak_count']
    candle_type = df_raw.iloc[index]['ha_type']
    
    current_date = pd.to_datetime(df_raw.iloc[index]['timestamp']).dayofyear
    rsi = df_raw.iloc[index]['rsi_14']
    stochrsi = df_raw.iloc[index]['stochrsi_d_14'] - df_raw.iloc[index]['stochrsi_k_14']
    if start_date is None:
        start_date = current_date
        start_index = index
        
    candlestick_pattern = eval(candlestick_pattern.replace('null', '2').replace('false', '0').replace('true', '1'))
    if confidence < 0.7:
        predicted_class = 2
    if predicted_class == 1 and rsi > 40:
        predicted_class = 2
    if predicted_class == 0 and rsi < 60:
        predicted_class = 2
        
    if count_ema_7 < 10 or abs(ema_7 - ema_25)/current_price < 3/2000:
        predicted_class = 2
    
    # if abs(df_raw.iloc[index]['diff_ema_34']) > 1 and abs(df_raw.iloc[index]['diff_ema_89']) > 5:
    #     predicted_class = 2
    
    list_predict_class.append(predicted_class)
    list_predict_class = list_predict_class[-2:]
    if predicted_class in [0, 1]:
            list_predict.append([index, confidence, predicted_class, candlestick_pattern, rsi])
            number_ha_candle, ha_status, status = plot_candlestick(df_raw, index, predicted_class, confidence)
            if "TP" in status:
                count_tp += 1
            elif "SL" in status:
                count_sl += 1
            else:
                count_fail += 1
        
            print(f"TP: {count_tp} SL: {count_sl}")
    current_index = index
    
    if current_date - start_date > 9:
        end_date = current_date
        end_index = index
        plot_predict(df_raw, start_index, end_index, list_predict, list_ema_7, list_ema_25) 
        list_predict = [] 
        list_ema_7 = []
        list_ema_25 = []
        start_date = current_date
        start_index = index
        
        
print(count_tp, count_sl, count_fail, count_equal)
print("TP acc:", count_tp / (count_tp + count_sl) * 100)

