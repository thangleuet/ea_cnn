import os
import sys
import joblib
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

CANDLE_PATTERN = {"Hammer": 1, "Inverted Hammer": 2, "Hanging Man": 3, "Shooting Star": 4, "Marubozu" : 5, "Doji": 6, "Dragonfly Doji": 7, "Gravestone Doji": 8, "Spinning Top": 9}

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
    
    output_ta = df_raw['output_ta'].values[start_index:end_index]

    ha_candle = []
    for ta in output_ta:
        ha = eval(ta)["ha_candle"]
        ha_candle.append(ha)

    ha_candle = pd.DataFrame(ha_candle, columns=['Open', 'High', 'Low', 'Close'], index=df_candle.index)
    ha_candle = ha_candle - 30
    ha_candle.index = df_candle.index
    
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
    mpf.plot(
        ha_candle, type='candle', style='charles', ax=ax1,
        volume=False, ylabel='Heikin-Ashi Close'
    )
    ax1.set_title('Predict Price')
    ax1.set_ylabel('Price')
    
    for index, confidence, predicted_class, candlestick_pattern in list_predict:
        candle_type = candlestick_pattern["candle_type"]
        candle_pattern = candlestick_pattern["candle_pattern"]
        next_trend = candlestick_pattern["next_trend"]
        
        price = df_raw.iloc[index]['close']
        color = 'green' if predicted_class == 1 else 'red'
        ax1.scatter([index-start_index], [price], color=color, s=100)
        ax1.text(index-start_index, df_raw.iloc[index]['low'], f"{round(confidence, 2)}", fontsize=10, ha='center', va='center', color=color)
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
    
def trend_analys(ha_turn_list, offset):
    main_trend = None
    main_trend_start = None
    main_trend_end = None
    entry_direction = None
    current_max_high = None
    current_min_low = None
    
    start = max(0, len(ha_turn_list) - offset)
    min_max_index = start
    len_list = len(ha_turn_list)

    while min_max_index < len_list:
        max_high = -float('inf')
        min_low = float('inf')
        max_high_index = min_low_index = -1
        max_high_date_time = min_low_date_time = None

        # Tìm giá trị cao nhất và thấp nhất trong danh sách
        for i in range(min_max_index, len_list):
            value = ha_turn_list[i][2]
            date_time = ha_turn_list[i][1]
            if value >= max_high:
                max_high = value
                max_high_index = i
                max_high_date_time = date_time
            if value <= min_low:
                min_low = value
                min_low_index = i
                min_low_date_time = date_time

        # Xác định xu hướng chính nếu chênh lệch >= 10
        diff = abs(max_high - min_low)
        if max_high_index > min_low_index and diff >= 10:
            main_trend = "UP"
            main_trend_start = min_low_date_time
            main_trend_end = max_high_date_time
            min_max_index = max_high_index + 1
            current_max_high = max_high
            current_min_low = min_low
            
        elif max_high_index < min_low_index and diff >= 10:
            main_trend = "DOWN"
            main_trend_start = max_high_date_time
            main_trend_end = min_low_date_time
            min_max_index = min_low_index + 1
            current_max_high = max_high
            current_min_low = min_low
        else:
            min_max_index += 1

    entry_direction = "BUY" if main_trend == "UP" else "SELL" if main_trend == "DOWN" else None
    return main_trend, entry_direction, main_trend_start, main_trend_end, current_max_high, current_min_low

csv_path = r"indicator_data_xau_table_h1_2023_10.csv"
df_raw = pd.read_csv(csv_path)

list_features = np.load('weights/list_features.npy', allow_pickle='TRUE')

# Load scaler
scaler = np.load('weights/scaler.npy', allow_pickle='TRUE').item()

# Load feat_indx
feat_indx = np.load('weights/feat_idx.npy', allow_pickle='TRUE')

# Load model
params = {'batch_size': 80, 'conv2d_layers': {'conv2d_do_1': 0.2, 'conv2d_filters_1': 64, 'conv2d_kernel_size_1': 3, 'conv2d_mp_1': 0, 
                                               'conv2d_strides_1': 1, 'kernel_regularizer_1': 0.0, 'conv2d_do_2': 0.3, 
                                               'conv2d_filters_2': 64, 'conv2d_kernel_size_2': 3, 'conv2d_mp_2': 2, 'conv2d_strides_2': 1, 
                                               'kernel_regularizer_2': 0.0, 'layers': 'two'}, 
           'dense_layers': {'dense_do_1': 0.3, 'dense_nodes_1': 128, 'kernel_regularizer_1': 0.0, 'layers': 'one'},
           'epochs': 300, 'lr': 0.001, 'optimizer': 'adam'}

model = create_model_cnn(params)
best_model_path = os.path.join('weights', 'best_model.h5') 
# best_model_path = "model_epoch_160.h5"
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
for index in tqdm.tqdm(range(len(df_raw))):
    if  index >= len(df_raw) - 100 or index < 100:
        continue
    filter_data = [df_raw.iloc[index].get(feat) for feat in list_features]
    input_data = np.array(filter_data)
    input_data = scaler.transform(input_data.reshape(1, -1))
    input_data = input_data[:, feat_indx]
    candlestick_pattern = df_raw.iloc[index]['candlestick_pattern']
    ema_25 = df_raw.iloc[index]['ema_34']
    ema_7 = df_raw.iloc[index]['ema_89']
    list_ema_7.append(ema_7)
    list_ema_25.append(ema_25)

    ha_candle_status = df_raw.iloc[index]['streak_count']
    candle_type = df_raw.iloc[index]['ha_type']
   
    batch_x = input_data
    x_temp = np.reshape(batch_x, (7, 7))
    x_temp = np.stack((x_temp,) * 1, axis=-1)
    x_temp = np.expand_dims(x_temp, axis=0)
    output = model(x_temp)
    # convert tensor to numpy
    output = output.numpy()
    pred = model.predict(x_temp)
    predicted_class = np.argmax(pred, axis=1)[0]
    confidence = float(np.max(pred, axis=1)[0])
    
    current_date = pd.to_datetime(df_raw.iloc[index]['timestamp']).dayofyear
    if start_date is None:
        start_date = current_date
        start_index = index
        
    candlestick_pattern = eval(candlestick_pattern.replace('null', '2').replace('false', '0').replace('true', '1'))
    if confidence < 0.7 or confidence > 0.9:
        predicted_class = 2
    list_predict_class.append(predicted_class)
    list_predict_class = list_predict_class[-2:]
    # if ha_candle_status == 1:
    #     if (1 in list_predict_class and candle_type == 1) or (0 in list_predict_class and candle_type == 0):
    if predicted_class in [0, 1]:
            # predicted_class = 1 if 1 in list_predict_class else 0
            list_predict.append([index, confidence, predicted_class, candlestick_pattern])
            number_ha_candle, ha_status, status = plot_candlestick(df_raw, index, predicted_class, confidence)
            if "TP" in status:
                count_tp += 1
            elif "SL" in status:
                count_sl += 1
            else:
                count_fail += 1
        
            print(f"TP: {count_tp} SL: {count_sl}")
    
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
print(total_tp, total_sl)
