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

def check_tp_sl_current(df_raw, index, predicted_class, output_ta, main_trend, current_max_high, current_min_low, main_trend_end, number_ha_candle, check_candle_status_current):
    # Get the close price of the (index + 1) candle
    entry_price = df_raw.iloc[index]['close']

    status_tp = 0
    tp = 5
    sl = 5
    status_candle_last_1 = 1 if eval(output_ta[index - 1])["ha_candle"]["Close"] - eval(output_ta[index - 1])["ha_candle"]["Open"] > 0 else 0
    status_candle_current = 1 if eval(output_ta[index])["ha_candle"]["Close"] - eval(output_ta[index])["ha_candle"]["Open"] > 0 else 0

    start_value = current_max_high if main_trend == "DOWN" else current_min_low
    end_value = current_min_low if main_trend == "DOWN" else current_max_high

    if predicted_class == 1:
        if (number_ha_candle < 5 and check_candle_status_current == 1) or (number_ha_candle < 4 and check_candle_status_current == 0):
            predicted_class = 2
    elif predicted_class == 0:
        if (number_ha_candle < 5 and check_candle_status_current == 0) or (number_ha_candle < 4 and check_candle_status_current == 1):
            predicted_class = 2

    if main_trend is not None:
        current_value = df_raw.iloc[index]['high'] if main_trend == "UP" else df_raw.iloc[index]['low']
        percent = (end_value - current_value) / (end_value - start_value)

        
        # prev_candle = df_raw.iloc[main_trend_end : index - 1].values
        # for candle in prev_candle:

        if main_trend == "UP":
            if 0.01 < percent < 0.3 and predicted_class == 0:
                predicted_class = 0
                # if index - main_trend_end > 20:
                #     predicted_class = -1
            elif percent > 0.5 and predicted_class == 1:
                predicted_class = 1
            # elif -0.2 < percent < 0.2 and predicted_class == 1:
            #     predicted_class = 1
            #     if index - main_trend_end < 10:
            #         predicted_class = -1
            else:
                predicted_class = -1
        elif main_trend == "DOWN":
            if percent > 0.5 and predicted_class == 0:
                predicted_class = 0
            elif 0.01 < percent < 0.3 and predicted_class == 1:
                predicted_class = 1
                # if index - main_trend_end > 20:
                #     predicted_class = -1ơ mà mai e đ
            # elif -0.2 < percent < 0.2 and predicted_class == 0:
            #     predicted_class = 0
            #     if index - main_trend_end < 10:
            #         predicted_class = -1
            else:
                predicted_class = -1

        # if index - main_trend_end < 3:
        #     predicted_class = -1
    else:
        predicted_class = -1

    if predicted_class != -1:
        for i in range(1, 100):
            if index + i < len(df_raw):  # Ensure we don't go out of bounds
                if predicted_class == 1:
                    if entry_price - df_raw.loc[index + i]['low'] >= sl:
                        status_tp = 0
                        break
                    if df_raw.loc[index + i]['high'] - entry_price >= tp:
                        status_tp = 1
                        break
                else:
                    if df_raw.loc[index + i]['high'] - entry_price >= sl:
                        status_tp = 0
                        break
                    if entry_price - df_raw.loc[index + i]['low'] >= tp:
                        status_tp = 1
                        break
    else:
        status_tp = -1

    return status_tp, sl, tp, predicted_class


def plot_candlestick(df_raw, index, predicted_class, status_tp, sl, tp, conf, current_rsi, output_ta, main_trend, main_trend_start, main_trend_end, current_max_high, current_min_low): 
    time_stamp = df_raw.iloc[index]['timestamp'].replace('-', '').replace(':', '').replace(' ', '')
    df_candle = df_raw.iloc[index - 100:index + 100][['timestamp', 'open', 'high', 'low', 'close']]
    df_candle['timestamp'] = pd.to_datetime(df_candle['timestamp'])
    df_candle.set_index('timestamp', inplace=True)         

    output_ta = df_raw['output_ta'].values[index-100:index+100]

    ha_candle = []
    for ta in output_ta:
        ha = eval(ta)["ha_candle"]
        ha_candle.append(ha)

    ha_candle = pd.DataFrame(ha_candle, columns=['Open', 'High', 'Low', 'Close'], index=df_candle.index)

    # Create output folder based on prediction class (buy/sell)
    output_folder_image = f"output/buy/{status_tp}" if predicted_class == 1 else f"output/sell/{status_tp}"
    os.makedirs(output_folder_image, exist_ok=True)
    
    output_image_path = os.path.join(
        output_folder_image, f"{time_stamp}_{index}_{round(conf, 2)}"
        f"sl{sl}_tp{tp}.png"
    )

    # Define scatter plot data aligned with timestamps
    marker = '^' if predicted_class == 1 else 'v'
    color = 'green' if predicted_class == 1 else 'red'
    label = 'Buy' if predicted_class == 1 else 'Sell'

    # Initialize scatter data with NaNs
    scatter_data = np.full(len(df_candle), np.nan)
    scatter_data[100] = df_candle['close'].iloc[100]  # Target point for the scatter plot

    ha_scatter = np.full(len(ha_candle), np.nan)
    ha_scatter[100] = ha_candle['Close'].iloc[100]  # Adjusted point in ha_candle for visibility

    # First subplot for candlestick plot with scatter_data

    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(2, 2, width_ratios=[4, 1], figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[:, 1])

    # Add text in ax3
    ax3.text(
        0.5, 0.5, f"Index: {index}, \nTime: {time_stamp}, \nRSI: {round(current_rsi, 2)}", fontsize=12, ha="center", va="center", transform=ax3.transAxes
    )
    if main_trend is not None:
        ax3.text(
            0.5, 0.3, f"Main Trend: {main_trend}", fontsize=12, ha="center", va="center", transform=ax3.transAxes
        )
    ax3.set_axis_off() 
    plt.subplots_adjust(wspace=0.15, left=0.05, right=0.95, top=0.95, bottom=0.05)
    mc= mpf.make_marketcolors(up='green', down='red', edge={'up': 'green', 'down': 'red'}, wick={'up': 'green', 'down': 'red'}, volume={'up': 'green', 'down': 'red'})
    s = mpf.make_mpf_style(marketcolors=mc)
    
    mpf.plot(
        df_candle, type='candle', style='charles', ax = ax1,
        addplot=[mpf.make_addplot(scatter_data, type='scatter', markersize=100, marker=marker, color=color, label=label, ax=ax1)],
        volume=False)
    ax1.set_title('Predict Price')
    ax1.set_ylabel('Price')
    ax1.axhline(y= df_candle['close'].iloc[100] + 5, color='green' if predicted_class == 1 else 'red', linestyle='--', alpha=0.5)
    ax1.axhline(y= df_candle['close'].iloc[100] - 5, color='red' if predicted_class == 1 else 'green', linestyle='--', alpha=0.5)

    # main_trend_start, main_trend_end
    if main_trend is not None:
        start_index = main_trend_start - (index - 100) if main_trend_start - (index - 100) > 0 else 0
        end_index = main_trend_end - (index - 100) if main_trend_end - (index - 100)  > 0 else 0
        start_value = current_max_high if main_trend == 'DOWN' else current_min_low
        end_value = current_min_low if main_trend == 'DOWN' else current_max_high
        ax1.scatter(start_index, start_value, marker='o', color='blue', label='Main Trend Start')
        ax1.scatter(end_index, end_value, marker='o', color='blue', label='Main Trend End')
        ax1.plot([start_index, end_index], [start_value, end_value], color='blue', label='Main Trend')
        ax1.legend()
    
    ha_scatter_plot = mpf.make_addplot(
        ha_scatter, type='scatter', markersize=100, marker=marker, color=color, label=f"{label} HA", ax=ax2
    )
    mpf.plot(
        ha_candle, type='candle', style='charles', ax=ax2,
        addplot=[ha_scatter_plot],  # Correctly associate this addplot with ax2
        volume=False, ylabel='Heikin-Ashi Close'
    )

    # Save the figure
    plt.savefig(output_image_path)
    plt.clf()

    return number_ha_candle, ha_status


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

csv_path = r"indicator_data_table_m15_2022.csv"
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
rsi_value = df_raw['rsi_14'].values
output_ta = df_raw['output_ta'].values

number_ha_candle = 0
ha_status = None
ha_turn_list = []

for i in tqdm.tqdm(range(len(df_raw))):
    if i < 1200:
        continue

    filter_data = [df_raw.iloc[i].get(feat) for feat in list_features]
    input_data = np.array(filter_data)
    input_data = scaler.transform(input_data.reshape(1, -1))
    input_data = input_data[:, feat_indx]

    batch_x = input_data
    x_temp = np.reshape(batch_x, (9, 9))
    x_temp = np.stack((x_temp,) * 3, axis=-1)
    x_temp = np.expand_dims(x_temp, axis=0)
    pred = model.predict(x_temp)
    predicted_class = np.argmax(pred, axis=1)[0]
    confidence = np.max(pred, axis=1)[0] 
    if i == 1266 or i == 3867 or i==2903:
        print(1)

    status_candle_last_1 = 1 if eval(output_ta[i - 1])["ha_candle"]["Close"] - eval(output_ta[i - 1])["ha_candle"]["Open"] > 0 else 0
    status_candle_current = 1 if eval(output_ta[i])["ha_candle"]["Close"] - eval(output_ta[i])["ha_candle"]["Open"] > 0 else 0

    ha_candle_signal_up_condition = status_candle_last_1 == 0 and status_candle_current == 1
    ha_candle_signal_down_condition = status_candle_last_1 == 1 and status_candle_current == 0

    if ha_status is None or ha_status == status_candle_last_1:
        number_ha_candle += 1
    else:
        number_ha_candle = 1
    ha_status = status_candle_last_1

    if ha_candle_signal_up_condition:
        ha_turn_list.append(("up", i, eval(output_ta[i - 1])["ha_candle"]["Low"]))
    elif ha_candle_signal_down_condition:
        ha_turn_list.append(("down", i, eval(output_ta[i - 1])["ha_candle"]["High"]))

    ha_turn_list = ha_turn_list[-30:]
    
    main_trend, entry_direction, main_trend_start, main_trend_end, current_max_high, current_min_low = trend_analys(ha_turn_list, 30)
   
    if predicted_class in [0, 1] and 0.9 > confidence > 0.6 and i < len(df_raw) - 100 and i >=100:
        current_rsi = rsi_value[i]
        status_tp, sl, tp, pred = check_tp_sl_current(df_raw, i, predicted_class, output_ta, main_trend, current_max_high, current_min_low, main_trend_end, number_ha_candle, status_candle_current)

        # if i - current_entry > 5 or current_status != pred:

        current_entry = i
        current_status = pred
        number_ha_candle, ha_status = plot_candlestick(df_raw, i, predicted_class, status_tp, sl, tp, confidence, current_rsi, output_ta, main_trend, main_trend_start, main_trend_end, current_max_high, current_min_low)
        if status_tp == 1:
            count_tp += 1
            total_tp += tp
            profit += tp
        elif status_tp == 0:
            count_sl += 1
            total_sl += sl
            profit -= sl
        elif status_tp == -1:
            count_fail += 1
        else:
            count_equal += 1

        print(f"Profit: {profit} TP: {count_tp} SL: {count_sl}")
print(count_tp, count_sl, count_fail, count_equal)
print(total_tp, total_sl)
