import os
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

def check_tp_sl_current(df_raw, index, predicted_class, current_rsi, output_ta):
    # Get the close price of the (index + 1) candle
    entry_price = df_raw.iloc[index]['close']

    status_tp = 0
    tp = 5
    sl = 5
    status_candle_last_1 = 1 if eval(output_ta[index - 1])["ha_candle"]["Close"] - eval(output_ta[index - 1])["ha_candle"]["Open"] > 0 else 0
    status_candle_current = 1 if eval(output_ta[index])["ha_candle"]["Close"] - eval(output_ta[index])["ha_candle"]["Open"] > 0 else 0

    if predicted_class == 1:
        if status_candle_last_1 == 0 and status_candle_current == 1:
            is_ha_candle_last1_turn_up = 1
        else:
            is_ha_candle_last1_turn_up = 0
        if is_ha_candle_last1_turn_up == 0:
            status_tp = -1
            predicted_class = -1
    else:
        if status_candle_last_1 == 1 and status_candle_current == 0:
            is_ha_candle_last1_turn_down = 1
        else:
            is_ha_candle_last1_turn_down = 0
        
        if is_ha_candle_last1_turn_down == 0:
            status_tp = -1
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

    return status_tp, sl, tp, predicted_class


def plot_candlestick(df_raw, index, predicted_class, status_tp, sl, tp, conf, current_rsi, output_ta): 

    time_stamp = df_raw.iloc[index]['timestamp'].replace('-', '').replace(':', '').replace(' ', '')
    df_candle = df_raw.iloc[index - 50:index + 50][['timestamp', 'open', 'high', 'low', 'close']]
    df_candle['timestamp'] = pd.to_datetime(df_candle['timestamp'])
    df_candle.set_index('timestamp', inplace=True)         

    output_ta = df_raw['output_ta'].values[index-50:index+50]

    ha_candle = []
    for ta in output_ta:
        ha = eval(ta)["ha_candle"]
        ha_candle.append(ha)

    ha_candle = pd.DataFrame(ha_candle, columns=['Open', 'High', 'Low', 'Close'], index=df_candle.index)

    # Create output folder based on prediction class (buy/sell)
    output_folder_image = f"output/buy/{status_tp}" if predicted_class == 1 else f"output/sell/{status_tp}"
    os.makedirs(output_folder_image, exist_ok=True)
    
    output_image_path = os.path.join(
        output_folder_image, f"{time_stamp}_{index}_{round(conf, 2)}_{current_rsi}"
        f"sl{sl}_tp{tp}.png"
    )

    # Define scatter plot data aligned with timestamps
    marker = '^' if predicted_class == 1 else 'v'
    color = 'green' if predicted_class == 1 else 'red'
    label = 'Buy' if predicted_class == 1 else 'Sell'

    # Initialize scatter data with NaNs
    scatter_data = np.full(len(df_candle), np.nan)
    scatter_data[50] = df_candle['close'].iloc[50]  # Target point for the scatter plot

    ha_scatter = np.full(len(ha_candle), np.nan)
    ha_scatter[50] = ha_candle['Close'].iloc[50]  # Adjusted point in ha_candle for visibility

    # First subplot for candlestick plot with scatter_data

    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(2, 1, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    plt.subplots_adjust(wspace=0.15, left=0.05, right=0.95, top=0.95, bottom=0.05)
    mc= mpf.make_marketcolors(up='green', down='red', edge={'up': 'green', 'down': 'red'}, wick={'up': 'green', 'down': 'red'}, volume={'up': 'green', 'down': 'red'})
    s = mpf.make_mpf_style(marketcolors=mc)
    
    mpf.plot(
        df_candle, type='candle', style='charles', ax = ax1,
        addplot=[mpf.make_addplot(scatter_data, type='scatter', markersize=100, marker=marker, color=color, label=label, ax=ax1)],
        volume=False)
    ax1.set_title('Predict Price')
    ax1.set_ylabel('Price')

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

for i in tqdm.tqdm(range(len(df_raw))):
    if i < 1200:
        continue

    filter_data = [df_raw.iloc[i-1].get(feat) for feat in list_features]
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
    if i == 1240 or i == 3867 or i==2903:
        print(1)

    if predicted_class in [0, 1] and 0.9 > confidence > 0.6 and i < len(df_raw) - 50 and i >=50:
        current_rsi = rsi_value[i]
        status_tp, sl, tp, pred = check_tp_sl_current(df_raw, i, predicted_class, current_rsi, output_ta)

        # if i - current_entry > 5 or current_status != pred:
        current_entry = i
        current_status = pred
        number_ha_candle, ha_status = plot_candlestick(df_raw, i, predicted_class, status_tp, sl, tp, confidence, current_rsi, output_ta)
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
