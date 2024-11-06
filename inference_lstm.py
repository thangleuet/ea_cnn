import os
import joblib
import numpy as np
import pandas as pd
import mplfinance as mpf
import tensorflow as tf
from tensorflow.python import keras
from keras.models import load_model

from models.model_lstm_utils import create_model_lstm

def create_sequences(x, timestep):
    x_seq = []
    for i in range(len(x) - timestep):
        x_seq.append(x[i:i + timestep])  # Lấy 12 bước thời gian
    return np.array(x_seq)

def check_tp_sl(df_raw, index, predicted_class):
    # Get the close price of the (index + 1) candle
    entry_price = df_raw.iloc[index+ 1]['close']

    status_tp = 0
    tp = 8 - abs(df_raw.iloc[index+ 1]['close'] - df_raw.iloc[index+ 1]['open'])

    for i in range(2, 21):
        if index + i < len(df_raw):  # Ensure we don't go out of bounds
            current_price = df_raw.loc[index + i]['close']
            if predicted_class == 1:
                sl = max(df_raw.iloc[index+1]['close'] - min(df_raw.iloc[index]['low'], df_raw.iloc[index+1]['low']) + 1, 5)
                # Filter 
                if df_raw.loc[index+1]['open'] - df_raw.loc[index+1]['close'] > 0:
                    status_tp = -1
                    break
                # if sl >= 8:
                #     status_tp = -1
                #     break


                if entry_price - df_raw.loc[index + i]['low'] >= sl:
                    if status_tp != 2:
                        status_tp = 0
                        break
                if df_raw.loc[index + i]['high'] - entry_price >= tp:
                    status_tp = 1
                    break
                if df_raw.loc[index + i]['high'] - entry_price >= 3:
                    status_tp = 2
            else:
                sl = max(max(df_raw.iloc[index]['high'], df_raw.iloc[index+1]['high']) - df_raw.iloc[index+1]['close'] + 1, 5)
                if df_raw.loc[index+1]['open'] - df_raw.loc[index+1]['close'] < 0:
                    status_tp = -1
                    break
                # if sl >= 8:
                #     status_tp = -1
                #     break
                if df_raw.loc[index + i]['high'] - entry_price >= sl:
                    if status_tp != 2:
                        status_tp = 0
                        break
                if entry_price - df_raw.loc[index + i]['low'] >= tp:
                    status_tp = 1
                    break
                if entry_price - df_raw.loc[index + i]['low'] >= 3:
                    status_tp = 2

    if tp <= 0:
        status_tp = -1
    return status_tp, sl, tp
def plot_candlestick(df_raw, index, predicted_class, status_tp, sl, tp):  
    df_candle = df_raw.iloc[index - 20:index + 20][['timestamp', 'open', 'high', 'low', 'close']]
    df_candle.set_index('timestamp', inplace=True)         
    # Create output folder based on prediction class (buy/sell)
    output_folder_image = f"output_lstm/buy/{status_tp}" if predicted_class == 1 else f"output_lstm/sell/{status_tp}"
    if not os.path.exists(output_folder_image):
        os.makedirs(output_folder_image)
    output_image_path = os.path.join(output_folder_image, f"{index}_{status_tp}_sl{sl}_tp{tp}.png")

    # Create scatter plot data aligned with timestamps
    marker = '^' if predicted_class == 1 else 'v'
    color = 'green' if predicted_class == 1 else 'red'
    label = 'Buy' if predicted_class == 1 else 'Sell'

    # Initialize an array of NaN with the same length as df_candle
    scatter_data = np.full(len(df_candle), np.nan)
    scatter_data[20] = df_candle['close'].iloc[20]

    # Add the scatter plot
    ap = [
        mpf.make_addplot(
            scatter_data, type='scatter', markersize=100, marker=marker, color=color, label=label
        )
    ]

    # Plot using mplfinance
    mpf.plot(
        df_candle, type='candle', style='charles',
        title='Predict Price', ylabel='Price', volume=False,
        addplot=ap, figratio=(12, 6), figscale=1.2,
        savefig=output_image_path
    )

csv_path = "test/features_test.csv"
df_raw = pd.read_csv(csv_path)
df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])

list_features = np.load('weights/list_features.npy', allow_pickle='TRUE')
input_data = df_raw.loc[:, list_features].values

# Load scaler
scaler = np.load('weights_lstm/scaler.npy', allow_pickle='TRUE').item()
input_data = scaler.transform(input_data)

# Load feat_indx
# feat_indx = np.load('weights_lstm/feat_idx.npy', allow_pickle='TRUE')
# input_data = input_data[:, feat_indx]

timesteps = 12
num_features = len(input_data[0])  # should be a perfect square
input_data = create_sequences(input_data, timesteps)

# Load model
params = {
    'lstm_units': 128,
    'dropout_rate': 0.3,
    'dense_units': 64,
    'dense_dropout': 0.3,
    'optimizer': 'adam',
    'epochs': 200,
    'batch_size': 128,
}
model = create_model_lstm(params, timesteps,num_features, 3)
best_model_path = os.path.join('weights_lstm', 'best_model.h5') 
# best_model_path = "model_epoch_100.h5"
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
for i in range(len(input_data)):
    batch_x = input_data[i]
    x_temp = np.expand_dims(batch_x, axis=0)
    pred = model.predict(x_temp)
    predicted_class = np.argmax(pred, axis=1)[0]
    confidence = np.max(pred, axis=1)[0] 
    if i == 141:
        print(1)
    if predicted_class in [0, 1] and confidence > 0.8 and i < len(df_raw) - 20 and i >= 20:
        status_tp, sl, tp = check_tp_sl(df_raw, i, predicted_class)

        if i - current_entry >= 5 or current_status != status_tp:
            current_entry = i
            current_status = status_tp
            plot_candlestick(df_raw, i, predicted_class, status_tp, sl, tp)
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

            print(f"Profit: {profit}")
print(count_tp, count_sl, count_fail, count_equal)
print(total_tp, total_sl)
