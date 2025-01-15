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
    
    tp = 0.005
    sl = 0.005
    entry_price = df_candle['close'].iloc[100]
    
    if predicted_class == 1:
        tp_price = entry_price + tp*entry_price
        sl_price = entry_price - sl*entry_price
        ax1.axhline(y= df_candle['close'].iloc[100] + tp*entry_price, color='green', linestyle='--', alpha=0.5)
        ax1.axhline(y= df_candle['close'].iloc[100] - sl*entry_price, color='red', linestyle='--', alpha=0.5)
    else:
        tp_price = entry_price - tp*entry_price
        sl_price = entry_price + sl*entry_price
        ax1.axhline(y= df_candle['close'].iloc[100] - tp*entry_price, color='green', linestyle='--', alpha=0.5)
        ax1.axhline(y= df_candle['close'].iloc[100] + sl*entry_price, color='red', linestyle='--', alpha=0.5)
    
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

def plot_predict(df_raw, start_index, end_index, list_predict, list_ema_7, list_ema_25, list_ema_34, list_ema_89, list_upperband, list_lowerband):
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
    
    for index, confidence, predicted_class, diff_ema_7_25, count_ha in list_predict:
        price = df_raw.iloc[index]['close']
        if predicted_class == 1:
            color = 'green'
        elif predicted_class == 0:
            color = 'red'
        else:
            color = 'blue'
        ax1.scatter([index-start_index], [price], color=color, s=100)
        ax1.text(index-start_index, df_raw.iloc[index]['low']-2, f"{round(count_ha, 2)} _ {round(diff_ema_7_25, 2)}", fontsize=12, ha='center', va='center', color=color)
    time_stamp = df_raw.iloc[start_index]['timestamp'].replace('-', '').replace(':', '').replace(' ', '')
        
    ax1.plot(list_ema_7, color='pink')
    ax1.plot(list_ema_25, color='purple')
    # ax1.plot(list_ema_34, color='orange')
    # ax1.plot(list_ema_89, color='blue')
    ax1.plot(list_upperband, color='red')
    ax1.plot(list_lowerband, color='green')
    
    output_folder_image = f"output/date"
    os.makedirs(output_folder_image, exist_ok=True)
    output_image_path = os.path.join(
        output_folder_image, f"{time_stamp}_{start_index}.png"
    )

    # Save the figure
    plt.savefig(output_image_path)
    plt.clf()
    

csv_path = r"indicator_data_xau_table_h1_2022_0.005.csv"
df_raw = pd.read_csv(csv_path)

list_features = np.load('weights/list_features.npy', allow_pickle='TRUE')

# Load scaler
scaler = np.load('weights/scaler.npy', allow_pickle='TRUE').item()

# Load feat_indx
feat_indx = np.load('weights/feat_idx.npy', allow_pickle='TRUE')

model = create_model_cnn()
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
list_ema_34 = []
list_ema_89 = []
list_predict_class = []
list_upperband = []
list_lowerband = []
current_index = 0
for index in tqdm.tqdm(range(len(df_raw))):
    if  index >= len(df_raw) - 100 or index < 100:
        continue
    filter_data = [df_raw.iloc[index].get(feat) for feat in list_features]
    input_data = np.array(filter_data)
    input_data = scaler.transform(input_data.reshape(1, -1))
    input_data = input_data[:, feat_indx]
    candlestick_pattern = df_raw.iloc[index]['candlestick_pattern']
    ema_7 = df_raw.iloc[index]['ema_7']
    ema_25 = df_raw.iloc[index]['ema_25']
    ema_34 = df_raw.iloc[index]['ema_34']
    ema_89 = df_raw.iloc[index]['ema_89']
    ema_50 = df_raw.iloc[index]['ema_50']
    list_ema_7.append(ema_7)
    list_ema_25.append(ema_25)
    list_ema_34.append(ema_34)
    list_ema_89.append(ema_89)
    
    current_price = df_raw.iloc[index]['close']

    ha_candle_status = df_raw.iloc[index]['streak_count']
    candle_type = df_raw.iloc[index]['ha_type']
    
    diff_ema_7_50 = df_raw.iloc[index]['ema_7'] - df_raw.iloc[index]['ema_25']
    
    count_ema_7 = df_raw.iloc[index]['count_ema_7_ema_25']
    count_ema_34 = df_raw.iloc[index]['count_ema_34_ema_89']
 
    batch_x = input_data
    x_temp = np.reshape(batch_x, (5, 5))
    x_temp = np.stack((x_temp,) * 1, axis=-1)
    x_temp = np.expand_dims(x_temp, axis=0)
    output = model(x_temp)
    # convert tensor to numpy
    output = output.numpy()
    pred = model.predict(x_temp)
    predicted_class = np.argmax(pred, axis=1)[0]
    confidence = float(np.max(pred, axis=1)[0])
    
    current_date = pd.to_datetime(df_raw.iloc[index]['timestamp']).dayofyear
    rsi = df_raw.iloc[index]['rsi_14']
    upperband = df_raw.iloc[index]['upperband']
    lowerband = df_raw.iloc[index]['lowerband']
    signal = df_raw.iloc[index]['signals']
    
    list_upperband.append(upperband)
    list_lowerband.append(lowerband)

    if start_date is None:
        start_date = current_date
        start_index = index
        
    # candlestick_pattern = eval(candlestick_pattern.replace('null', '2').replace('false', '0').replace('true', '1'))
    diff_ema_7_25 = df_raw.iloc[index]['ema_7'] - df_raw.iloc[index]['ema_25']
    
    if confidence < 0.7:
        predicted_class = 2
    if predicted_class == 1 and rsi > 40:
        predicted_class = 2
    if predicted_class == 0 and rsi < 60:
        predicted_class = 2
        
    if count_ema_7 < 10 or abs(ema_7 - ema_25)/current_price < 3/2000:
        predicted_class = 2
        
    list_predict_class.append(predicted_class)
    list_predict_class = list_predict_class[-2:]
    if predicted_class in [0, 1]:
            list_predict.append([index, confidence, predicted_class, diff_ema_7_25, count_ema_7])
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
        plot_predict(df_raw, start_index, end_index, list_predict, list_ema_7, list_ema_25, list_ema_34, list_ema_89, list_upperband, list_lowerband) 
        list_predict = [] 
        list_ema_7 = []
        list_ema_25 = []
        list_ema_34 = []
        list_ema_89 = []
        list_upperband = []
        list_lowerband = []
        start_date = current_date
        start_index = index
        
        
print(count_tp, count_sl, count_fail, count_equal)
print("TP acc:", count_tp / (count_tp + count_sl) * 100)

