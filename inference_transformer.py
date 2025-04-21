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
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

def create_sequences(x, timestep):
    x_seq =[]
    for i in range(timestep, len(x)):
        x_seq.append(x[i-timestep:i])  # Lấy 12 bước thời gian
    return np.array(x_seq)

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, n_layers, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        self.input_linear = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(d_model, 3)  # 3 class: 0, 1, 2

    def forward(self, x):
        x = self.input_linear(x)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x[-1, :, :]
        x = self.fc(x)
        return x

# Class Dataset cho inference
class TimeSeriesDataset(Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

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

    # Define scatter plot data aligned with timestamps
    marker = '^' if predicted_class == 1 else 'v'
    color = 'green' if predicted_class == 1 else 'red'
    label = 'Buy' if predicted_class == 1 else 'Sell'

    # Initialize scatter data with NaNs
    scatter_data = np.full(len(df_candle), np.nan)
    scatter_data[100] = df_candle['close'].iloc[100]  # Target point for the scatter plot

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
    
    tp = 5
    sl = 5
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

    ax2.text(0.5, 0.5, f"Status: {status}", fontsize=15, ha='center', va='center')
    ax2.text(0.5, 0.4, f"Confidence: {confidence}", fontsize=15, ha='center', va='center')
    ax2.text(0.5, 0.3, f"Label: {labels}", fontsize=15, ha='center', va='center')

    output_folder_image = f"output/buy/{status}" if predicted_class == 1 else f"output/sell/{status}"
    os.makedirs(output_folder_image, exist_ok=True)
    output_image_path = os.path.join(
        output_folder_image, f"{time_stamp}_{index}_{round(confidence, 2)}.png"
    )
    plt.clf()

    return number_ha_candle, ha_status, status

def plot_predict(df_raw, start_index, end_index, list_predict, list_ema_7, list_ema_25, list_ema_34, list_ema_89, list_ema_200):
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
    
    for index, confidence, predicted_class, count_ha, status in list_predict:
        price = df_raw.iloc[index]['close']
        if predicted_class == 1:
            color = 'green'
        elif predicted_class == 0:
            color = 'red'
        else:
            color = 'blue'
        ax1.scatter([index-start_index], [price], color=color, s=100)
        ax1.text(index-start_index, df_raw.iloc[index]['low'], f"{round(count_ha, 2)} _ {status}", fontsize=12, ha='center', va='center', color=color)
    time_stamp = df_raw.iloc[start_index]['timestamp'].replace('-', '').replace(':', '').replace(' ', '')
        
    ax1.plot(list_ema_34, color='orange')
    ax1.plot(list_ema_89, color='blue')
    ax1.plot(list_ema_200, color='brown')
    
    output_folder_image = f"output/date"
    os.makedirs(output_folder_image, exist_ok=True)
    output_image_path = os.path.join(
        output_folder_image, f"{time_stamp}_{start_index}.png"
    )
    # Save the figure
    plt.savefig(output_image_path)
    plt.clf()
    

csv_path = r"indicator_data_xau_table_m15_2024_10.csv"
df_raw = pd.read_csv(csv_path)



list_features = np.load('weights_lstm/list_features.npy', allow_pickle='TRUE')

# Load scaler
scaler = np.load('weights_lstm/scaler.npy', allow_pickle='TRUE').item()

timestep = 24
input_dim = len(list_features)
d_model = 64
n_heads = 4
n_layers = 2
model = TransformerClassifier(input_dim, d_model, n_heads, n_layers)
best_model_path = os.path.join('weights_lstm', 'best_transformer_f1.pth') 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Chuẩn bị toàn bộ dữ liệu thành sequence
data = df_raw[list_features].values
data_scaled = scaler.transform(data)
sequences = create_sequences(data_scaled, timestep)

# Biến đổi dữ liệu thành tensor
dataset = TimeSeriesDataset(sequences)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Tải trọng số mô hình đã huấn luyện
checkpoint = torch.load(best_model_path, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

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
list_features_output = []
start_date = None
end_date = None
list_ema_7 = []
list_ema_25 = []
list_ema_34 = []
list_ema_89 = []
list_ema_200 = []
list_predict_class = []
current_index = 0
for i, batch in enumerate(tqdm.tqdm(dataloader)):
    index = i + timestep
    if  index >= len(df_raw) - 100 or index < 100:
        continue
    ema_7 = df_raw.iloc[index]['ema_7']
    ema_25 = df_raw.iloc[index]['ema_25']
    ema_34 = df_raw.iloc[index]['ema_34']
    ema_89 = df_raw.iloc[index]['ema_89']
    ema_200 = df_raw.iloc[index]['ema_200']
    list_ema_7.append(ema_7)
    list_ema_25.append(ema_25)
    list_ema_34.append(ema_34)
    list_ema_89.append(ema_89)
    list_ema_200.append(ema_200)
    
    current_price = df_raw.iloc[index]['close']
    
    diff_ema_34_89 = df_raw.iloc[index]['ema_34'] - df_raw.iloc[index]['ema_89']
    
    count_ema_34 = df_raw.iloc[index]['count_ema_34_ema_89']
    
    current_time = df_raw.iloc[index]['timestamp']
    if current_time == "2023-03-23 01:00:00":
        print("found it")
    # Dự đoán
    with torch.no_grad():
        input_tensor = batch.to(device)  # Shape: (batch_size, timestep, input_dim)
        pred = model(input_tensor)
        pred = torch.softmax(pred, dim=1).cpu().numpy()
        predicted_class = np.argmax(pred, axis=1)[0]
        confidence = float(np.max(pred, axis=1)[0])
    
    current_date = pd.to_datetime(df_raw.iloc[index]['timestamp']).dayofyear
    rsi = df_raw.iloc[index]['rsi_14']


    if start_date is None:
        start_date = current_date
        start_index = index
        
    diff_ema_7_25 = df_raw.iloc[index]['ema_7'] - df_raw.iloc[index]['ema_25']
    
    if confidence < 0.7:
        predicted_class = 2
    if predicted_class == 1 and (rsi > 40):
        predicted_class = 2
    if predicted_class == 0 and (rsi < 60):
        predicted_class = 2
    if abs(diff_ema_34_89) < 1 and count_ema_34 < 5:
        predicted_class = 2
        
    if predicted_class == 1:
        print("BUY")
    if predicted_class == 0:
        print("SELL")
        
    list_predict_class.append(predicted_class)
    list_predict_class = list_predict_class[-2:]
   
    if predicted_class in [0,1]:
        number_ha_candle, ha_status, status = plot_candlestick(df_raw, index, predicted_class, confidence)
        list_features_output.append([index, confidence, predicted_class, predicted_class, status])
        if "TP" in status:
            count_tp += 1
        elif "SL" in status:
            count_sl += 1
        else:
            count_fail += 1
    
        print(f"TP: {count_tp} SL: {count_sl}")
    current_index = index
    
    if current_date - start_date > 2:
        end_date = current_date
        end_index = index
        plot_predict(df_raw, start_index, end_index, list_features_output, list_ema_7, list_ema_25, list_ema_34, list_ema_89, list_ema_200) 
        list_features_output = [] 
        list_ema_7 = []
        list_ema_25 = []
        list_ema_34 = []
        list_ema_89 = []
        list_ema_200 = []
        list_upperband = []
        list_lowerband = []
        start_date = current_date
        start_index = index
        
        
print(count_tp, count_sl, count_fail, count_equal)
print("TP acc:", count_tp / (count_tp + count_sl) * 100)

