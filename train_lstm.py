import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import tqdm
# 1. Load dữ liệu
df = pd.read_csv(r"indicator_data_xau_table_m5_2024_10.csv")  # Thay bằng file dữ liệu của bạn
close_prices = df["close"].values.reshape(-1, 1)
low_prices = df["low"].values.reshape(-1, 1)
high_prices = df["high"].values.reshape(-1, 1)
# 2. Chuẩn hóa dữ liệu
scaler = MinMaxScaler()
close_prices_scaled = scaler.fit_transform(close_prices)

# 3. Tạo dữ liệu đầu vào cho LSTM
def create_dataset(data, time_step=10):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i : i + time_step, 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 24
X, y = create_dataset(close_prices_scaled, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)  # Reshape cho LSTM

# 4. Chia train/test
split = int(len(X) * 0.8)

X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# 5. Xây dựng mô hình LSTM
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(time_step, 1)),
    LSTM(64, return_sequences=False),
    Dense(32),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=256, validation_data=(X_test, y_test))

# 6. Dự đoán
acc = 0

win = 0
loss = 0
close_prices = df["Close"].values.reshape(-1, 1)[split:]
close_prices_scaled_realtime = scaler.transform(close_prices)
low_prices = df["Low"].values.reshape(-1, 1)[split:]
high_prices = df["High"].values.reshape(-1, 1)[split:]
for i in tqdm.tqdm(range(time_step, len(close_prices))):
    prd = model.predict(close_prices_scaled_realtime[i-time_step:i].reshape(1, time_step, 1), verbose=0)
    prd = scaler.inverse_transform(prd.reshape(-1, 1))
    if prd[0] - close_prices[i] > 1:
        for j in range(i, len(close_prices)):
            if close_prices[i] - low_prices[j] > 3:
                loss += 1
                break
            if high_prices[j] - close_prices[i] > 1:
                win += 1
                break
            
    elif prd[0] - close_prices[i] < -1:
        for j in range(i, len(close_prices)):
            if high_prices[j] - close_prices[i] > 3:
                loss += 1
                break
            if close_prices[i] - low_prices[j] > 1:
                win += 1
                break
            
print("Win: ", win)
print("Loss: ", loss)
print("Win rate: ", win/(win+loss)) 