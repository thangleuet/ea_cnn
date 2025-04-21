import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from statsmodels.tsa.arima.model import ARIMA

# 1. Đọc dữ liệu CSV
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.reset_index(drop=True, inplace=True)
    return df

# 2. Tính toán độ biến động bằng GARCH
def apply_garch(df):
    df['returns'] = df['close'].pct_change().dropna()
    garch_model = arch_model(df['returns'].dropna() * 100, vol='Garch', p=3, q=3)
    garch_fit = garch_model.fit(disp="off")
    df['garch_volatility'] = garch_fit.conditional_volatility
    return df

# 4. Chuẩn bị dữ liệu cho LSTM
def prepare_lstm_data(df, feature_cols, target_col, lookback=30):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feature_cols].dropna())
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i])
        y.append(df[target_col].values[i])
    
    return np.array(X), np.array(y), scaler

# 5. Xây dựng mô hình LSTM
def build_lstm(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# 6. Dự báo giá bằng LSTM
def train_and_forecast_lstm(df, feature_cols, target_col):
    df.dropna(inplace=True)  # Loại bỏ hàng chứa NaN
    X, y, scaler = prepare_lstm_data(df, feature_cols, target_col)
    
    X_train, X_test = X[1:-100], X[-100:]
    y_train, y_test = y[1:-100], y[-100:]
    
    model = build_lstm(X_train.shape[1:])
    model.fit(X_train, y_train, epochs=500, batch_size=64, verbose=1)
    
    model.save('lstm_model.h5')
    y_pred = model.predict(X_test)
    
    df.loc[df.index[-100:], 'lstm_forecast'] = y_pred.flatten()
    return df, model

# 7. Kết hợp tín hiệu từ GARCH, ARIMA & LSTM
def generate_trading_signal(df):
    df['signal'] = np.where(
        (df['lstm_forecast'] > df['close']) &
        (df['garch_volatility'] < df['garch_volatility'].rolling(10).mean()), 1, -1)
    return df

# 8. Backtest chiến lược
def backtest(df, tp=1, sl=3):
    df['buy_signal'] = (df['signal'] == 1)
    df['sell_signal'] = (df['signal'] == -1)

    results = []
    df = df[-100:]
    
    for i in range(len(df)):
        if df['buy_signal'].values[i]:
            entry_price = df['close'][i]
            for j in range(i + 1, len(df)):
                if df['low'][j] - entry_price <= -sl:
                    results.append(0)  # Thua
                    break
                elif df['high'][j] - entry_price >= tp:
                    results.append(1)  # Thắng
                    break

        elif df['sell_signal'][i]:
            entry_price = df['close'][i]
            for j in range(i + 1, len(df)):
                if entry_price - df['high'][j] <= -sl:
                    results.append(0)  # Thua
                    break
                elif entry_price - df['low'][j] >= tp:
                    results.append(1)  # Thắng
                    break

    winrate = np.mean(results) * 100
    print(f"Winrate: {winrate:.2f}%")
    return winrate

# === CHẠY TOÀN BỘ PIPELINE ===
file_path = "indicator_data_eur_table_m5_2024_10.csv"
df = load_data(file_path)
df = apply_garch(df)
# df = apply_arima(df)  # Thêm ARIMA
df, lstm_model = train_and_forecast_lstm(df, ['open', 'high', 'low', 'close', 'volume', 'garch_volatility'], 'close')
df = generate_trading_signal(df)
winrate = backtest(df)
