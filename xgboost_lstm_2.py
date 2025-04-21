import os
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.decomposition import PCA  # Thêm PCA
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Đường dẫn thư mục chứa dữ liệu train và file test
TRAIN_DIR = 'data'
TEST_FILE = 'indicator_data_xau_table_m5_2022.csv'
TARGET = 'labels'
TIME_SERIES_COLS = ['open', 'high', 'low', 'close']
TIMESTEPS = 30
PCA_N_COMPONENTS = 0.95  # Giữ 95% phương sai (có thể đổi thành số cụ thể, ví dụ: 50)

# 1. Chuẩn bị dữ liệu cho LSTM
def prepare_lstm_data(df, time_series_cols, timesteps, target, scaler_ts=None):
    X_time_series = []
    y = []
    scaler_ts = StandardScaler() if scaler_ts is None else scaler_ts
    ts_data = scaler_ts.fit_transform(df[time_series_cols])
    for i in range(timesteps, len(df)):
        X_time_series.append(ts_data[i-timesteps:i])
        y.append(df[target].iloc[i])
        print(f"Processing row {i} of {len(df)}")
    X_time_series = np.array(X_time_series)
    y = np.array(y)
    return X_time_series, y, scaler_ts

# 2. Xây dựng và trích xuất đặc trưng từ LSTM
def build_lstm_model(timesteps, n_features, output_dim=8):
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=False, input_shape=(timesteps, n_features)))
    model.add(Dense(output_dim, activation='relu'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def extract_lstm_features(lstm_model, X_time_series):
    return lstm_model.predict(X_time_series)

# 3. Chia dữ liệu train/validation
def split_train_val(X_time_series, X_other_features, train_ratio=0.8):
    train_size = int(train_ratio * len(X_time_series))
    X_ts_train = X_time_series[:train_size]
    X_other_train = X_other_features[:train_size]
    y_train = y[:train_size]
    X_ts_val = X_time_series[train_size:]
    X_other_val = X_other_features[train_size:]
    y_val = y[train_size:]
    return X_ts_train, X_other_train, y_train, X_ts_val, X_other_val, y_val

# 4. Tính trọng số lớp
def compute_class_weights(y_train):
    class_counts = pd.Series(y_train).value_counts()
    n_samples = len(y_train)
    n_classes = len(class_counts)
    weights = {}
    for cls in class_counts.index:
        weights[cls] = 0.3 * n_samples / (n_classes * class_counts[cls]) if cls != 2 else 0.5
    sample_weights = np.array([weights[cls] for cls in y_train])
    return sample_weights

def plot_confusion_matrix(y_true, y_pred, save_dir='trading_signals'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Sell', 'Buy', 'Hold'], yticklabels=['Sell', 'Buy', 'Hold'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    filepath = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(filepath, bbox_inches='tight')
    print(f"Đã lưu confusion matrix: {filepath}")
    plt.show()
    plt.close()

# 5. Huấn luyện mô hình XGBoost
def train_model(X_train, y_train, X_val, y_val):
    sample_weights = compute_class_weights(y_train)
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        max_depth=9,
        learning_rate=0.05,
        n_estimators=300,
        lambda_=1,
        alpha=0.5,
        eval_metric='mlogloss'
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        sample_weight=sample_weights,
        verbose=True
    )
    return model

# 6. Đánh giá mô hình
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy trên tập test: {accuracy:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred)
    return y_pred

# 7. Xây dựng chiến lược giao dịch
def trading_strategy(df_test, y_pred, timesteps):
    df_test = df_test.iloc[timesteps:].copy()  # Cắt bỏ các hàng đầu không có dự đoán
    df_test['predicted_labels'] = y_pred
    df_test['profit'] = 0.0
    df_test['trade_outcome'] = None
    df_test['TP'] = 0.0
    df_test['SL'] = 0.0
    trades = []
    
    for i in range(len(df_test)):
        atr = df_test['atr'].iloc[i] if 'atr' in df_test.columns else 5.0
        entry_price = df_test['close'].iloc[i]
        TP = 10
        if df_test['predicted_labels'].iloc[i] == 1:  # Buy
            SL = max(5, abs(entry_price - df_test['low'].iloc[max(0, i-4):i].min()+1))
            df_test['TP'].iloc[i] = entry_price + TP
            df_test['SL'].iloc[i] = entry_price - SL
            for j in range(i+1, len(df_test)):
                if entry_price - df_test['low'].iloc[j] >= SL:
                    df_test['profit'].iloc[i] = -SL
                    df_test['trade_outcome'].iloc[i] = 'loss'
                    trades.append(0)
                    break
                elif df_test['high'].iloc[j] - entry_price >= TP:
                    df_test['profit'].iloc[i] = TP
                    df_test['trade_outcome'].iloc[i] = 'win'
                    trades.append(1)
                    break
        elif df_test['predicted_labels'].iloc[i] == 0:  # Sell
            SL = max(5, abs(entry_price - df_test['high'].iloc[max(0, i-4):i].max()+1))
            df_test['TP'].iloc[i] = entry_price - TP
            df_test['SL'].iloc[i] = entry_price + SL
            for j in range(i+1, len(df_test)):
                if df_test['high'].iloc[j] - entry_price >= SL:
                    df_test['profit'].iloc[i] = -SL
                    df_test['trade_outcome'].iloc[i] = 'loss'
                    trades.append(0)
                    break
                elif entry_price - df_test['low'].iloc[j] >= TP:
                    df_test['profit'].iloc[i] = TP
                    df_test['trade_outcome'].iloc[i] = 'win'
                    trades.append(1)
                    break
    
    if trades:
        winrate = sum(trades) / len(trades) * 100
        print(f"Winrate {len(trades)}: {winrate:.2f}%")
    else:
        print("Không có giao dịch nào được thực hiện.")
    
    df_test['cumulative_profit'] = df_test['profit'].cumsum()
    return df_test

# 8. Vẽ biểu đồ các điểm vào lệnh
def plot_trading_signals(df_test, save_dir='trading_signals'):
    # Tạo thư mục lưu ảnh nếu chưa tồn tại
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Đảm bảo df_test có các cột cần thiết cho nến
    required_columns = ['open', 'high', 'low', 'close']
    for col in required_columns:
        if col not in df_test.columns:
            df_test[col] = pd.read_csv(TEST_FILE)[col]
    
    # Đảm bảo index là DatetimeIndex
    if not isinstance(df_test.index, pd.DatetimeIndex):
        if 'timestamp' in df_test.columns:
            df_test['timestamp'] = pd.to_datetime(df_test['timestamp'])
            df_test = df_test.set_index('timestamp')
        else:
            raise ValueError("Index không phải DatetimeIndex và không có cột 'timestamp' để chuyển đổi.")
    
    # Chuẩn bị dữ liệu nến cho mplfinance
    df_test_mpf = df_test.copy()
    
    # Xác định các điểm mua (buy) và bán (sell)
    buy_signals = df_test[df_test['predicted_labels'] == 1].index
    sell_signals = df_test[df_test['predicted_labels'] == 0].index
    
    # Hàm vẽ biểu đồ nến cho một tín hiệu và lưu ảnh
    def plot_candlestick_signal(signal_index, signal_type, start_idx, end_idx, signal_count):
        # Lấy dữ liệu trong khoảng
        plot_data = df_test_mpf.loc[start_idx:end_idx]
        
        # Chuẩn bị dữ liệu cho tín hiệu
        buy_marker = pd.Series(index=plot_data.index, dtype=float)
        sell_marker = pd.Series(index=plot_data.index, dtype=float)
        
        outcome = df_test.loc[signal_index, 'trade_outcome'] if signal_index in df_test.index else None
        profit = df_test.loc[signal_index, 'profit'] if signal_index in df_test.index else 0.0
        
        # Xác định giá vào lệnh
        entry_price = df_test.loc[signal_index, 'close']
        candle_type = df_test.loc[signal_index, 'candle_type'] if 'candle_type' in df_test.columns else 'Unknown'
        
        if signal_index in plot_data.index:
            if signal_type == 'buy':
                buy_marker[signal_index] = df_test.loc[signal_index, 'low'] * 0.999
            else:
                sell_marker[signal_index] = df_test.loc[signal_index, 'high'] * 1.001
        
        # Tạo style cho biểu đồ
        mc = mpf.make_marketcolors(up='green', down='red', inherit=True)
        s = mpf.make_mpf_style(marketcolors=mc)
        
        # Tạo danh sách các điểm tín hiệu để thêm vào biểu đồ
        apds = []
        if signal_type == 'buy':
            apds.append(mpf.make_addplot(buy_marker, type='scatter', markersize=100, marker='^', color='green', label=f'Buy Signal {candle_type}'))
        if signal_type == 'sell':
            apds.append(mpf.make_addplot(sell_marker, type='scatter', markersize=100, marker='v', color='red', label=f'Sell Signal {candle_type}'))
        
        # Thêm EMA nếu có
        if 'ema_34' in plot_data.columns and 'ema_89' in plot_data.columns:
            apds.append(mpf.make_addplot(plot_data['ema_34'], type='line', color='blue', label='EMA 34'))
            apds.append(mpf.make_addplot(plot_data['ema_89'], type='line', color='orange', label='EMA 89'))
        
        # Thêm đường ngang cho TP hoặc SL
        tp_price = df_test.loc[signal_index, 'TP']
        sl_price = df_test.loc[signal_index, 'SL']
        
        tp_line = pd.Series(tp_price, index=plot_data.index)
        apds.append(mpf.make_addplot(tp_line, type='line', color='green', linestyle='--', label='Take Profit'))
        sl_line = pd.Series(sl_price, index=plot_data.index)
        apds.append(mpf.make_addplot(sl_line, type='line', color='red', linestyle='--', label='Stop Loss'))
        
        # Vẽ biểu đồ
        fig, ax = mpf.plot(
            plot_data[['open', 'high', 'low', 'close']],
            type='candle',
            style=s,
            title=f'{signal_type.capitalize()} Signal at {signal_index}',
            ylabel='Price',
            addplot=apds,
            figsize=(14, 8),
            returnfig=True
        )
        
        # Thêm chú thích cho TP/SL
        ax = ax[0]
        if outcome:
            outcome_text = f"{outcome.capitalize()} (Profit: {profit:.2f})"
            ax.annotate(
                outcome_text,
                xy=(0.05, 0.95),
                xycoords='axes fraction',
                fontsize=12,
                color='green' if outcome == 'win' else 'red',
                bbox=dict(facecolor='white', alpha=0.8)
            )
        
        # Thêm chú thích giá vào lệnh
        ax.annotate(
            f"Entry: {entry_price:.2f} \n TP: {df_test.loc[signal_index, 'TP']:.2f} \n SL: {df_test.loc[signal_index, 'SL']:.2f}",
            xy=(0.05, 0.80),
            xycoords='axes fraction',
            fontsize=10,
            color='black',
            bbox=dict(facecolor='white', alpha=0.8)
        )
        
        # Lưu biểu đồ
        signal_time_str = signal_index.strftime('%Y%m%d_%H%M%S')
        filename = f"{signal_type}_signal_{signal_count}_{signal_time_str}.png"
        filepath = os.path.join(save_dir, filename)
        fig.savefig(filepath, bbox_inches='tight')
        print(f"Đã lưu biểu đồ: {filepath}")
        plt.close(fig)
    
    # Đếm số lượng tín hiệu để đánh số file
    buy_count = 1
    sell_count = 1
    
    # Vẽ và lưu từng tín hiệu mua
    for buy_idx in buy_signals:
        idx_pos = df_test.index.get_loc(buy_idx)
        start_pos = max(0, idx_pos - 50)
        end_pos = min(len(df_test), idx_pos + 51)
        start_idx = df_test.index[start_pos]
        end_idx = df_test.index[end_pos - 1]
        plot_candlestick_signal(buy_idx, 'buy', start_idx, end_idx, buy_count)
        buy_count += 1
    
    # Vẽ và lưu từng tín hiệu bán
    for sell_idx in sell_signals:
        idx_pos = df_test.index.get_loc(sell_idx)
        start_pos = max(0, idx_pos - 50)
        end_pos = min(len(df_test), idx_pos + 51)
        start_idx = df_test.index[start_pos]
        end_idx = df_test.index[end_pos - 1]
        plot_candlestick_signal(sell_idx, 'sell', start_idx, end_idx, sell_count)
        sell_count += 1
# 9. Vẽ biểu đồ lợi nhuận
def plot_profit(df_test):
    plt.figure(figsize=(12, 6))
    plt.plot(df_test.index, df_test['cumulative_profit'], label='Cumulative Profit', color='purple')
    plt.title('Cumulative Profit Over Time')
    plt.xlabel('Time')
    plt.ylabel('Profit')
    plt.legend()
    plt.grid()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Tải và xử lý dữ liệu train
    print("Đang tải dữ liệu train từ thư mục:", TRAIN_DIR)
    csv_train = [f for f in os.listdir(TRAIN_DIR) if f.endswith('.csv') and 'features' not in f]
    df_train = pd.concat([pd.read_csv(os.path.join(TRAIN_DIR, f)) for f in csv_train])
    df_train = df_train.dropna()
    # Các đặc trưng không thuộc time series
    FEATURES = [
        'rsi_14', 'macd_diff', 'diff_ema_34', 'diff_ema_89', 'count_ema_34_ema_89',
        'RSI_Slope_LR', 'EMA34_Slope_LR', 'cci_20', 'body_size', 'candle_type', 'hour'
    ]
    # FEATURES = [
    #     'rsi_14', 'macd_diff', 'diff_ema_34', 'diff_ema_89', 'count_ema_34_ema_89', "diff_ema_34_89",
    #     'RSI_Slope_LR', 'EMA34_Slope_LR', 'cci_20', 'atr', 'body_size', 'adx_14', 'stochrsi_14', 'wr_14'
    # ]
        
    # Chuẩn bị dữ liệu cho LSTM
    X_time_series, y, scaler_ts = prepare_lstm_data(
        df_train, TIME_SERIES_COLS, TIMESTEPS, TARGET
    )
    X_other_features = df_train[FEATURES].iloc[TIMESTEPS:].values

    # Chia tập train/validation
    X_ts_train, X_other_train, y_train, X_ts_val, X_other_val, y_val = split_train_val(
        X_time_series, X_other_features
    )

    # Xây dựng và trích xuất đặc trưng từ LSTM
    print("Đang xây dựng và trích xuất đặc trưng từ LSTM...")
    lstm_model = build_lstm_model(timesteps=TIMESTEPS, n_features=len(TIME_SERIES_COLS))
    
    # Huấn luyện LSTM
    lstm_model.fit(X_ts_train, y_train, epochs=20, batch_size=64, verbose=1)
    
    # Trích xuất đặc trưng
    lstm_features_train = extract_lstm_features(lstm_model, X_ts_train)
    lstm_features_val = extract_lstm_features(lstm_model, X_ts_val)

    # Chuẩn hóa các đặc trưng khác
    scaler_other = StandardScaler()
    X_other_train_scaled = scaler_other.fit_transform(X_other_train)
    X_other_val_scaled = scaler_other.transform(X_other_val)

    # Kết hợp đặc trưng từ LSTM và các đặc trưng khác
    X_train_combined = np.hstack([lstm_features_train, X_other_train_scaled])
    X_val_combined = np.hstack([lstm_features_val, X_other_val_scaled])

    # Áp dụng PCA để giảm chiều
    print("Đang áp dụng PCA để giảm chiều dữ liệu...")
    pca = PCA(n_components=PCA_N_COMPONENTS)
    X_train_pca = pca.fit_transform(X_train_combined)
    X_val_pca = pca.transform(X_val_combined)
    print(f"Số chiều sau PCA: {X_train_pca.shape[1]} (giữ {PCA_N_COMPONENTS*100}% phương sai)")

    # Tải và xử lý dữ liệu test
    print("Đang tải dữ liệu test từ file:", TEST_FILE)
    test_df = pd.read_csv(TEST_FILE)
    X_ts_test, y_test, _ = prepare_lstm_data(
        test_df, TIME_SERIES_COLS, TIMESTEPS, TARGET, scaler_ts
    )
    X_other_test = test_df[FEATURES].iloc[TIMESTEPS:].values
    
    lstm_features_test = extract_lstm_features(lstm_model, X_ts_test)
    X_other_test_scaled = scaler_other.transform(X_other_test)
    X_test_combined = np.hstack([lstm_features_test, X_other_test_scaled])
    X_test_pca = pca.transform(X_test_combined)

    # Huấn luyện mô hình XGBoost
    print("Đang huấn luyện mô hình XGBoost...")
    model = train_model(X_train_pca, y_train, X_val_pca, y_val)

    # Đánh giá mô hình trên tập test
    print("Đánh giá mô hình trên tập test:")
    y_pred = evaluate_model(model, X_test_pca, y_test)

    # Xây dựng và đánh giá chiến lược giao dịch
    print("Áp dụng chiến lược giao dịch...")
    result_df = trading_strategy(test_df.copy(), y_pred, TIMESTEPS)

    # In kết quả lợi nhuận
    total_profit = result_df['cumulative_profit'].iloc[-1]
    print(f"Tổng lợi nhuận: {total_profit:.2f}")

    # Vẽ biểu đồ các điểm vào lệnh
    # plot_trading_signals(result_df)

    plot_profit(result_df)
    # Hiển thị tầm quan trọng của đặc trưng
    xgb.plot_importance(model)
    plt.show()