import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
import matplotlib.pyplot as plt

# Hàm chuẩn bị dữ liệu
def prepare_data(data, scaler_X=None, scaler_y=None, pca=None, target_column='Close', time_steps=30, variance_threshold=0.90, is_train=True):
    # Loại bỏ các cột không liên quan
    exclude_columns = ['id', 'volume', 'Date', 'date_time', target_column]
    feature_columns = [col for col in data.columns if col not in exclude_columns and data[col].dtype in [np.float64, np.int64]]
    X = data[feature_columns]
    y = data[target_column]
    
    # Kiểm tra dữ liệu
    if X.empty:
        raise ValueError("Không có cột số nào để huấn luyện mô hình.")
    if len(X) < 100:
        raise ValueError("Số lượng mẫu dữ liệu quá ít để huấn luyện mô hình.")
    
    # Xử lý giá trị thiếu
    X = X.fillna(X.mean())
    y = y.fillna(method='ffill')  # Điền giá trị trước đó cho Close
    
    if is_train:
        # Chuẩn hóa dữ liệu
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        
        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
        
        # Áp dụng PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        # Chọn số thành phần chính
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        print(f"Số lượng thành phần chính giữ lại ({variance_threshold*100}% phương sai): {n_components}")
        
        pca = PCA(n_components=n_components)
        X_pca_reduced = pca.fit_transform(X_scaled)
    else:
        X_scaled = scaler_X.transform(X)
        X_pca_reduced = pca.transform(X_scaled)
        y_scaled = scaler_y.transform(y.values.reshape(-1, 1))
    
    return X_pca_reduced, y_scaled, scaler_X, scaler_y, pca

# Hàm tạo sequences
def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# Hàm train và test
def train_and_test_lstm(data_folder, test_file, target_column='Close', time_steps=30, variance_threshold=0.90):
    # [Giữ nguyên phần load data, prepare data, train model như trước]
    
    train_data = pd.DataFrame()
    for filename in os.listdir(data_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_folder, filename)
            df = pd.read_csv(file_path)
            train_data = pd.concat([train_data, df], ignore_index=True)
    
    print(f"Đã load {len(os.listdir(data_folder))} file để train")
    
    X_train_pca, y_train_scaled, scaler_X, scaler_y, pca = prepare_data(
        train_data, None, None, None, target_column, time_steps, variance_threshold, is_train=True
    )
    
    X_train_seq, y_train_seq = create_sequences(X_train_pca, y_train_scaled, time_steps)
    
    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, input_shape=(time_steps, X_train_pca.shape[1])))
    model.add(Dropout(0.3))
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    history = model.fit(
        X_train_seq, y_train_seq,
        epochs=50,
        batch_size=64,
        validation_split=0.1,
        verbose=1,
    )
    
    # Load và chuẩn bị dữ liệu test
    test_data = pd.read_csv(test_file)
    X_test_pca, y_test_scaled, _, _, _ = prepare_data(
        test_data, scaler_X, scaler_y, pca, target_column, time_steps, variance_threshold, is_train=False
    )
    X_test_seq, y_test_seq = create_sequences(X_test_pca, y_test_scaled, time_steps)
    
    # Dự đoán
    y_pred_scaled = model.predict(X_test_seq)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test = scaler_y.inverse_transform(y_test_seq)
    
    # Đánh giá
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"\nTest Results:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    
    # Plot dữ liệu test
    plt.figure(figsize=(15, 6))
    plt.plot(y_test, label='Actual Close', color='blue')
    plt.plot(y_pred, label='Predicted Close', color='red', linestyle='--')
    plt.title('Actual vs Predicted Close Prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    # Thêm thông tin file test vào title nếu muốn
    plt.title(f'Actual vs Predicted Close Prices\nTest File: {os.path.basename(test_file)}')
    
    # Hiển thị plot
    plt.show()
    
    # Lưu plot nếu cần
    plt.savefig('prediction_plot.png')
    print("Đã lưu biểu đồ vào 'prediction_plot.png'")
    
    # Lưu mô hình và các đối tượng
    model.save('lstm_price_model.h5')
    joblib.dump(scaler_X, 'scaler_X.pkl')
    joblib.dump(scaler_y, 'scaler_y.pkl')
    joblib.dump(pca, 'pca.pkl')
    
    return model, scaler_X, scaler_y, pca, test_data, y_pred

if __name__ == "__main__":
    data_folder = 'data'
    test_file = 'indicator_data_xau_table_m5_2022_10.csv'
    
    print("=== Huấn luyện và kiểm tra LSTM dự đoán giá Close ===")
    lstm_model, scaler_X, scaler_y, pca, test_data, y_pred = train_and_test_lstm(
        data_folder,
        test_file,
        target_column='close',
        time_steps=30,
        variance_threshold=0.95
    )
    
    # In một số dự đoán mẫu
    print("\nMột số dự đoán mẫu (Actual vs Predicted):")
    time_steps = 30
    for i in range(min(5, len(y_pred))):
        actual = test_data['close'].iloc[time_steps + i]
        predicted = y_pred[i][0]
        print(f"Sample {i+1}: Actual = {actual:.4f}, Predicted = {predicted:.4f}")