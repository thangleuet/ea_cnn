import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LayerNormalization, MultiHeadAttention, Dropout, Conv1D
from tensorflow.keras.layers import Concatenate, Add, GlobalAveragePooling1D, BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# 1. Load dữ liệu
df = pd.read_csv(r"indicator_data_xau_table_m5_2024_10.csv")  # Thay bằng file dữ liệu của bạn

# 2. Thêm các đặc trưng kỹ thuật
def add_technical_indicators(df):
    # Thêm các đặc trưng kỹ thuật phổ biến
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.copy()
    loss = delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = -loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['MA20_std'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['MA20'] + (df['MA20_std'] * 2)
    df['Lower_Band'] = df['MA20'] - (df['MA20_std'] * 2)
    
    # Tính toán biến động giá (Volatility)
    df['HL_Diff'] = df['High'] - df['Low']
    df['OC_Diff'] = abs(df['Open'] - df['Close'])
    df['Volatility'] = df['HL_Diff'].rolling(window=5).mean()
    
    # Xóa các hàng có giá trị NaN
    return df.dropna().reset_index(drop=True)

df = add_technical_indicators(df)

# Các cột đặc trưng sẽ sử dụng
feature_columns = ['open', 'high', 'low', 'Close', 
                  'MA5', 'MA20', 'MA50', 'RSI', 'MACD', 'Signal_Line',
                  'Upper_Band', 'Lower_Band', 'Volatility']

# 3. Chuẩn hóa dữ liệu
scaler_dict = {}
scaled_data = np.zeros((len(df), len(feature_columns)))

for i, col in enumerate(feature_columns):
    scaler = StandardScaler()  # Thay bằng StandardScaler
    values = df[col].values.reshape(-1, 1)
    scaled_data[:, i] = scaler.fit_transform(values).flatten()
    scaler_dict[col] = scaler

# Giữ riêng scaler cho giá đóng cửa để dự đoán ngược lại
close_price_scaler = scaler_dict['Close']

# 4. Tạo dữ liệu đầu vào cho Transformer
def create_dataset(data, time_step=10):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i : i + time_step])
        # Chỉ dự đoán giá đóng cửa (cột thứ 3)
        y.append(data[i + time_step, feature_columns.index('Close')])
    return np.array(X), np.array(y)

time_step = 48  # Tăng độ dài chuỗi thời gian để bắt được xu hướng dài hạn hơn
X, y = create_dataset(scaled_data, time_step)

# 5. Chia train/test/validation
train_split = int(len(X) * 0.7)
val_split = int(len(X) * 0.85)

X_train, y_train = X[:train_split], y[:train_split]
X_val, y_val = X[train_split:val_split], y[train_split:val_split]
X_test, y_test = X[val_split:], y[val_split:]

# 6. Xây dựng mô hình Transformer cải tiến
def transformer_encoder_block(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Multi-Head Attention
    attention_output = LayerNormalization(epsilon=1e-6)(inputs)
    attention_output = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(attention_output, attention_output)
    attention_output = Dropout(dropout)(attention_output)
    attention_output = Add()([inputs, attention_output])
    
    # Feed Forward Network
    ffn_output = LayerNormalization(epsilon=1e-6)(attention_output)
    ffn_output = Dense(ff_dim, activation="gelu")(ffn_output)  # GELU thay vì ReLU
    ffn_output = Dropout(dropout)(ffn_output)
    ffn_output = Dense(inputs.shape[-1])(ffn_output)
    ffn_output = Dropout(dropout)(ffn_output)
    
    return Add()([attention_output, ffn_output])

def convolutional_projection(inputs, filters):
    # 1D Convolutional layer to extract local patterns
    conv_output = Conv1D(filters=filters, kernel_size=3, padding="same", activation="relu")(inputs)
    conv_output = LayerNormalization(epsilon=1e-6)(conv_output)
    return conv_output

def build_improved_transformer_model(
    input_shape, 
    head_size=32, 
    num_heads=4, 
    ff_dim=128, 
    num_transformer_blocks=4, 
    mlp_units=[128, 64], 
    dropout=0.1, 
    mlp_dropout=0.2
):
    inputs = Input(shape=input_shape)
    
    # Convolutional projection for local pattern extraction
    conv_features = convolutional_projection(inputs, filters=64)
    
    # Apply multiple transformer encoder blocks
    x = conv_features
    for _ in range(num_transformer_blocks):
        x = transformer_encoder_block(x, head_size, num_heads, ff_dim, dropout)
    
    # Global features
    x = LayerNormalization(epsilon=1e-6)(x)
    x = GlobalAveragePooling1D()(x)
    
    # Apply final MLP layers
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(mlp_dropout)(x)
    
    outputs = Dense(1)(x)
    
    return Model(inputs, outputs)

# Thông số mô hình transformer cải tiến
input_shape = (time_step, len(feature_columns))
head_size = 32
num_heads = 8
ff_dim = 256
num_transformer_blocks = 4
mlp_units = [256, 128, 64]
mlp_dropout = 0.3
dropout = 0.2

model = build_improved_transformer_model(
    input_shape=input_shape,
    head_size=head_size,
    num_heads=num_heads,
    ff_dim=ff_dim,
    num_transformer_blocks=num_transformer_blocks,
    mlp_units=mlp_units,
    dropout=dropout,
    mlp_dropout=mlp_dropout,
)

# Sử dụng optimizer Adam với learning rate decay
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
model.summary()

# 7. Callbacks để cải thiện quá trình huấn luyện
model_checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

# 8. Huấn luyện mô hình
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=128,
    validation_data=(X_val, y_val),
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        model_checkpoint,
        reduce_lr
    ]
)

# 9. Đánh giá mô hình trên tập test
model.load_weights('best_model.h5')  # Load best model
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

# 10. Trực quan hóa quá trình huấn luyện
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss during Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('MAE during Training')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.tight_layout()
plt.savefig('training_history.png')
plt.close()

# 11. Dự đoán và đánh giá hiệu suất giao dịch
y_pred = model.predict(X_test)

# Chuyển dự đoán và giá trị thực về giá gốc
y_pred_original = close_price_scaler.inverse_transform(y_pred)
y_test_original = close_price_scaler.inverse_transform(y_test.reshape(-1, 1))

# Trực quan hóa dự đoán
plt.figure(figsize=(12, 6))
plt.plot(y_test_original, label='Actual')
plt.plot(y_pred_original, label='Predicted')
plt.title('Gold Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.savefig('prediction_results.png')
plt.close()

# 12. Đánh giá hiệu suất giao dịch
win = 0
loss = 0
no_trade = 0
threshold = 1.0  # Ngưỡng dự đoán để ra quyết định giao dịch

# Đánh giá hiệu suất giao dịch
real_close_prices = df['Close'].values[val_split + time_step:]
real_high_prices = df['High'].values[val_split + time_step:]
real_low_prices = df['Low'].values[val_split + time_step:]

for i in tqdm.tqdm(range(len(y_pred_original))):
    pred_price = y_pred_original[i][0]
    actual_price = real_close_prices[i]
    
    # Kiểm tra xu hướng tăng
    if pred_price - actual_price > threshold:
        # Tìm trong 5 nến tiếp theo
        look_ahead = min(5, len(real_close_prices) - i - 1)
        for j in range(1, look_ahead + 1):
            # Kiểm tra SL (Stop Loss)
            if actual_price - real_low_prices[i + j] > 3:
                loss += 1
                break
            # Kiểm tra TP (Take Profit)
            if real_high_prices[i + j] - actual_price > 1:
                win += 1
                break
            # Nếu đã xem xét tất cả các nến và không có kết quả
            if j == look_ahead:
                no_trade += 1
    
    # Kiểm tra xu hướng giảm
    elif actual_price - pred_price > threshold:
        # Tìm trong 5 nến tiếp theo
        look_ahead = min(5, len(real_close_prices) - i - 1)
        for j in range(1, look_ahead + 1):
            # Kiểm tra SL (Stop Loss)
            if real_high_prices[i + j] - actual_price > 3:
                loss += 1
                break
            # Kiểm tra TP (Take Profit)
            if actual_price - real_low_prices[i + j] > 1:
                win += 1
                break
            # Nếu đã xem xét tất cả các nến và không có kết quả
            if j == look_ahead:
                no_trade += 1
    else:
        no_trade += 1

total_trades = win + loss
win_rate = win / total_trades if total_trades > 0 else 0

print(f"Số lệnh thắng: {win}")
print(f"Số lệnh thua: {loss}")
print(f"Số lệnh không đạt ngưỡng giao dịch: {no_trade}")
print(f"Tỷ lệ thắng: {win_rate:.2f}")
print(f"Tổng số lệnh giao dịch: {total_trades}")
print(f"Tỷ lệ tín hiệu giao dịch: {total_trades/(total_trades+no_trade):.2f}")

# Tính toán ROI
trade_size = 100  # Giả sử mỗi lệnh đầu tư 100$
profit_per_win = 1  # TP là 1$
loss_per_loss = 3   # SL là 3$

total_profit = win * profit_per_win - loss * loss_per_loss
roi = (total_profit / (trade_size * total_trades)) * 100 if total_trades > 0 else 0

print(f"Tổng lợi nhuận: ${total_profit}")
print(f"ROI: {roi:.2f}%")

# 13. Lưu mô hình và scaler để sử dụng trong tương lai
model.save('gold_price_transformer_model.h5')

# Lưu scaler để sử dụng khi dự đoán dữ liệu mới
import pickle
with open('feature_scalers.pkl', 'wb') as file:
    pickle.dump(scaler_dict, file)