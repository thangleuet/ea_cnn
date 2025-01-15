
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from keras import layers, models
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sqlalchemy import create_engine

DB_HOST = "42.96.41.209"
DB_USER = "xttrade"

DB_PASSWORD ="Xttrade1234$"
DB_NAME = "XTTRADE"

db_url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
engine = create_engine(db_url, echo=False)

def get_candles_data_before(current_time, timestep):
        """Get data from database

        Returns:
            string: str: SQL Query
        """
        sql_query = f"""
            SELECT id, Open, High, Low, Close, ema34, ema89, output_ta 
            FROM exness_xau_usd_h1 
            WHERE date_time <= '{current_time}' 
            ORDER BY date_time DESC 
            LIMIT {timestep}
            """
            
        df_pd = pd.read_sql(sql_query, con=engine)
        return df_pd

df_raw = pd.read_csv("indicator_data_xau_valiation.csv")

features = df_raw.drop(columns=["labels"])
name_features = list(features.columns)
# remove entry_date_time
name_features = [feature for feature in name_features if 'entry_date_time' not in feature]
print(name_features)
timestep = 24

lstm_features = ['ema34', 'ema89', 'Open', 'Close', 'High', 'Low']

def create_sequences(data, lstm_features, timestep=12):
    sequences = []
    entry_times = data["entry_date_time"]
    for time in entry_times:
        df_candle = get_candles_data_before(time, timestep)
        sequences.append(df_candle[lstm_features].values)
    return np.array(sequences)

lstm_data = create_sequences(df_raw, lstm_features, timestep)
nn_data = df_raw[name_features].iloc[:].values
labels = df_raw["labels"].iloc[:].values

# Chuẩn hóa dữ liệu
scaler_lstm = StandardScaler()
lstm_data = scaler_lstm.fit_transform(lstm_data.reshape(-1, len(lstm_features))).reshape(-1, timestep, len(lstm_features))

scaler_nn = StandardScaler()
nn_data = scaler_nn.fit_transform(nn_data)

# Tách dữ liệu train, validation, test
X_lstm_train, X_lstm_test, X_nn_train, X_nn_test, y_train, y_test = train_test_split(
    lstm_data, nn_data, labels, test_size=0.2, random_state=42)

# Thiết kế mô hình
# LSTM Model
lstm_input = tf.keras.layers.Input(shape=(timestep, len(lstm_features)), name='LSTM_Input')
x_lstm = tf.keras.layers.LSTM(64, return_sequences=False)(lstm_input)
x_lstm = tf.keras.layers.Dense(32, activation='relu')(x_lstm)

# NN Model
nn_input = tf.keras.layers.Input(shape=(len(name_features),), name='NN_Input')
x_nn = tf.keras.layers.Dense(32, activation='relu')(nn_input)

# Kết hợp
x = tf.keras.layers.Concatenate()([x_lstm, x_nn])
x = tf.keras.layers.Dense(32, activation='relu')(x)
output = tf.keras.layers.Dense(3, activation='softmax', name='Output')(x)

# Tạo model
model = tf.keras.Model(inputs=[lstm_input, nn_input], outputs=output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint

# Callback lưu model tốt nhất dựa trên validation accuracy
checkpoint = ModelCheckpoint(
    "lstm_model.h5",  # Tên file lưu model
    monitor="val_accuracy",  # Tiêu chí đánh giá
    save_best_only=True,  # Chỉ lưu model tốt nhất
    mode="max",  # Tối đa hóa accuracy
    verbose=1,
)

# Huấn luyện
history = model.fit(
    [X_lstm_train, X_nn_train], y_train,
    validation_split=0.1,
    epochs=200,
    batch_size=64,
    callbacks=[checkpoint]
)
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train_loss', 'val_loss', 'accuracy', 'val_accuracy'], loc='upper left')
plt.show()
# save image
plt.savefig('model_loss.png')
# Đánh giá
test_loss, test_accuracy = model.evaluate([X_lstm_test, X_nn_test], y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Dự đoán trên tập test
y_pred_probs = model.predict([X_lstm_test, X_nn_test])
y_pred = np.argmax(y_pred_probs, axis=1)  # Lấy class dự đoán cao nhất

# Tính confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()