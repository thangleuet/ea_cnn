
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

df_raw = pd.read_csv("indicator_data_xau_valiation.csv")
df_raw.drop(columns=["entry_date_time"], inplace=True)

features = df_raw.drop(columns=["labels"])
labels = df_raw["labels"]
name_features = list(features.columns)

x_train, x_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels
)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

y_train_tensor = tf.convert_to_tensor(y_train.values, dtype=tf.int64)  # Cần dtype là int64 cho classification
y_test_tensor = tf.convert_to_tensor(y_test.values, dtype=tf.int64)

x_train_reshaped = x_train_scaled.reshape(x_train_scaled.shape[0], x_train_scaled.shape[1], 1)
x_test_reshaped = x_test_scaled.reshape(x_test_scaled.shape[0], x_test_scaled.shape[1], 1)

x_train_tensor = tf.convert_to_tensor(x_train_reshaped, dtype=tf.float32)
x_test_tensor = tf.convert_to_tensor(x_test_reshaped, dtype=tf.float32)

model = models.Sequential([
    layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(x_train_tensor.shape[1], 1)),  
    layers.MaxPooling1D(pool_size=2),  
    layers.Conv1D(filters=64, kernel_size=3, activation='relu'),  
    layers.MaxPooling1D(pool_size=2),  
    layers.Flatten(),  
    layers.Dense(64, activation='relu'),  
    layers.Dense(32, activation='relu'), 
    layers.Dense(2, activation='softmax')  
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  
              metrics=['accuracy'])

# Xem tóm tắt mô hình
model.summary()

checkpoint = ModelCheckpoint(
    "cnn_model_confirm.h5",  
    monitor="val_loss",  # Tiêu chí để theo dõi
    save_best_only=True,  # Chỉ lưu model tốt nhất
    mode="min",  # Lưu khi giá trị nhỏ hơn là tốt hơn
    verbose=1
)

# Huấn luyện model với callback
history = model.fit(
    x_train_tensor,
    y_train_tensor,
    epochs=200,
    batch_size=64,
    validation_split=0.1,
    callbacks=[checkpoint]  # Thêm callback vào đây
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

# Đánh giá mô hình
model = load_model("cnn_model_confirm.h5")
test_loss, test_accuracy = model.evaluate(x_test_tensor, y_test_tensor)

print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy*100:.2f}%')

y_pred = model.predict(x_test_tensor)

# Lấy lớp dự đoán (class with the highest probability)
y_pred_class = np.argmax(y_pred, axis=1)

# Tính confusion matrix
cm = confusion_matrix(y_test_tensor, y_pred_class)

# In confusion matrix
print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
