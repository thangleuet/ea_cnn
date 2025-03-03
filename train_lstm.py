
import pandas as pd 
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, OneHotEncoder
from operator import itemgetter
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import tensorflow as tf
from tensorflow.python import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, Callback
from keras.models import Sequential, load_model
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from functools import *
from sklearn.metrics import f1_score
import os
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from matplotlib import pyplot as plt
from keras.optimizers import Adam
from keras.utils import get_custom_objects
from models.metrics import f1_weighted, f1_metric
import torch
from torch.utils.data import Dataset, DataLoader

get_custom_objects().update({"f1_metric": f1_metric, "f1_weighted": f1_weighted})

def create_sequences(x, timestep):
    x_seq =[]
    for i in range(timestep, len(x)):
        x_seq.append(x[i-timestep:i])  # Lấy 12 bước thời gian
    return np.array(x_seq)

def get_sample_weights(y):
    """
    calculate the sample weights based on class weights. Used for models with
    imbalanced data and one hot encoding prediction.

    params:
        y: class labels as integers
    """

    y = y.astype(int)  # compute_class_weight needs int labels
    class_weights = compute_class_weight('balanced', np.unique(y), y)
    
    print("real class weights are {}".format(class_weights), np.unique(y))
    print("value_counts", np.unique(y, return_counts=True))
    sample_weights = y.copy().astype(float)
    for i in np.unique(y):
        sample_weights[sample_weights == i] = class_weights[i] if i == 2 else 1 * class_weights[i]
    return sample_weights

# use the path printed in above output cell after running stock_cnn.py. It's in below format
csv_tain = [f for f in os.listdir('data') if f.endswith('.csv') and 'features' not in f]
df_train = pd.concat([pd.read_csv(os.path.join('data', f)) for f in csv_tain])
df_train.drop(columns=["output_ta"], inplace=True) 

df_train = df_train.dropna()
df_train['labels'] = df_train['labels'].astype("int")

df_train.drop(columns=["ha_open", "ha_high", "ha_low", "ha_close", "ha_type"], inplace=True)

df_test = pd.read_csv("test/indicator_data_xau_table_h1_2024_7.csv")
df_test.drop(columns=["output_ta"], inplace=True) 

df_test = df_test.dropna()
df_test['labels'] = df_test['labels'].astype("int")

list_features = list(df_train.loc[:,"close":].columns)
# remove labels
list_features = [feature for feature in list_features if 'labels' not in feature]

df_train.reset_index(drop=True, inplace=True)
print('Total number of features', len(list_features))
x_train = df_train.loc[100:, list_features].values
y_train = df_train.loc[100:, 'labels'].values

x_test = df_test.loc[100:, list_features].values
y_test =  df_test.loc[100:, 'labels'].values

mm_scaler = StandardScaler()
x_train = mm_scaler.fit_transform(x_train)
x_test = mm_scaler.transform(x_test)

# save scaler
folder_model_path = 'weights_lstm'
if not os.path.exists(folder_model_path):
    os.makedirs(folder_model_path)
np.save(os.path.join(folder_model_path, 'scaler.npy'), mm_scaler)

# Save list_features
np.save(os.path.join(folder_model_path, 'list_features.npy'), list_features)

window_size = 11
X_seq_train, y_seq_train = [], []
for i in range(len(x_train) - window_size + 1):
    X_seq_train.append(x_train[i:i + window_size])
    y_seq_train.append(y_train[i + window_size - 1])  # Nhãn của điểm cuối cửa sổ

X_seq_train = np.array(X_seq_train)
y_seq_train = np.array(y_seq_train)

X_seq_test, y_seq_test = [], []
for i in range(len(x_test) - window_size + 1):
    X_seq_test.append(x_test[i:i + window_size])
    y_seq_test.append(y_test[i + window_size - 1])  # Nhãn của điểm cuối cửa sổ

X_seq_test = np.array(X_seq_test)
y_seq_test = np.array(y_seq_test)

# Tính trọng số class
class_weights = compute_class_weight('balanced', classes=[0, 1, 2], y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TimeSeriesDataset(X_seq_train, y_seq_train)
test_dataset = TimeSeriesDataset(X_seq_test, y_seq_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

import torch.nn as nn

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, n_layers, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # Layer để chuyển đổi đầu vào sang d_model
        self.input_linear = nn.Linear(input_dim, d_model)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Đầu ra phân loại
        self.fc = nn.Linear(d_model, 3)  # 3 class: 0, 1, 2

    def forward(self, x):
        # x shape: (batch_size, window_size, input_dim)
        x = self.input_linear(x)  # (batch_size, window_size, d_model)
        x = x.permute(1, 0, 2)    # (window_size, batch_size, d_model) cho Transformer
        x = self.transformer_encoder(x)
        x = x[-1, :, :]           # Lấy output của timestep cuối (batch_size, d_model)
        x = self.fc(x)            # (batch_size, 3)
        return x
    
# Khởi tạo mô hình
input_dim = x_train.shape[1]
d_model = 64              # Kích thước embedding
n_heads = 4               # Số đầu chú ý
n_layers = 2              # Số layer Transformer
model = TransformerClassifier(input_dim, d_model, n_heads, n_layers)

from torch.optim import Adam
from sklearn.metrics import f1_score

# Thiết lập thiết bị
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
class_weights_tensor = class_weights_tensor.to(device)

# Loss và optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = Adam(model.parameters(), lr=0.001)

# Huấn luyện
n_epochs = 100
for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f'Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(train_loader):.4f}')

# Đánh giá
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(y_batch.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

f1 = f1_score(y_true, y_pred, average='weighted')
print(f'F1-Score: {f1:.4f}')