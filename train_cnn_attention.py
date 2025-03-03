
import pandas as pd 
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, OneHotEncoder
from operator import itemgetter
from sklearn.feature_selection import SelectKBest, f_classif
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import  load_model
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from functools import *
from sklearn.metrics import f1_score
import os
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, Dropout, Flatten, Attention, Input, Reshape
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler


def build_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(32, return_sequences=True)(x)
    attention = Attention()([x, x])
    x = Flatten()(attention)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(3, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    return model

# use the path printed in above output cell after running stock_cnn.py. It's in below format
csv_tain = [f for f in os.listdir('data') if f.endswith('.csv') and 'features' not in f]
df_train = pd.concat([pd.read_csv(os.path.join('data', f)) for f in csv_tain])

df_train = df_train.dropna()
df_train['labels'] = df_train['labels'].astype("int")

df_train.drop(columns=["ha_open", "ha_high", "ha_low", "ha_close", "ha_type"], inplace=True)
# df_train.drop(columns=["ema_7","ema_14", "ema_17", "ema_21", "ema_25", "ema_34", "ema_89", "ema_50", "upperband", "lowerband"], inplace=True)

df_test = pd.read_csv("test/indicator_data_xau_table_m15_2024_15.csv")
# df_test.drop(columns=["output_ta"], inplace=True) 

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

# Reshape input for CNN/LSTM
X_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
X_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

# save scaler
folder_model_path = 'weights'
if not os.path.exists(folder_model_path):
    os.makedirs(folder_model_path)
np.save(os.path.join(folder_model_path, 'scaler.npy'), mm_scaler)

# Save list_features
np.save(os.path.join(folder_model_path, 'list_features.npy'), list_features)

print("Shape of x, y train/test {} {} {} {}".format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))

model = build_model((X_train.shape[1], 1))
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=200, batch_size=128, validation_data=(X_test, y_test))

# Load model and evaluate
model = load_model(best_model_path)
test_res = model.evaluate(x_test, y_test, verbose=0)
print("keras evaluate =", test_res)

# Predict and filter by confidence > 0.9
pred = model.predict(x_test)
pred_probs = np.max(pred, axis=1)  # Max probability for each prediction
confident_mask = pred_probs > 0.7   # Filter predictions with confidence > 0.9

# Get predicted and true classes with confidence > 0.9
pred_classes = np.argmax(pred, axis=1)[confident_mask]
y_test_classes = np.argmax(y_test, axis=1)[confident_mask]

print(f"Number of confident predictions: {len(pred_classes)}")

# Baseline check and metrics
# check_baseline(pred_classes, y_test_classes)
conf_mat = confusion_matrix(y_test_classes, pred_classes)
print(conf_mat)

# F1 scores
print("F1 score (weighted):", f1_score(y_test_classes, pred_classes, average='weighted'))
print("F1 score (macro):", f1_score(y_test_classes, pred_classes, average='macro'))
print("F1 score (micro):", f1_score(y_test_classes, pred_classes, average='micro'))

# Cohen's Kappa
print("Cohen's Kappa:", cohen_kappa_score(y_test_classes, pred_classes))

# Recall per class
recall = []
for i, row in enumerate(conf_mat):
    recall_value = np.round(row[i] / np.sum(row), 2) if np.sum(row) > 0 else 0.0
    recall.append(recall_value)
    print(f"Recall of class {i} = {recall_value}")

print("Recall avg =", sum(recall) / len(recall))

