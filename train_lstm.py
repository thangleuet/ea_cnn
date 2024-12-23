
import pandas as pd 
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
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
from models.model_lstm_utils import create_model_lstm
from sklearn.metrics import precision_score, recall_score

def get_sample_weights(y):
    """
    calculate the sample weights based on class weights. Used for models with
    imbalanced data and one hot encoding prediction.

    params:
        y: class labels as integers
    """

    y = y.astype(int)  # compute_class_weight needs int labels
    class_weights = compute_class_weight('balanced' , np.unique(y), y)
    
    print("real class weights are {}".format(class_weights), np.unique(y))
    print("value_counts", np.unique(y, return_counts=True))
    sample_weights = y.copy().astype(float)
    for i in np.unique(y):
        sample_weights[sample_weights == i] = class_weights[i]  if i == 2 else 1.5 * class_weights[i]
    return sample_weights

def check_baseline(pred, y_test):
    print("size of test set", len(y_test))
    e = np.equal(pred, y_test)
    print("TP class counts", np.unique(y_test[e], return_counts=True))
    print("True class counts", np.unique(y_test, return_counts=True))
    print("Pred class counts", np.unique(pred, return_counts=True))
    holds = np.unique(y_test, return_counts=True)[1][2]  # number 'hold' predictions
    print("baseline acc:", (holds/len(y_test)*100))


def create_sequences(x, y, timestep):
    x_seq, y_seq = [], []
    for i in range(len(x) - timestep):
        x_seq.append(x[i:i + timestep])  # Lấy 12 bước thời gian
        y_seq.append(y[i + timestep])  # Nhãn tương ứng với thời điểm tiếp theo
    return np.array(x_seq), np.array(y_seq)

# use the path printed in above output cell after running stock_cnn.py. It's in below format
csv_tain = [f for f in os.listdir('data') if f.endswith('.csv') and 'features' not in f]
df_train = pd.concat([pd.read_csv(os.path.join('data', f)) for f in csv_tain])
df_train.reset_index(drop=True, inplace=True)
df_train.drop(columns=["output_ta"], inplace=True) 
df_train.drop(columns=["candle_type"], inplace=True)
df_train.drop(columns=["ema_7","ema_14", "ema_17", "ema_21", "ema_25", "ema_34", "ema_89", "ema_50"], inplace=True)

df_train = df_train.dropna()
df_train['labels'] = df_train['labels'].astype("int")

df_test = pd.read_csv("test/indicator_data_xau_table_h1_2023_10.csv")
df_test.drop(columns=["output_ta"], inplace=True) 
df_test = df_test.dropna()
df_test['labels'] = df_test['labels'].astype("int")

df_train.reset_index(drop=True, inplace=True)

list_features = list(df_train.loc[:,"open":].columns)
# list_features = ["open", "high", "low", "close", "ema_34", "ema_89"]
# remove labels
list_features = [feature for feature in list_features if 'labels' not in feature]

# list_feature_drop = ['y_resistance_max', 'y_resistance_min', 'y_support_max', 'y_support_min', 'td_seq_ha_trend', 'td_seq_ha_number']
# list_feature_drop = [feature for feature in list_features if 'rsi' in feature]
# list_features = list(set(list_features) - set(list_feature_drop))
print('Total number of features', len(list_features))
x_train = df_train.loc[:, list_features].values
y_train = df_train['labels'].values

x_test = df_test.loc[:, list_features].values
y_test =  df_test['labels'].values


# mm_scaler = StandardScaler() # or StandardScaler?
mm_scaler = MinMaxScaler(feature_range=(0, 1))
x_train = mm_scaler.fit_transform(x_train)
x_test = mm_scaler.transform(x_test)

# save scaler
folder_model_path = 'weights_lstm'
if not os.path.exists(folder_model_path):
    os.makedirs(folder_model_path)
np.save(os.path.join(folder_model_path, 'scaler.npy'), mm_scaler)
np.save(os.path.join(folder_model_path, 'list_features.npy'), list_features)

x_main = x_train.copy()
print("Shape of x, y train/test {} {} {} {}".format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))

num_features = 49  # should be a perfect square
timesteps = 12
selection_method = 'all'
topk = 55 if selection_method == 'all' else num_features

# select_k_best = SelectKBest(f_classif, k=topk)
# select_k_best.fit(x_main, y_train)
# selected_features_anova = itemgetter(*select_k_best.get_support(indices=True))(list_features)
# print("****************************************")

# select_k_best = SelectKBest(mutual_info_classif, k=topk)
# select_k_best.fit(x_main, y_train)
# selected_features_mic = itemgetter(*select_k_best.get_support(indices=True))(list_features)

# if selection_method == 'all':
#     common = list(set(selected_features_anova).intersection(selected_features_mic))
#     if len(common) < num_features:
#         raise Exception('number of common features found {} < {} required features. Increase "topk variable"'.format(len(common), num_features))
#     feat_idx = []
#     for c in common:
#         feat_idx.append(list_features.index(c))
#     feat_idx = sorted(feat_idx[0:num_features])

# Save feat_idx
# np.save(os.path.join(folder_model_path, 'feat_idx.npy'), feat_idx)

# if selection_method == 'all':
#     x_train = x_train[:, feat_idx]
#     x_test = x_test[:, feat_idx]
num_features = x_train.shape[1]

x_train, y_train = create_sequences(x_train, y_train, timesteps)
x_test, y_test = create_sequences(x_test, y_test, timesteps)

_labels, _counts = np.unique(y_train, return_counts=True)
print("percentage of class 0 = {}, class 1 = {}".format(_counts[0]/len(y_train) * 100, _counts[1]/len(y_train) * 100))

sample_weights = get_sample_weights(y_train)

one_hot_enc = OneHotEncoder(sparse_output=False, categories='auto')  # , categories='auto'
y_train = one_hot_enc.fit_transform(y_train.reshape(-1, 1))
y_test = one_hot_enc.transform(y_test.reshape(-1, 1))

params = {
    'lstm_units': 64,
    'dropout_rate': 0.2,
    'dense_units': 32,
    'dense_dropout': 0.3,
    'optimizer': 'adam',
    'epochs': 200,
    'batch_size': 128,
}
model = create_model_lstm(params, timesteps,num_features, 3)
best_model_path = os.path.join(folder_model_path, 'best_model.h5')
# best_model_path = 'model_epoch_200.h5'

rlp = ReduceLROnPlateau(monitor='val_loss', factor=0.02, patience=20, verbose=1, mode='min',
                        min_delta=0.001, cooldown=1, min_lr=0.001)
mcp = ModelCheckpoint(best_model_path, monitor='val_f1_metric', verbose=1,
                      save_best_only=True, save_weights_only=False, mode='max', period=1)  # val_f1_metric

history = model.fit(x_train, y_train, epochs=params['epochs'], verbose=1,
                            batch_size=128, shuffle=False,
                            validation_data=(x_test, y_test),
                             callbacks=[mcp, rlp]
                            , sample_weight=sample_weights)

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['f1_metric'])
plt.plot(history.history['val_f1_metric'])

plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train_loss', 'val_loss', 'f1', 'val_f1'], loc='upper left')
plt.show()


# Load model and evaluate
model = load_model(best_model_path)
pred = model.predict(x_test)
pred_classes = np.argmax(pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

conf_mat = confusion_matrix(y_test_classes, pred_classes)
print("Confusion Matrix:", conf_mat)
print("F1 score (weighted):", f1_score(y_test_classes, pred_classes, average='weighted'))
print("Cohen's Kappa:", cohen_kappa_score(y_test_classes, pred_classes))

conf_mat = confusion_matrix(y_test_classes, pred_classes)
print(conf_mat)

precision = precision_score(y_test_classes, pred_classes, average=None)
recall = recall_score(y_test_classes, pred_classes, average=None)

# Print results
print("Precision for each class:", precision)
print("Recall for each class:", recall)

# Weighted precision and recall
precision_weighted = precision_score(y_test_classes, pred_classes, average='weighted')
recall_weighted = recall_score(y_test_classes, pred_classes, average='weighted')

print("Weighted Precision:", precision_weighted)
print("Weighted Recall:", recall_weighted)
