
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
        sample_weights[sample_weights == i] = class_weights[i]  # if i == 2 else 0.8 * class_weights[i]
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
df_train = pd.read_csv("data/features_data.csv")
df_train['labels'] = df_train['labels'].astype(np.int8)

df_test = pd.read_csv("test/features_test.csv")
df_test['labels'] = df_test['labels'].astype(np.int8)

list_features = list(df_train.loc[:, 'open':'eom_200'].columns)
print('Total number of features', len(list_features))
x_train = df_train.loc[200:, 'open':'eom_200'].values
y_train = df_train['labels'][200:].values

x_test = df_test.loc[:, 'open':'eom_200'].values
y_test =  df_test['labels'].values


mm_scaler = StandardScaler() # or StandardScaler?
x_train = mm_scaler.fit_transform(x_train)
x_test = mm_scaler.transform(x_test)

# save scaler
folder_model_path = 'weights'
if not os.path.exists(folder_model_path):
    os.makedirs(folder_model_path)
np.save(os.path.join(folder_model_path, 'scaler.npy'), mm_scaler)

x_main = x_train.copy()
print("Shape of x, y train/test {} {} {} {}".format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))

num_features = 81  # should be a perfect square
timesteps = 12
selection_method = 'all'
topk = 150 if selection_method == 'all' else num_features

select_k_best = SelectKBest(f_classif, k=topk)
select_k_best.fit(x_main, y_train)
selected_features_anova = itemgetter(*select_k_best.get_support(indices=True))(list_features)
print("****************************************")

select_k_best = SelectKBest(mutual_info_classif, k=topk)
select_k_best.fit(x_main, y_train)
selected_features_mic = itemgetter(*select_k_best.get_support(indices=True))(list_features)

if selection_method == 'all':
    common = list(set(selected_features_anova).intersection(selected_features_mic))
    if len(common) < num_features:
        raise Exception('number of common features found {} < {} required features. Increase "topk variable"'.format(len(common), num_features))
    feat_idx = []
    for c in common:
        feat_idx.append(list_features.index(c))
    feat_idx = sorted(feat_idx[0:num_features])

# Save feat_idx
np.save(os.path.join(folder_model_path, 'feat_idx.npy'), feat_idx)

if selection_method == 'all':
    x_train = x_train[:, feat_idx]
    x_test = x_test[:, feat_idx]

x_train_seq, y_train_seq = create_sequences(x_train, y_train, timesteps)
x_test_seq, y_test_seq = create_sequences(x_test, y_test, timesteps)

_labels, _counts = np.unique(y_train, return_counts=True)
print("percentage of class 0 = {}, class 1 = {}".format(_counts[0]/len(y_train) * 100, _counts[1]/len(y_train) * 100))

sample_weights = get_sample_weights(y_train)

one_hot_enc = OneHotEncoder(sparse=False, categories='auto')  # , categories='auto'
y_train = one_hot_enc.fit_transform(y_train.reshape(-1, 1))
y_test = one_hot_enc.transform(y_test.reshape(-1, 1))

params = {
    'lstm_units': 128,
    'dropout_rate': 0.3,
    'dense_units': 64,
    'dense_dropout': 0.3,
    'optimizer': 'adam',
    'epochs': 200,
    'batch_size': 128,
}
model = create_model_lstm(params, timesteps,num_features, 3)
best_model_path = os.path.join(folder_model_path, 'best_model.h5')

rlp = ReduceLROnPlateau(monitor='val_loss', factor=0.02, patience=20, verbose=1, mode='min',
                        min_delta=0.001, cooldown=1, min_lr=0.001)
mcp = ModelCheckpoint(best_model_path, monitor='val_loss', verbose=1,
                      save_best_only=True, save_weights_only=False, mode='max', period=1)  # val_f1_metric

mcp_periodic = ModelCheckpoint(
    filepath='model_epoch_{epoch:02d}.h5', verbose=1, save_weights_only=False,
    save_freq=10 * (len(x_train) // 128)  # Lưu mỗi 10 epoch
)

history = model.fit(x_train, y_train, epochs=params['epochs'], verbose=1,
                            batch_size=128, shuffle=False,
                            validation_data=(x_test, y_test),
                             callbacks=[mcp, mcp_periodic, rlp]
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

