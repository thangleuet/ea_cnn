
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

df_train.drop(columns=["ema_7","ema_14", "ema_17", "ema_21", "ema_25", "ema_34", "ema_89", "ema_50", "ha_open", "ha_high", "ha_low", "ha_close"], inplace=True)

df_test = pd.read_csv("test/indicator_data_xau_table_h1_2024_0.005.csv")
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

x_main = x_train.copy()
print("Shape of x, y train/test {} {} {} {}".format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))

num_features = 25  # should be a perfect square

select_k_best = SelectKBest(f_classif, k=num_features)
select_k_best.fit(x_main, y_train)
selected_features_anova = itemgetter(*select_k_best.get_support(indices=True))(list_features)
print("****************************************")

if len(selected_features_anova) < num_features:
    raise Exception('number of common features found {} < {} required features. Increase "topk variable"'.format(len(selected_features_anova), num_features))
feat_idx = []
for c in selected_features_anova:
    feat_idx.append(list_features.index(c))
feat_idx = sorted(feat_idx[0:num_features])

# Save feat_idx
np.save(os.path.join(folder_model_path, 'feat_idx.npy'), feat_idx)

x_train = x_train[:, feat_idx]
x_test = x_test[:, feat_idx]
name_features = x_train[0]

# LSTM
timestep = 48
lstm_features = ['close', 'open', 'high', 'low', 'volume']
x_train_lstm = create_sequences(df_train[lstm_features].values, timestep)[100:]
x_train = x_train[timestep:]
y_train = y_train[timestep:]

x_test_lstm = create_sequences(df_test[lstm_features].values, timestep)[100:]
x_test = x_test[timestep:]
y_test = y_test[timestep:]

lstm_scaler = StandardScaler()
x_train_lstm = lstm_scaler.fit_transform(x_train_lstm.reshape(-1, len(lstm_features))).reshape(-1, timestep, len(lstm_features))
x_test_lstm = lstm_scaler.transform(x_test_lstm.reshape(-1, len(lstm_features))).reshape(-1, timestep, len(lstm_features))
# save scaler
np.save(os.path.join(folder_model_path, 'lstm_scaler.npy'), lstm_scaler)

_labels, _counts = np.unique(y_train, return_counts=True)
print("percentage of class 0 = {}, class 1 = {}".format(_counts[0]/len(y_train) * 100, _counts[1]/len(y_train) * 100))

sample_weights = get_sample_weights(y_train)

one_hot_enc = OneHotEncoder(sparse_output=False, categories='auto')  # , categories='auto'
y_train = one_hot_enc.fit_transform(y_train.reshape(-1, 1))
y_test = one_hot_enc.transform(y_test.reshape(-1, 1))

print("final shape of x, y train/test {} {} {} {}".format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))

def check_baseline(pred, y_test):
    print("size of test set", len(y_test))
    e = np.equal(pred, y_test)
    print("TP class counts", np.unique(y_test[e], return_counts=True))
    print("True class counts", np.unique(y_test, return_counts=True))
    print("Pred class counts", np.unique(pred, return_counts=True))
    holds = np.unique(y_test, return_counts=True)[1][2]  # number 'hold' predictions
    print("baseline acc:", (holds/len(y_test)*100))

attention_input = tf.keras.layers.Input(shape=(len(name_features),), name='Attention_Input')
attention = tf.keras.layers.Reshape((1, len(name_features)))(attention_input)  # Reshape for attention
x_attention = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=16)(attention, attention)
x_attention = tf.keras.layers.GlobalAveragePooling1D()(x_attention)
x_attention = tf.keras.layers.Dropout(0.3)(x_attention)
x_attention = tf.keras.layers.BatchNormalization()(x_attention)

# Combine LSTM and Attention outputs
x = tf.keras.layers.Dense(16, activation='relu')(x_attention)
x = tf.keras.layers.Dropout(0.3)(x)
output = tf.keras.layers.Dense(3, activation='softmax', name='Output')(x)

# Tạo model
model = tf.keras.Model(inputs=attention_input, outputs=output)
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy', f1_metric])

best_model_path = os.path.join(folder_model_path, 'best_model.h5')

rlp = ReduceLROnPlateau(monitor='val_loss', factor=0.02, patience=20, verbose=1, mode='min',
                        min_delta=0.001, cooldown=1, min_lr=0.001)
mcp = ModelCheckpoint(best_model_path, monitor='val_f1_metric', verbose=1,
                      save_best_only=True, save_weights_only=False, mode='max', period=1)  # val_f1_metric

history = model.fit(x_train, y_train, epochs=100, verbose=1,
                            batch_size=128, shuffle=False,
                            validation_data= (x_test, y_test),
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
# save image
plt.savefig(os.path.join(folder_model_path, 'model_loss.png'))

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

