
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
from models.model_cnn_utils import create_model_cnn

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
        sample_weights[sample_weights == i] = class_weights[i] if i == 2 else 1.5 * class_weights[i]
    return sample_weights

def reshape_as_image(x, img_width, img_height):
    x_temp = np.zeros((len(x), img_height, img_width))
    for i in range(x.shape[0]):
        x_temp[i] = np.reshape(x[i], (img_height, img_width))

    return x_temp

# use the path printed in above output cell after running stock_cnn.py. It's in below format
df_train = pd.read_csv("data/features_data.csv")
df_train['labels'] = df_train['labels'].astype(np.int8)

df_test = pd.read_csv("test/features_test.csv")
df_test['labels'] = df_test['labels'].astype(np.int8)

list_features = list(df_train.loc[:, 'open':'bb_200'].columns)
# list_feature_drop = ['y_resistance_max', 'y_resistance_min', 'y_support_max', 'y_support_min']
# list_features = list(set(list_features) - set(list_feature_drop))

print('Total number of features', len(list_features))

x_train = df_train.loc[200:, list_features].values
y_train = df_train['labels'][200:].values

x_test = df_test.loc[:, list_features].values
y_test =  df_test['labels'].values

mm_scaler = StandardScaler() # or StandardScaler?
x_train = mm_scaler.fit_transform(x_train)
x_test = mm_scaler.transform(x_test)

# save scaler
folder_model_path = 'weights'
if not os.path.exists(folder_model_path):
    os.makedirs(folder_model_path)
np.save(os.path.join(folder_model_path, 'scaler.npy'), mm_scaler)

# Save list_features
np.save(os.path.join(folder_model_path, 'list_features.npy'), list_features)

x_main = x_train.copy()
print("Shape of x, y train/test {} {} {} {}".format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))

num_features = 81  # should be a perfect square
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

_labels, _counts = np.unique(y_train, return_counts=True)
print("percentage of class 0 = {}, class 1 = {}".format(_counts[0]/len(y_train) * 100, _counts[1]/len(y_train) * 100))

sample_weights = get_sample_weights(y_train)

one_hot_enc = OneHotEncoder(sparse=False, categories='auto')  # , categories='auto'
y_train = one_hot_enc.fit_transform(y_train.reshape(-1, 1))
y_test = one_hot_enc.transform(y_test.reshape(-1, 1))

dim = int(np.sqrt(num_features))
x_train = reshape_as_image(x_train, dim, dim)
x_test = reshape_as_image(x_test, dim, dim)
# adding a 1-dim for channels (3)
x_train = np.stack((x_train,) * 3, axis=-1)
x_test = np.stack((x_test,) * 3, axis=-1)
print("final shape of x, y train/test {} {} {} {}".format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))


params = {'batch_size': 80, 'conv2d_layers': {'conv2d_do_1': 0.2, 'conv2d_filters_1': 64, 'conv2d_kernel_size_1': 3, 'conv2d_mp_1': 0, 
                                               'conv2d_strides_1': 1, 'kernel_regularizer_1': 0.0, 'conv2d_do_2': 0.3, 
                                               'conv2d_filters_2': 64, 'conv2d_kernel_size_2': 3, 'conv2d_mp_2': 2, 'conv2d_strides_2': 1, 
                                               'kernel_regularizer_2': 0.0, 'layers': 'two'}, 
           'dense_layers': {'dense_do_1': 0.3, 'dense_nodes_1': 128, 'kernel_regularizer_1': 0.0, 'layers': 'one'},
           'epochs': 300, 'lr': 0.001, 'optimizer': 'adam'}



def check_baseline(pred, y_test):
    print("size of test set", len(y_test))
    e = np.equal(pred, y_test)
    print("TP class counts", np.unique(y_test[e], return_counts=True))
    print("True class counts", np.unique(y_test, return_counts=True))
    print("Pred class counts", np.unique(pred, return_counts=True))
    holds = np.unique(y_test, return_counts=True)[1][2]  # number 'hold' predictions
    print("baseline acc:", (holds/len(y_test)*100))

model = create_model_cnn(params)
best_model_path = os.path.join(folder_model_path, 'best_model.h5')

rlp = ReduceLROnPlateau(monitor='val_loss', factor=0.02, patience=20, verbose=1, mode='min',
                        min_delta=0.001, cooldown=1, min_lr=0.001)
mcp = ModelCheckpoint(best_model_path, monitor='val_f1_metric', verbose=1,
                      save_best_only=True, save_weights_only=False, mode='max', period=1)  # val_f1_metric

# mcp_periodic = ModelCheckpoint(
#     filepath='model_epoch_{epoch:02d}.h5', verbose=1, save_weights_only=False,
#     save_freq=10 * (len(x_train) // 128)  # Lưu mỗi 10 epoch
# )

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
test_res = model.evaluate(x_test, y_test, verbose=0)
print("keras evaluate =", test_res)

# Predict and filter by confidence > 0.9
pred = model.predict(x_test)
pred_probs = np.max(pred, axis=1)  # Max probability for each prediction
confident_mask = pred_probs > 0.8  # Filter predictions with confidence > 0.9

# Get predicted and true classes with confidence > 0.9
pred_classes = np.argmax(pred, axis=1)[confident_mask]
y_test_classes = np.argmax(y_test, axis=1)[confident_mask]

print(f"Number of confident predictions: {len(pred_classes)}")

# Baseline check and metrics
check_baseline(pred_classes, y_test_classes)
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

