import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from sklearn.metrics import f1_score, confusion_matrix, cohen_kappa_score
from matplotlib import pyplot as plt
import os
import shap
from models.model_cnn_utils import create_model_cnn
from sklearn.feature_selection import SelectKBest, f_classif
from operator import itemgetter
from imblearn.over_sampling import SMOTE

# Các hàm hỗ trợ giữ nguyên
def create_sequences(x, y, timestep):
    x_seq, y_seq = [], []
    for i in range(len(x) - timestep):
        x_seq.append(x[i:i + timestep])
        y_seq.append(y[i + timestep])
    return np.array(x_seq), np.array(y_seq)

def get_sample_weights(y):
    y = y.astype(int)
    class_weights = compute_class_weight('balanced', np.unique(y), y)
    print("real class weights are {}".format(class_weights), np.unique(y))
    print("value_counts", np.unique(y, return_counts=True))
    sample_weights = y.copy().astype(float)
    for i in np.unique(y):
        sample_weights[sample_weights == i] = class_weights[i] if i == 2 else 0.8 * class_weights[i]
    return sample_weights

def reshape_as_image(x, img_width, img_height):
    x_temp = np.zeros((len(x), img_height, img_width))
    for i in range(x.shape[0]):
        x_temp[i] = np.reshape(x[i], (img_height, img_width))
    return x_temp

# Load và xử lý dữ liệu
csv_tain = [f for f in os.listdir('data') if f.endswith('.csv') and 'features' not in f]
df_train = pd.concat([pd.read_csv(os.path.join('data', f)) for f in csv_tain])
df_train = df_train.dropna()
df_train['label'] = df_train['label'].astype("int")

# drop timestamp
df_train = df_train.drop('timestamp', axis=1)

df_test = pd.read_csv("indicator_data_xau_table_m5_2024_10.csv")
df_test = df_test.dropna()
df_test['label'] = df_test['label'].astype("int")

list_features = list(df_train.loc[:,"close":].columns)
list_features = [feature for feature in list_features if 'label' not in feature]

df_train.reset_index(drop=True, inplace=True)
print('Total number of features', len(list_features))
x_train = df_train.loc[100:, list_features].values
y_train = df_train.loc[100:, 'label'].values
x_test = df_test.loc[100:, list_features].values
y_test = df_test.loc[100:, 'label'].values

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

folder_model_path = 'weights'
if not os.path.exists(folder_model_path):
    os.makedirs(folder_model_path)
np.save(os.path.join(folder_model_path, 'scaler.npy'), scaler)
np.save(os.path.join(folder_model_path, 'list_features.npy'), list_features)

print("Shape of x, y train/test {} {} {} {}".format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))


num_features = 25  # should be a perfect square

select_k_best = SelectKBest(f_classif, k=num_features)
select_k_best.fit(x_train, y_train)
selected_features_anova = itemgetter(*select_k_best.get_support(indices=True))(list_features)
print("****************************************")

if len(selected_features_anova) < num_features:
    raise Exception('number of common features found {} < {} required features. Increase "topk variable"'.format(len(selected_features_anova), num_features))
feat_idx = []
for c in selected_features_anova:
    feat_idx.append(list_features.index(c))
feat_idx = sorted(feat_idx[0:num_features])

np.save(os.path.join(folder_model_path, 'feat_idx.npy'), feat_idx)

# Cập nhật dữ liệu với features được chọn
x_train = x_train[:, feat_idx]
x_test = x_test[:, feat_idx]

# Chuẩn bị dữ liệu one-hot
one_hot_enc = OneHotEncoder(sparse_output=False)
y_train_onehot = one_hot_enc.fit_transform(y_train.reshape(-1, 1))
y_test_onehot = one_hot_enc.transform(y_test.reshape(-1, 1))

# Tiếp tục xử lý dữ liệu
_labels, _counts = np.unique(y_train, return_counts=True)
print("percentage of class 0 = {}, class 1 = {}".format(_counts[0]/len(y_train) * 100, _counts[1]/len(y_train) * 100))

sample_weights = get_sample_weights(y_train)

# Reshape dữ liệu cho CNN sau khi chọn feature
dim = int(np.sqrt(num_features))
x_train = reshape_as_image(x_train, dim, dim)
x_test = reshape_as_image(x_test, dim, dim)
x_train = np.stack((x_train,) * 1, axis=-1)
x_test = np.stack((x_test,) * 1, axis=-1)

print("final shape of x, y train/test {} {} {} {}".format(x_train.shape, y_train_onehot.shape, x_test.shape, y_test_onehot.shape))

# Huấn luyện mô hình CNN chính thức
model = create_model_cnn()
best_model_path = os.path.join(folder_model_path, 'best_model.h5')

rlp = ReduceLROnPlateau(monitor='val_loss', factor=0.02, patience=20, verbose=1, mode='min', min_delta=0.001, cooldown=1, min_lr=0.001)
mcp = ModelCheckpoint(best_model_path, monitor='val_f1_metric', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=1)

history = model.fit(x_train, y_train_onehot, epochs=100, verbose=1, batch_size=128, shuffle=False,
                    validation_data=(x_test, y_test_onehot), callbacks=[mcp, rlp], sample_weight=sample_weights)

# Vẽ biểu đồ
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

# Đánh giá mô hình
model = load_model(best_model_path)
test_res = model.evaluate(x_test, y_test_onehot, verbose=0)
print("keras evaluate =", test_res)

pred = model.predict(x_test)
pred_probs = np.max(pred, axis=1)
confident_mask = pred_probs > 0.7
pred_classes = np.argmax(pred, axis=1)[confident_mask]
y_test_classes = np.argmax(y_test_onehot, axis=1)[confident_mask]

print(f"Number of confident predictions: {len(pred_classes)}")
conf_mat = confusion_matrix(y_test_classes, pred_classes)
print("Confusion Matrix:\n", conf_mat)

print("F1 score (weighted):", f1_score(y_test_classes, pred_classes, average='weighted'))
print("F1 score (macro):", f1_score(y_test_classes, pred_classes, average='macro'))
print("F1 score (micro):", f1_score(y_test_classes, pred_classes, average='micro'))
print("Cohen's Kappa:", cohen_kappa_score(y_test_classes, pred_classes))

recall = []
for i, row in enumerate(conf_mat):
    recall_value = np.round(row[i] / np.sum(row), 2) if np.sum(row) > 0 else 0.0
    recall.append(recall_value)
    print(f"Recall of class {i} = {recall_value}")
print("Recall avg =", sum(recall) / len(recall))