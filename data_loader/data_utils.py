import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from operator import itemgetter

def load_data(train_path, test_path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_train['labels'] = df_train['labels'].astype(np.int8)
    df_test['labels'] = df_test['labels'].astype(np.int8)
    return df_train, df_test

def preprocess_data(df, features):
    x = df.loc[:, features].values
    y = df['labels'].values
    return x, y
def select_features(x_data, y_data, list_features):
    num_features = 225
    topk = 250
    select_k_best = SelectKBest(f_classif, k=topk)
    select_k_best.fit(x_data, y_data)
    selected_features_anova = itemgetter(*select_k_best.get_support(indices=True))(list_features)

    select_k_best = SelectKBest(mutual_info_classif, k=topk)
    select_k_best.fit(x_data, y_data)
    selected_features_mic = itemgetter(*select_k_best.get_support(indices=True))(list_features)

    common = list(set(selected_features_anova).intersection(selected_features_mic))
    if len(common) < num_features:
        raise Exception('number of common features found {} < {} required features. Increase "topk variable"'.format(len(common), num_features))
    feat_idx = []
    for c in common:
        feat_idx.append(list_features.index(c))
    # Save feat_idx
    
    return feat_idx


def scale_data(x_train, x_test):
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Save scaler
    np.save('scaler.npy', scaler)
    return x_train, x_test

def reshape_as_image(x, width, height):
    x_temp = np.zeros((len(x), height, width))
    for i in range(x.shape[0]):
        x_temp[i] = np.reshape(x[i], (height, width))
    return np.stack((x_temp,) * 3, axis=-1)

def one_hot_encode(y_train, y_test):
    encoder = OneHotEncoder(sparse=False)
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test = encoder.transform(y_test.reshape(-1, 1))
    return y_train, y_test
