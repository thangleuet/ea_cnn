

import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from keras.metrics import AUC, Precision, Recall
from models.metrics import f1_weighted, f1_metric
from keras.utils import get_custom_objects
from keras.layers import (
    GRU,
    LSTM,
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
import tensorflow_addons as tfa

def create_model_lstm(params, timesteps, num_features, num_classes):
    """
    Create an LSTM model based on given parameters.

    Parameters:
        params: Dictionary containing model parameters like 'lstm_units', 'dropout_rate', etc.
        timesteps: Number of timesteps (input sequence length).
        num_features: Number of features for each timestep.
        num_classes: Number of output classes (for classification).

    Returns:
        model: Compiled LSTM model.
    """
    # Tạo mô hình LSTM
    model = Sequential()

    # LSTM layer
    model.add(LSTM(128, input_shape=(timesteps, num_features), return_sequences=False, activation='tanh'))

    # Dropout để giảm overfitting
    model.add(Dropout(0.3))

    # Dense layer với số lượng đầu ra tương ứng số lớp
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))

    # Output layer với softmax để dự đoán 3 lớp
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile the model
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=params.get('learning_rate', 0.001)),
        metrics=[
            "accuracy",
            f1_metric,
        ],
    )
    return model

get_custom_objects().update({"f1_metric": f1_metric, "f1_weighted": f1_weighted})
