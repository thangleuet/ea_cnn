import tensorflow as tf
from tensorflow.python import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Flatten
from keras import regularizers, optimizers
from keras import backend as K
from keras.utils import get_custom_objects
from models.metrics import f1_weighted, f1_metric

def create_model_cnn(timesteps=7, num_features=7):
    model = Sequential()
    
    conv2d_layer1 = Conv2D(32,
                           3,
                           strides=1,
                           kernel_regularizer=regularizers.l2(0), 
                           padding='same',activation="relu", use_bias=True,
                           kernel_initializer='glorot_uniform',
                           input_shape=(timesteps, num_features,1))
    model.add(conv2d_layer1)
        
    model.add(Dropout(0.2))
    conv2d_layer2 = Conv2D(64,
                            3,
                            strides=1,
                            kernel_regularizer=regularizers.l2(0),
                            padding='same',activation="relu", use_bias=True,
                            kernel_initializer='glorot_uniform')
    model.add(conv2d_layer2)
        
    model.add(MaxPool2D(pool_size=2))
    
    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(3, activation='softmax')) 
    
    optimizer = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', f1_metric])
    
    return model

get_custom_objects().update({"f1_metric": f1_metric, "f1_weighted": f1_weighted})