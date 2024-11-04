from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

def create_model_lstm(params, timesteps, size_feature, num_class):
    model = Sequential()
    model.add(LSTM(params['lstm_units'], input_shape=(timesteps, size_feature), return_sequences=False))
    model.add(Dropout(params['dropout_rate']))
    model.add(LSTM(params['lstm_units']))
    
    model.add(Dense(params['dense_units'], activation='relu'))
    model.add(Dropout(params['dense_dropout']))
    model.add(Dense(num_class, activation='softmax'))
    model.compile(optimizer=params['optimizer'], loss='categorical_crossentropy', metrics=['accuracy'])
    return model
