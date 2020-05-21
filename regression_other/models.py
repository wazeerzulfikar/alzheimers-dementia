import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import mean_absolute_error, mean_squared_error

def build_model(config):
    model = Sequential()
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    return model