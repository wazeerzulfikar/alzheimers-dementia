import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD
from keras.losses import mean_absolute_error, mean_squared_error

def build_model(config):
    model = Sequential()
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    return model