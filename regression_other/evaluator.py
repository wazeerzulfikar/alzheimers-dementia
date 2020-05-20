from models import build_model
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn.model_selection import KFold
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD
from keras.losses import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import os

def evaluate_n(config, data):
    
    if config.mod_split=='none':
        evaluate_normal(config, data)
       
    elif config.mod_split=='human':
        evaluate_human(config, data)
    else:
        evaluate_kl(config, data)

def standard_scale(x_train, x_test):
    scalar = StandardScaler()
    scalar.fit(x_train)
    x_train = scalar.transform(x_train)
    x_test = scalar.transform(x_test)
    return x_train, x_test

def evaluate_normal(config, data):
    X = data['X']
    y = data['y']

    train_mse=[]
    val_mse=[]
    
    kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=42)
    fold=1
    for train_index, test_index in kf.split(X, [0]*len(X)):
        x_train = np.asarray(X[train_index])
        y_train = np.asarray(y[train_index])
        x_val = np.asarray(X[test_index])
        y_val = np.asarray(y[test_index])

        x_train, x_val = standard_scale(x_train, x_val)
        train_score, val_score = train_a_fold(fold, config, x_train, y_train, x_val, y_val)
        train_mse.append(train_score)
        val_mse.append(val_score)    
        fold+=1
    print("Train : MSE mean : ", np.mean(train_mse), " MSE std dev : ", np.std(train_mse))
    print("Val MSE mean : ", np.mean(val_mse), " MSE std dev : ", np.std(val_mse))

def train_a_fold(fold, config, x_train, y_train, x_val, y_val):
    model = build_model(config)
    loss = mean_squared_error
    optimizer = Adam(learning_rate=config.learning_rate)

    if config.loss=='mae':
        loss = mean_absolute_error
    if config.optimizer=='sgd':
        optimizer=SGD(learning_rate=config.learning_rate, momentum=0.9)

    model.compile(loss=mean_squared_error, optimizer=Adam(learning_rate=0.001))
    checkpoints = ModelCheckpoint(os.path.join(config.model_dir, config.dataset, 'fold_{}.h5'.format(fold)),
                                  monitor='val_loss', 
                                  verbose=0, 
                                  save_best_only=True,
                                  save_weights_only=False,
                                  mode='auto',
                                  period=1)
    model.fit(x_train,
              y_train,
              epochs=200,
              verbose=0,
              callbacks=[checkpoints],
              validation_data=(x_val, y_val))
    
    model = load_model(os.path.join(config.model_dir, config.dataset, 'fold_{}.h5'.format(fold)))
    train_score = model.evaluate(x_train, y_train, verbose=0)
    val_score = model.evaluate(x_val, y_val, verbose=0)
    
    print('Fold ', fold, ' Train mse:', train_score, ' Val mse:', val_score)
    return train_score, val_score