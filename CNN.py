import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

import numpy as np
import cv2
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

import glob
import os


dataset_dir = '../spectograms/'
cc_files = sorted(glob.glob(os.path.join(dataset_dir, 'cc-images/*.png')))
X_cc = np.array([cv2.resize(cv2.imread(f), (320,240))/255. for f in cc_files])
y_cc = np.zeros((X_cc.shape[0], 2))
y_cc[:,0] = 1

cd_files = sorted(glob.glob(os.path.join(dataset_dir, 'cd-images/*.png')))
X_cd = np.array([cv2.resize(cv2.imread(f), (320,240))/255. for f in cd_files])
y_cd = np.zeros((X_cd.shape[0], 2))
y_cd[:,1] = 1

X = np.concatenate((X_cc, X_cd), axis=0).astype(np.float32)
y = np.concatenate((y_cc, y_cd), axis=0).astype(np.float32)

p = np.random.permutation(len(X))
X = X[p]
y = y[p]

inp_shape = X_cc[0].shape
num_classes = 2

def create_model():
    model = tf.keras.Sequential()
    model.add(layers.Input(inp_shape))
    model.add(layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu'))
    model.add(layers.Conv2D(32, kernel_size=(3, 3), strides=(2, 2),
                     activation='relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu'))
    model.add(layers.Conv2D(64, kernel_size=(3, 3), strides=(2, 2),
                     activation='relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu'))
    model.add(layers.Conv2D(128, kernel_size=(3, 3), strides=(2, 2),
                     activation='relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu'))
    model.add(layers.Conv2D(256, kernel_size=(3, 3), strides=(2, 2),
                     activation='relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

model = create_model()
print(model.summary())
# training

n_split = 5
epochs = 30
batch_size = 4

for train_index, val_index in KFold(n_split, shuffle=True).split(X):

    x_train, x_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    model = create_model()

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['categorical_accuracy'])
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_val, y_val))
    score = model.evaluate(x_val, y_val, verbose=0)
    # print('Val accuracy:', score[1])
    exit()
