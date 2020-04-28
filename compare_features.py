
import glob
import os
import math
import numpy as np
import csv
np.random.seed(0)

import tensorflow as tf

from tensorflow.keras import layers
from sklearn.model_selection import KFold
from tensorflow.keras.models import Model
import time

import dataset_features, dataset_utils

def prepare_data():
    dataset_dir = '../ADReSS-IS2020-data/train/'

    cc_files = sorted(glob.glob(os.path.join(dataset_dir, 'compare_features/compare_cc/*.csv')))
    all_speakers_cc = []
    for filename in cc_files:
        all_speakers_cc.append(dataset_features.get_compare_features(filename))

    cd_files = sorted(glob.glob(os.path.join(dataset_dir, 'compare_features/compare_cd/*.csv')))
    all_speakers_cd = []
    for filename in cd_files:
        all_speakers_cd.append(dataset_features.get_compare_features(filename))

    ### Classification X and y values
    y_cc = np.zeros((len(all_speakers_cc), 2))
    y_cc[:,0] = 1

    y_cd = np.zeros((len(all_speakers_cd), 2))
    y_cd[:,1] = 1

    ### Regression y values
    y_reg_cc = dataset_utils.get_regression_values(os.path.join(dataset_dir, 'cc_meta_data.txt'))
    y_reg_cd = dataset_utils.get_regression_values(os.path.join(dataset_dir, 'cd_meta_data.txt'))


    ### X and y
    X = np.concatenate((all_speakers_cc, all_speakers_cd), axis=0).astype(np.float32)
    y = np.concatenate((y_cc, y_cd), axis=0).astype(np.float32)

    X_reg = np.copy(X)
    y_reg = np.concatenate((y_reg_cc, y_reg_cd), axis=0).astype(np.float32)

    filenames = np.concatenate((cc_files, cd_files), axis=0)

    p = np.random.permutation(len(X))
    X, X_reg = X[p], X_reg[p]
    y, y_reg = y[p], y_reg[p]
    filenames = filenames[p]

    return X, y, X_reg, y_reg, filenames

def create_model():
    INP = layers.Input(shape=(6373,))
    BN1 = layers.BatchNormalization()(INP)

    D1 = layers.Dense(16, activation='relu')(BN1)
    BN2 = layers.BatchNormalization()(D1)
    DP1 = layers.Dropout(0.5)(BN2)

    D2 = layers.Dense(32, activation='relu')(DP1)
    BN3 = layers.BatchNormalization()(D2)
    DP2 = layers.Dropout(0.5)(BN3)

    D3 = layers.Dense(32, activation='relu')(DP2)
    BN4 = layers.BatchNormalization()(D3)
    DP3 = layers.Dropout(0.5)(BN4)

    D4 = layers.Dense(2, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.01), activity_regularizer=tf.keras.regularizers.l1(0.01))(DP3)

    model = Model(INP, D4)
    return model


def training(loocv=False):

	epochs = 600
	batch_size = 8

	val_accuracies = []
	train_accuracies = []

	X, y, _, _, filenames = prepare_data()

	if loocv==True:
		n_split = X.shape[0]
		model_dir = 'loocv-models-compare'
	else:
		n_split = 5
		model_dir = '5-fold-models-compare'

	fold = 0
	models = []

	for train_index, val_index in KFold(n_split).split(X):
		fold+=1

		x_train, x_val = X[train_index], X[val_index]
		y_train, y_val = y[train_index], y[val_index]
		filenames_train, filenames_val = filenames[train_index], filenames[val_index]


		model = create_model()

		model.compile(loss=tf.keras.losses.categorical_crossentropy,
		              optimizer=tf.keras.optimizers.Adam(lr=0.001),
		              metrics=['categorical_accuracy'])

		checkpointer = tf.keras.callbacks.ModelCheckpoint(
		        os.path.join(model_dir, 'compare_{}.h5'.format(fold)), monitor='val_loss', verbose=0, save_best_only=True,
		        save_weights_only=False, mode='auto', save_freq='epoch')

		model.fit(x_train, y_train,
		          batch_size=batch_size,
		          epochs=epochs,
		          verbose=1,
		          callbacks=[checkpointer],
		          validation_data=(x_val, y_val))

		model = tf.keras.models.load_model(os.path.join(model_dir, 'compare_{}.h5'.format(fold)))
		val_pred = model.predict(x_val)

		for i in range(len(x_val)):
		    print(filenames_val[i], np.argmax(val_pred[i])==np.argmax(y_val[i]), val_pred[i])
		models.append(model)
		train_score = model.evaluate(x_train, y_train, verbose=0)

		train_accuracies.append(train_score[1])
		score = model.evaluate(x_val, y_val, verbose=0)
		print('Val accuracy:', score[1])
		val_accuracies.append(score[1])
		print('Val mean till fold {} is {}'.format(fold, np.mean(val_accuracies)))

	print('Train accuracies ', train_accuracies)
	print('Train mean', np.mean(train_accuracies))
	print('Train std', np.std(train_accuracies))

	print('Val accuracies ', val_accuracies)
	print('Val mean', np.mean(val_accuracies))
	print('Val std', np.std(val_accuracies))
    # exit()
	return models

def training_on_entire_dataset():

	epochs = 600
	batch_size = 8
	X, y, _, _, filenames = prepare_data()

	model = create_model()
	model.compile(loss=tf.keras.losses.categorical_crossentropy,
		              optimizer=tf.keras.optimizers.Adam(lr=0.001),
		              metrics=['categorical_accuracy'])

	checkpointer = tf.keras.callbacks.ModelCheckpoint(
	        'best_model_compare.h5', monitor='loss', verbose=0, save_best_only=True,
	        save_weights_only=False, mode='auto', save_freq='epoch')

	model.fit(X, y,
	          batch_size=batch_size,
	          epochs=epochs,
	          verbose=1,
	          callbacks=[checkpointer])

	model = tf.keras.models.load_model('best_model_compare.h5')
	train_loss, train_acc = model.evaluate(X, y, verbose=0)
	print('Train Loss: {}\t Train Accuracy: {}'.format(train_loss, train_acc))




models = training()
