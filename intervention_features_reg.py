'''
Counting the number of Investigator dialogues and using a hard threshold to classify AD.
The thresholds are looped over 1 to 9 to find best one.
'''
import glob
import os
import numpy as np
import time
import re
import math

import audio_length

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

from sklearn.model_selection import KFold
from sklearn import preprocessing


from sklearn.preprocessing import OneHotEncoder
import time

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

def prepare_data():
    print('------- CC ------')
    dataset_dir = '../ADReSS-IS2020-data/train/transcription/cc/'
    files = sorted(glob.glob(os.path.join(dataset_dir, '*.cha')))
    all_inv_counts_cc = []
    all_speakers_cc = []
    for filename in files:
        inv_count = 0
        with open(filename, 'r') as f:
            content = f.read().split('\n')
            speaker_cc = []
            uh_count = 0

            for c in content:
                if 'INV' in c:
                    speaker_cc.append('INV')
                if 'PAR' in c:
                    speaker_cc.append('PAR')
                    uh_count+=c.count('uh')
            all_speakers_cc.append(speaker_cc)
            PAR_first_index = speaker_cc.index('PAR')
            PAR_last_index = len(speaker_cc) - speaker_cc[::-1].index('PAR') - 1
            speaker_cc = speaker_cc[PAR_first_index:PAR_last_index]
            inv_count = speaker_cc.count('INV')
        all_inv_counts_cc.append(inv_count)
        # print('{} has {} INVs'.format(filename.split('/')[-1], inv_count))

    print('------- CD ------')

    dataset_dir = '../ADReSS-IS2020-data/train/transcription/cd/'
    files = sorted(glob.glob(os.path.join(dataset_dir, '*.cha')))
    all_inv_counts_cd = []
    all_speakers_cd = []
    for filename in files:
        inv_count = 0
        with open(filename, 'r') as f:
            content = f.read().split('\n')
            speaker_cd = []
            uh_count = 0

            for c in content:
                if 'INV' in c:
                    speaker_cd.append('INV')
                if 'PAR' in c:
                    speaker_cd.append('PAR')
                    uh_count+=c.count('uh')
            all_speakers_cd.append(speaker_cd)
            PAR_first_index = speaker_cd.index('PAR')
            PAR_last_index = len(speaker_cd) - speaker_cd[::-1].index('PAR') - 1
            speaker_cd = speaker_cd[PAR_first_index:PAR_last_index]
            inv_count = speaker_cd.count('INV')
        all_inv_counts_cd.append(inv_count)


    speaker_dict = {
        'INV': [0 ,0 , 1],
        'PAR': [0, 1, 0],
        'padding': [1, 0, 0]
    }

    all_speakers_cc_binary = list(map(lambda y: list(map(lambda x: speaker_dict[x], y)), all_speakers_cc))
    all_speakers_cd_binary = list(map(lambda y: list(map(lambda x: speaker_dict[x], y)), all_speakers_cd))

    longest_speakers_cc = max(all_speakers_cc_binary, key=lambda x: len(x))
    longest_speakers_cd = max(all_speakers_cd_binary, key=lambda x: len(x))

    longest_speaker_length = max(len(longest_speakers_cc), len(longest_speakers_cd))
    longest_speaker_length = 40

    def padder(x, length=51):
        if len(x) > length:
            return x[:length]
        else:
            pad_length = length - len(x)
            return x+[speaker_dict['padding']]*pad_length

    all_speakers_cc_binary = list(map(lambda x: padder(x, length=longest_speaker_length), all_speakers_cc_binary))
    all_speakers_cd_binary = list(map(lambda x: padder(x, length=longest_speaker_length), all_speakers_cd_binary))

    y_cc = np.zeros((len(all_speakers_cc_binary), 2))
    y_cc[:,0] = 1

    y_cd = np.zeros((len(all_speakers_cd_binary), 2))
    y_cd[:,1] = 1

    ### Classification values
    X_class = np.concatenate((all_speakers_cc_binary, all_speakers_cd_binary), axis=0).astype(np.float32)
    y_class = np.concatenate((y_cc, y_cd), axis=0).astype(np.float32)


    ### Regression values
    X_reg = np.copy(X_class)

    y_reg_cc = np.zeros((len(all_speakers_cc_binary), ))
    file = open('../ADReSS-IS2020-data/train/cc_meta_data.txt', 'r+')
    lines = file.readlines()[1:]
    for idx, line in enumerate(lines):
        token = line.split('; ')[-1].strip('\n')
        if token!='NA':		y_reg_cc[idx] = int(token)
        else:		y_reg_cc[idx] = 30

    y_reg_cd = np.zeros((len(all_speakers_cd_binary), ))
    file = open('../ADReSS-IS2020-data/train/cd_meta_data.txt', 'r+')
    lines = file.readlines()[1:]
    for idx, line in enumerate(lines):
        token = line.split('; ')[-1].strip('\n')
        y_reg_cd[idx] = int(token)

    y_reg = np.concatenate((y_reg_cc, y_reg_cd), axis=0).astype(np.float32)

    ### Premutation of data
    np.random.seed(0)
    p = np.random.permutation(len(X_class))

    X_class, X_reg = X_class[p], X_reg[p]
    y_class, y_reg = y_class[p], y_reg[p]

    return X_class, y_class, X_reg, y_reg

def create_model(longest_speaker_length, num_classes=2):
    model = tf.keras.Sequential()
    # model.add(layers.Input((longest_speaker_length)))
    # model.add(layers.TimeDistributed(layers.Flatten()))
    # model.add(layers.LSTM(128))
    model.add(layers.LSTM(16, input_shape=(longest_speaker_length, 3)))
    # model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

# print(create_model(longest_speaker_length).summary())

def regression(models):
	'''
	models is a list of loaded models
	'''

	_, _, X_reg, y_reg = prepare_data()
	fold = 0
	n_split = 5
	epochs = 1500
	batch_size = 8

	train_scores, val_scores = [], []
	all_train_predictions, all_val_predictions = [], []
	all_train_true, all_val_true = [], []

	for train_index, val_index in KFold(n_split).split(X_reg):

		# def root_mean_squared_error(y_true, y_pred):
		# 	return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

		x_train, x_val = X_reg[train_index], X_reg[val_index]
		y_train, y_val = y_reg[train_index], y_reg[val_index]

		model = models[fold]
		model.pop()
		for layer in model.layers:
			layer.trainable = False

		model_reg = tf.keras.Sequential()
		model_reg.add(model)
		model_reg.add(layers.Dense(8, activation='relu'))
		model_reg.add(layers.BatchNormalization())
		model_reg.add(layers.Dropout(0.5))
		model_reg.add(layers.Dense(1, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), activity_regularizer=tf.keras.regularizers.l1(0.01)))

		# print(model_reg.summary())
		model_reg.compile(loss=tf.keras.losses.mean_squared_error,
			optimizer=tf.keras.optimizers.Adam(lr=0.001))

		checkpointer = tf.keras.callbacks.ModelCheckpoint(
						'best_model_reg_{}.h5'.format(fold), monitor='val_loss', verbose=0, save_best_only=True,
						save_weights_only=False, mode='auto', save_freq='epoch')

		model_reg.fit(x_train, y_train,
							batch_size=batch_size,
							epochs=epochs,
							verbose=1,
							callbacks=[checkpointer],
							validation_data=(x_val, y_val))

		model_reg = tf.keras.models.load_model('best_model_reg_{}.h5'.format(fold))
		# models.append(model)
		train_score = math.sqrt(model_reg.evaluate(x_train, y_train, verbose=0))
		train_scores.append(train_score)
		val_score = math.sqrt(model_reg.evaluate(x_val, y_val, verbose=0))
		val_scores.append(val_score)
		train_predictions = model_reg.predict(x_train)
		all_train_predictions.append(train_predictions)
		val_predictions = model_reg.predict(x_val)
		all_val_predictions.append(val_predictions)
		all_train_true.append(y_train)
		all_val_true.append(y_val)
		fold+=1

	print('Train Scores ', train_scores)
	print('Train mean', np.mean(train_scores))
	print('Train std', np.std(train_scores))
	print()

	print('Val accuracies ', val_scores)
	print('Val mean', np.mean(val_scores))
	print('Val std', np.std(val_scores))
	print()

	print('################### TRAIN VALUES ###################')
	for f in range(n_split):
		print()
		print('################### FOLD {} ###################'.format(f))
		print('True Values \t Predicted Values')
		for i in range(all_train_true[f].shape[0]):
			print(all_train_true[f][i], '\t\t', all_train_predictions[f][i,0])

	print()
	print('################### VAL VALUES ###################')
	for f in range(n_split):
		print()
		print('################### FOLD {} ###################'.format(f))
		print('True Values \t Predicted Values')
		for i in range(all_val_true[f].shape[0]):
			print(all_val_true[f][i], '\t\t', all_val_predictions[f][i,0])


def training():
    n_split = 5
    epochs = 600
    batch_size = 8

    val_accuracies = []
    train_accuracies = []

    X, y, _, _ = prepare_data()
    longest_speaker_length = len(X[0])

    fold = 0
    models = []

    for train_index, val_index in KFold(n_split).split(X):

        x_train, x_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model = create_model(longest_speaker_length)

        # timeString = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        # log_name = "{}".format(timeString)
        # tensorboard = TensorBoard(log_dir="logs/{}".format(log_name), histogram_freq=1, write_graph=True, write_images=False)

        model.compile(loss=tf.keras.losses.categorical_crossentropy,
                                    optimizer=tf.keras.optimizers.Adam(lr=0.001),
                                    metrics=['categorical_accuracy'])

        checkpointer = tf.keras.callbacks.ModelCheckpoint(
                        'best_model_{}.h5'.format(fold), monitor='val_loss', verbose=0, save_best_only=True,
                        save_weights_only=False, mode='auto', save_freq='epoch')

        model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            callbacks=[checkpointer],
                            validation_data=(x_val, y_val))

        model = tf.keras.models.load_model('best_model_{}.h5'.format(fold))
        models.append(model)
        train_score = model.evaluate(x_train, y_train, verbose=0)

        train_accuracies.append(train_score[1])
        score = model.evaluate(x_val, y_val, verbose=0)
        print('Val accuracy:', score[1])
        val_accuracies.append(score[1])
        fold+=1

    print('Train accuracies ', train_accuracies)
    print('Train mean', np.mean(train_accuracies))
    print('Train std', np.std(train_accuracies))

    print('Val accuracies ', val_accuracies)
    print('Val mean', np.mean(val_accuracies))
    print('Val std', np.std(val_accuracies))

    return models

# training()

models = [tf.keras.models.load_model('best_model_{}.h5'.format(fold)) for fold in range(5)]
print(models)
regression(models)