'''
Counting the number of Investigator dialogues and using a hard threshold to classify AD.
The thresholds are looped over 1 to 9 to find best one.
'''

import glob
import os
import numpy as np
np.random.seed(0)

import time
import re
import math

import audio_length

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

from sklearn.model_selection import KFold
from sklearn import preprocessing

import dataset_features
import dataset_utils

def prepare_data():
  dataset_dir = '../ADReSS-IS2020-data/train'

  cc_transcription_files = sorted(glob.glob(os.path.join(dataset_dir, 'transcription/cc/*.cha')))
  cc_audio_files = sorted(glob.glob(os.path.join(dataset_dir, 'Full_wave_enhanced_audio/cc/*.wav')))

  all_counts_cc = []
  for t_f, a_f in zip(cc_transcription_files, cc_audio_files):
    pause_features = dataset_features.get_pause_features(t_f, a_f)
    all_counts_cc.append(pause_features)


  cd_transcription_files = sorted(glob.glob(os.path.join(dataset_dir, 'transcription/cd/*.cha')))
  cd_audio_files = sorted(glob.glob(os.path.join(dataset_dir, 'Full_wave_enhanced_audio/cd/*.wav')))

  all_counts_cd = [] 
  for t_f, a_f in zip(cd_transcription_files, cd_audio_files):
    pause_features = dataset_features.get_pause_features(t_f, a_f)
    all_counts_cd.append(pause_features)

  X = np.concatenate((all_counts_cc, all_counts_cd), axis=0).astype(np.float32)

  ### Regression y values
  y_reg_cc = dataset_utils.get_regression_values(os.path.join(dataset_dir, 'cc_meta_data.txt'))
  y_reg_cd = dataset_utils.get_regression_values(os.path.join(dataset_dir, 'cd_meta_data.txt'))

  y_reg = np.concatenate((y_reg_cc, y_reg_cd), axis=0).astype(np.float32)
  #######################

  ### Regression X values
  X_reg = np.copy(X)
  #######################

  ### Classification y values
  y_cc = np.zeros((len(all_counts_cc), 2))
  y_cc[:,0] = 1

  y_cd = np.zeros((len(all_counts_cd), 2))
  y_cd[:,1] = 1

  y = np.concatenate((y_cc, y_cd), axis=0).astype(np.float32)
  filenames = np.concatenate((cc_transcription_files, cd_transcription_files), axis=0)
  #######################

  p = np.random.permutation(len(X))
  X, X_reg = X[p], X_reg[p]
  y, y_reg = y[p], y_reg[p]
  filenames = filenames[p]

  return X, y, X_reg, y_reg, filenames


def create_model():
  # model = tf.keras.Sequential()
  # model.add(layers.Input(shape=(10,)))
  # model.add(layers.Dense(16, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
  # model.add(layers.Dense(32, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
  # model.add(layers.Dense(16, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
  # model.add(layers.Dropout(0.2))
  # model.add(layers.Dense(2, activation='softmax'))
  model = tf.keras.Sequential()
  model.add(layers.Input(shape=(11,)))
  # model.add(layers.Dropout(0.2))
  model.add(layers.BatchNormalization())
  model.add(layers.Dense(16, activation='relu'))
  # model.add(layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
  # model.add(layers.Dense(64, activation='relu'))
  # model.add(layers.Dense(128, activation='relu'))
  # model.add(layers.Dense(256, activation='sigmoid'))
  # model.add(layers.Dense(128, activation='relu'))
  # model.add(layers.Dense(64, activation='relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(32, activation='relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(16, activation='relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(2, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.01), activity_regularizer=tf.keras.regularizers.l1(0.01)))
  return model

def regression_baseline():

	_, _, _, y_reg = prepare_data()
	values = []
	for i in range(50):
		value = math.sqrt(np.mean(list(map(lambda x: (x-i)**2, list(y_reg)))))
		values.append(value)
	print(values)
	print()
	print(np.argmin(values), np.min(values)) # 23 	7.18279838173066


def regression(models):
	'''
	models is a list of loaded models
	'''

	_, _, X_reg, y_reg, _ = prepare_data()
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
		model_reg.add(layers.Dense(16, activation='relu'))
		model_reg.add(layers.Dense(16, activation='relu'))
		# model_reg.add(layers.BatchNormalization())
		# model_reg.add(layers.Dropout(0.5))
		model_reg.add(layers.Dense(1, activation='relu'))

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

	print()
	print('Train Scores ', train_scores)
	print('Train mean', np.mean(train_scores))
	print('Train std', np.std(train_scores))
	
	print()
	print('Val accuracies ', val_scores)
	print('Val mean', np.mean(val_scores))
	print('Val std', np.std(val_scores))


def training():

  n_split = 5
  epochs = 600
  batch_size = 8

  val_accuracies = []
  train_accuracies = []

  X, y, _, _, filenames = prepare_data()
  fold = 0
  models = []

  for train_index, val_index in KFold(n_split).split(X):

    x_train, x_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    filenames_train, filenames_val = filenames[train_index], filenames[val_index]


    model = create_model()

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
    val_pred = model.predict(x_val)

    for i in range(len(x_val)):
        print(filenames_val[i], np.argmax(val_pred[i])==np.argmax(y_val[i]), val_pred[i])
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
    # exit()





  return models

models = training()

models = [tf.keras.models.load_model('best_model_{}.h5'.format(fold)) for fold in range(5)]
print(models)
regression(models)

# regression_baseline()




# thresholds = []

# for threshold in thresholds:
#     cc_pred = 0
#     cd_pred = 0
#     for cc, cd in zip(all_inv_counts_cc, all_inv_counts_cd):
#         if cc > threshold:
#             cc_pred+=1
#         if cd > threshold:
#             cd_pred+=1
#     print('Threshold of {} Investigator dialogues'.format(threshold))
#     print('Diagnosed {} healthy people'.format(cc_pred))

#     print('Diagnosed {} AD people'.format(cd_pred))


#     precision = cd_pred / (cc_pred + cd_pred)
#     recall = cd_pred / ((54-cd_pred) + cd_pred)
#     accuracy = (54-cc_pred+cd_pred)/108
#     f1_score = (2*precision*recall)/(precision+recall)
#     print('Accuracy {:.3f} '.format(accuracy))
#     print('F1 score {:.3f}'.format(f1_score))
#     print('Precision {:.3f}'.format(precision))
#     print('Recall {:.3f}'.format(recall))

#     print('----'*50)
