import glob, os, math, time
import numpy as np
np.random.seed(0)

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import models

def train_n_folds(model_type, data, config):

	train_accuracies = []
	val_accuracies = []
	fold = 0

	x = data[model_type]

	if config.task == 'classification':
		y = data['y_clf']
	elif config.task == 'regression':
		y = data['y_reg']

	subjects = data['subjects']

	if config.split_reference == 'samples':
		splitter = x
	elif config.split_reference == 'subjects':
		splitter = subjects

	for train_index, val_index in KFold(config.n_folds).split(splitter):
		fold+=1

		if config.split_reference == 'samples':
			x_train, x_val = x[train_index], x[val_index]
			y_train, y_val = y[train_index], y[val_index]

		results = train_a_fold(model_type, x_train, y_train, x_val, y_val, fold, config)
		train_accuracy, val_accuracy= results

		train_accuracies.append(train_accuracy)
		val_accuracies.append(val_accuracy)
	
	return train_accuracies, val_accuracies

def train_a_fold(model_type, x_train, y_train, x_val, y_val, fold, config):

	print('Training fold {} of {}'.format(fold, model_type))

	if model_type == 'pause':
		model = models.create_pause_model(config.task, config.n_pause_features)
		epsilon = 1e-07

	elif model_type == 'intervention':
		model = models.create_intervention_model(config.task, config.longest_speaker_length)
		epsilon = 1e-07

	elif model_type == 'compare':
		model = models.create_compare_model(config.task, config.compare_features_size)
		epsilon = 1e-07

		sc = StandardScaler()
		sc.fit(x_train)

		x_train = sc.transform(x_train)
		x_val = sc.transform(x_val)

		pca = PCA(n_components=config.compare_features_size)
		pca.fit(x_train)

		x_train = pca.transform(x_train)
		x_val = pca.transform(x_val)

	if config.task == 'classification':
		epochs = 10
		batch_size = 8

		model.compile(loss= tf.keras.losses.categorical_crossentropy,
				  optimizer=tf.keras.optimizers.Adam(lr=0.01, epsilon=epsilon),
				  metrics=['categorical_accuracy'])
	elif config.task == 'regression':
		epochs = 20
		batch_size = 8

		model.compile(loss=tf.keras.losses.mean_squared_error, 
			optimizer=tf.keras.optimizers.Adam(lr=0.01, epsilon=epsilon))

	checkpointer = tf.keras.callbacks.ModelCheckpoint(
			os.path.join(config.model_dir, model_type, 'fold_{}.h5'.format(fold)), monitor='val_loss', verbose=0, save_best_only=True,
			save_weights_only=False, mode='auto', save_freq='epoch')

	model.fit(x_train, y_train,
			  batch_size=batch_size,
			  epochs=epochs,
			  verbose=1,
			  callbacks=[checkpointer],
			  validation_data=(x_val, y_val))

	model = tf.keras.models.load_model(os.path.join(config.model_dir, model_type, 'fold_{}.h5'.format(fold)))

	train_score = model.evaluate(x_train, y_train, verbose=0)
	if config.task == 'classification':
		train_score = train_score[1]

	val_score = model.evaluate(x_val, y_val, verbose=0)
	if config.task == 'classification':
		val_score = val_score[1]

	print('Fold Val accuracy:', val_score)

	return train_score, val_score
