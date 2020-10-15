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
from sklearn.metrics import mean_squared_error
from pickle import dump

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

	if config.dataset_split == 'kfold':
		for train_index, val_index in KFold(config.n_folds).split(splitter):
			fold+=1

			if config.split_reference == 'samples':
				x_train, x_val = x[train_index], x[val_index]
				y_train, y_val = y[train_index], y[val_index]

			results = train_a_fold(model_type, x_train, y_train, x_val, y_val, fold, config)
			train_accuracy, val_accuracy= results

			train_accuracies.append(train_accuracy)
			val_accuracies.append(val_accuracy)
	else:
		numpy_seeds = [913293, 653261, 84754, 645, 13451235]

		for i in range(config.n_folds):
			fold+=1
			np.random.seed(numpy_seeds[i])

			p = np.random.permutation(len(x))
			x = x[p]
			y = y[p]

			n_train = int(config.split_ratio * len(x))
			x_train, x_val = x[:n_train], x[n_train:]
			y_train, y_val = y[:n_train], y[n_train:]

			results = train_a_fold(model_type, x_train, y_train, x_val, y_val, fold, config)
			train_accuracy, val_accuracy= results

			train_accuracies.append(train_accuracy)
			val_accuracies.append(val_accuracy)

	return train_accuracies, val_accuracies

def train_a_fold(model_type, x_train, y_train, x_val, y_val, fold, config, sample_weight=None):

	print('Training fold {} of {}'.format(fold, model_type))

	if model_type == 'pause':
		model = models.create_pause_model(config.task, config.n_pause_features, config.uncertainty)
		epsilon = 1e-07
		config.lr = 0.00125
		config.batch_size = 24

	elif model_type == 'intervention':
		model = models.create_intervention_model(config.task, config.longest_speaker_length, config.uncertainty)
		epsilon = 1e-07
		config.lr = 0.00125
		config.batch_size = 24

	elif model_type == 'compare':
		model = models.create_compare_model(config.task, config.compare_features_size, config.uncertainty)
		epsilon = 1e-07
		config.lr = 0.01
		config.batch_size = 16

		sc = StandardScaler()
		sc.fit(x_train)

		x_train = sc.transform(x_train)
		x_val = sc.transform(x_val)

		pca = PCA(n_components=config.compare_features_size)
		pca.fit(x_train)

		x_train = pca.transform(x_train)
		x_val = pca.transform(x_val)

		dump(sc, open(os.path.join(config.model_dir, 'compare/scaler_{}.pkl'.format(fold)), 'wb'))
		dump(pca, open(os.path.join(config.model_dir, 'compare/pca_{}.pkl'.format(fold)), 'wb'))

	elif model_type == 'silences':
		model = models.create_silences_model(config.task, config.uncertainty)
		epsilon = 1e-07

	save_weights_only = False

	if config.task == 'classification':

		best_model = 'val_loss'
		model.compile(loss= tf.keras.losses.categorical_crossentropy,
				  optimizer=tf.keras.optimizers.Adam(lr=config.lr, epsilon=epsilon),
				  metrics=['categorical_accuracy'])

	elif config.task == 'regression':

		best_model = 'val_loss'
		if config.uncertainty:
			def negloglik(y, p_y):
				return -p_y.log_prob(y)
			model.compile(loss=negloglik, 
				optimizer=tf.keras.optimizers.Adam(lr=config.lr, epsilon=epsilon), metrics=['mse'])
			save_weights_only = False

		else:
			model.compile(loss=tf.keras.losses.mean_squared_error, 
				optimizer=tf.keras.optimizers.Adam(lr=config.lr, epsilon=epsilon), metrics=['mse'])

	checkpointer = tf.keras.callbacks.ModelCheckpoint(
			os.path.join(config.model_dir, model_type, 'fold_{}.h5'.format(fold)), monitor=best_model, verbose=0, save_best_only=True,
			save_weights_only=save_weights_only, mode='auto', save_freq='epoch')

	hist = model.fit(x_train, y_train,
			  batch_size=config.batch_size,
			  epochs=config.n_epochs,
			  verbose=config.verbose,
			  callbacks=[checkpointer],
			  validation_data=(x_val, y_val),
			  sample_weight=sample_weight)


	if config.uncertainty:
		model = tf.keras.models.load_model(os.path.join(config.model_dir, model_type, 'fold_{}.h5'.format(fold)),
			custom_objects={'negloglik': negloglik})

		preds = model(x_train)
		mus = preds.mean().numpy()
		train_score = mean_squared_error(y_train,mus, squared=False)

		preds = model(x_val)
		mus = preds.mean().numpy()
		val_score = mean_squared_error(y_val, mus, squared=False)

	else:
		model = tf.keras.models.load_model(os.path.join(config.model_dir, model_type, 'fold_{}.h5'.format(fold)))

		train_score = model.evaluate(x_train, y_train, verbose=0)
		if config.task == 'classification':
			train_score = train_score[1]

		val_score = model.evaluate(x_val, y_val, verbose=0)
		if config.task == 'classification':
			val_score = val_score[1]

	epoch_val_losses = hist.history['val_loss']
	best_epoch_val_loss, best_epoch = np.min(epoch_val_losses), np.argmin(epoch_val_losses)+1
	best_epoch_train_loss = hist.history['loss'][best_epoch-1]

	print('Best Epoch: {:d}'.format(best_epoch))
	print('Best Val loss {:.3f}'.format(best_epoch_val_loss))
	print('Fold Val accuracy:', val_score)

	return train_score, val_score
