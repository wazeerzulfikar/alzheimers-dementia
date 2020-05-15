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

def train_n_folds(model_type, x, y, n_folds=5, model_dir='default'):

	train_scores = []
	val_scores = []
	fold = 0

	for train_index, val_index in KFold(n_folds).split(x):
		fold+=1

		x_train, x_val = x[train_index], x[val_index]
		y_train, y_val = y[train_index], y[val_index]

		results = train_a_fold(model_type, x_train, y_train, x_val, y_val, fold, model_dir)
		train_score, val_score = results

		train_scores.append(train_score)
		val_scores.append(val_score)
	
	return train_scores, val_scores

def train_a_fold(model_type, x_train, y_train, x_val, y_val, fold, model_dir):

	print('Training fold {} of {}'.format(fold, model_type))

	if model_type == 'spectogram':
		raise Exception('Regression has not been done with spectograms')

	model = tf.keras.models.load_model(os.path.join(model_dir, model_type, 'fold_{}.h5'.format(fold)))
	model.pop()
	for layer in model.layers:
		layer.trainable = False

	if model_type == 'pause':
		model_reg = create_pause_model(model)
		epochs = 1500
		batch_size = 8

	elif model_type == 'intervention':
		model_reg = create_intervention_model(model)
		epochs = 2000
		batch_size = 8

	elif model_type == 'compare':
		features_size = 21
		model_reg = create_compare_model(model)	
		epochs = 20000
		batch_size = 8

		sc = StandardScaler()
		sc.fit(x_train)

		x_train = sc.transform(x_train)
		x_val = sc.transform(x_val)

		pca = PCA(n_components=features_size)
		pca.fit(x_train)

		x_train = pca.transform(x_train)
		x_val = pca.transform(x_val)

	checkpointer = tf.keras.callbacks.ModelCheckpoint(
			os.path.join(model_dir, model_type, 'reg', 'fold_{}.h5'.format(fold)), monitor='val_loss', verbose=0, save_best_only=True,
			save_weights_only=False, mode='auto', save_freq='epoch')

	model_reg.compile(loss=tf.keras.losses.mean_squared_error, 
			optimizer=tf.keras.optimizers.Adam(lr=0.001))

	model_reg.fit(x_train, y_train,
						batch_size=batch_size,
						epochs=epochs,
						verbose=1,
						callbacks=[checkpointer],
						validation_data=(x_val, y_val))

	model_reg = tf.keras.model.load_model(os.path.join(model_dir, model_type, 'reg', 'fold_{}.h5'.format(fold))) 

	train_score = math.sqrt(model_reg.evaluate(x_train, y_train, verbose=0))

	val_score = math.sqrt(model_reg.evaluate(x_val, y_val, verbose=0))
	print('Fold Val accuracy:', val_score)

	return train_score, val_score	  


def create_pause_model(model):
	model_reg = tf.keras.Sequential()
	model_reg.add(model)
	model_reg.add(layers.Dense(16, activation='relu'))
	model_reg.add(layers.Dense(16, activation='relu'))
	model_reg.add(layers.Dense(1))
	model_reg.add(layers.ReLU(max_value=30))
	return model_reg

def create_intervention_model(model):
	model_reg = tf.keras.Sequential()
	model_reg.add(model)
	model_reg.add(layers.Dense(16, activation='relu'))
	model_reg.add(layers.Dense(8, activation='relu'))
	model_reg.add(layers.Dense(1))
	model_reg.add(layers.ReLU(max_value=30))
	return model_reg

def create_compare_model(model):
	model_reg = tf.keras.Sequential()
	model_reg.add(model)
	model_reg.add(layers.Dense(8, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), activity_regularizer=tf.keras.regularizers.l1(0.01)))
	model_reg.add(layers.Dense(8, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), activity_regularizer=tf.keras.regularizers.l1(0.01)))
	model_reg.add(layers.Dropout(0.5))
	model_reg.add(layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01), activity_regularizer=tf.keras.regularizers.l1(0.01)))
	model_reg.add(layers.ReLU(max_value=30))
	return model_reg


