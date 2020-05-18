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

from kerastuner.tuners import RandomSearch

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="5"

import dataset
def train_n_folds(model_type, x, y, n_folds=5, model_dir='default'):

	train_scores = []
	val_scores = []
	fold = 0

	for train_index, val_index in KFold(n_folds).split(x):
		fold+=1
		if fold != 4:
			continue

		x_train, x_val = x[train_index], x[val_index]
		y_train, y_val = y[train_index], y[val_index]
		# filenames_train, filenames_val = filenames[train_index], filenames[val_index]

		results = train_a_fold(model_type, x_train, y_train, x_val, y_val, fold, model_dir)
		train_score, val_score = results

		train_scores.append(train_score)
		val_scores.append(val_score)
		# break
	
	return train_scores, val_scores

def train_a_fold(model_type, x_train, y_train, x_val, y_val, fold, model_dir):

	print('Training fold {} of {}'.format(fold, model_type))

	# if model_type == 'spectogram':
	# 	spectogram_size = (480, 640, 3)
	# 	model = create_spectogram_model(spectogram_size)
	# 	epochs = 30
	# 	batch_size = 8
	# 	epsilon = 0.1
	if model_type == 'spectogram':
		raise Exception('Regression has not been done with spectograms')

	# model = tf.keras.models.load_model(os.path.join(model_dir, model_type, 'fold_{}.h5'.format(fold)))
	# model.pop()
	# for layer in model.layers:
	# 	layer.trainable = False

	if model_type == 'pause':
		n_features = 11
		model_fn = create_pause_model
		# model_fn = lambda x: create_pause_model(x, model=model)

		epochs = 5000
		batch_size = 8

	elif model_type == 'intervention':
		longest_speaker_length = 32
		model_fn = create_intervention_model
		# model_fn = lambda x: create_pause_model(x, model=model)
		epochs = 2000
		batch_size = 8

	elif model_type == 'compare':
		features_size = 21
		# model = create_compare_model(features_size)
		model_fn = lambda x: create_pause_model(x, model=model)

		epochs = 15000
		batch_size = 8

		sc = StandardScaler()
		sc.fit(x_train)

		x_train = sc.transform(x_train)
		x_val = sc.transform(x_val)

		pca = PCA(n_components=features_size)
		pca.fit(x_train)

		x_train = pca.transform(x_train)
		x_val = pca.transform(x_val)

	# checkpointer = tf.keras.callbacks.ModelCheckpoint(
	# 		os.path.join(model_dir, model_type, 'fold_{}.h5'.format(fold)), monitor='val_loss', verbose=0, save_best_only=True,
	# 		save_weights_only=False, mode='auto', save_freq='epoch')

	# model.compile(loss=tf.keras.losses.categorical_crossentropy,
	# 			  optimizer=tf.keras.optimizers.Adam(lr=0.001, epsilon=epsilon),
	# 			  metrics=['categorical_accuracy'])

	tuner = RandomSearch(
		model_fn,
		objective='val_loss',
		max_trials=40,
		executions_per_trial=2,
		directory='tuner/'+model_type,
		project_name=model_type)
	tuner.search(x_train, y_train,
			  batch_size=batch_size,
			  epochs=epochs,
			  verbose=1,
			  # callbacks=[checkpointer],
			  validation_data=(x_val, y_val)
		)
	# model.fit(x_train, y_train,
	# 		  batch_size=batch_size,
	# 		  epochs=epochs,
	# 		  verbose=1,
	# 		  callbacks=[checkpointer],
	# 		  validation_data=(x_val, y_val))

	# model = tf.keras.models.load_model(os.path.join(model_dir, model_type, 'fold_{}.h5'.format(fold)))
	model_reg = tuner.get_best_models(num_models=1)[0]
	print(tuner.results_summary())
	# print(model.summary())

	train_score = math.sqrt(model_reg.evaluate(x_train, y_train, verbose=0))
	# train_accuracy = train_score[1]
	# train_loss = train_score[0]

	val_score = math.sqrt(model_reg.evaluate(x_val, y_val, verbose=0))
	print('Fold Val accuracy:', val_score)
	# val_accuracy = val_score[1]
	# val_loss = val_score[0]

	return train_score, val_score

def create_intervention_model(hp):
	model_reg = tf.keras.Sequential()
	model_reg.add(layers.Input((32,3)))
	model_reg.add(layers.LSTM(hp.Int('units_LSTM',
								min_value=4,
								max_value=16,
								step=4)))
	model_reg.add(layers.BatchNormalization())

	for i in range(hp.Int('num_layers', 1, 4)):
		model_reg.add(layers.Dropout(hp.Float('rate_{}'.format(i),
								min_value=0.0,
								max_value=0.5,
								step=0.1)))
		model_reg.add(layers.Dense(hp.Int('units_{}'.format(i),
								min_value=8,
								max_value=24,
								step=8), activation='relu'))

	model_reg.add(layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(hp.Choice('l2', values=[1e-1, 1e-2, 1e-3])), activity_regularizer=tf.keras.regularizers.l1(hp.Choice('l1', values=[1e-1, 1e-2, 1e-3]))))
	model_reg.add(layers.ReLU(max_value=30))
	model_reg.compile(
		optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate',
					  values=[1e-2, 1e-3, 1e-4])),
		loss=tf.keras.losses.mean_squared_error)
	return model_reg

def create_spectogram_model(spectogram_size):
	model2_input = layers.Input(shape=spectogram_size,  name='spectrogram_input')
	model2_BN = layers.BatchNormalization()(model2_input)
	
	model2_hidden1 = layers.Conv2D(16, kernel_size=(3, 3), strides=(1, 1),
						 activation='relu')(model2_BN)
	# model2_hidden2 = layers.Conv2D(16, kernel_size=(3, 3), strides=(2, 2),
	# 					 activation='relu')(model2_hidden1)
	model2_BN1 = layers.BatchNormalization()(model2_hidden1)
	model2_hidden2 = layers.MaxPool2D()(model2_BN1)
	
	model2_hidden3 = layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
						 activation='relu')(model2_hidden2)
	# model2_hidden4 = layers.Conv2D(32, kernel_size=(3, 3), strides=(2, 2),
	# 					 activation='relu')(model2_hidden3)
	model2_BN2 = layers.BatchNormalization()(model2_hidden3)
	model2_hidden4 = layers.MaxPool2D()(model2_BN2)

	model2_hidden5 = layers.Conv2D(64, kernel_size=(5, 5), strides=(1, 1),
						 activation='relu')(model2_hidden4)
	# model2_hidden6 = layers.Conv2D(64, kernel_size=(3, 3), strides=(2, 2),
	# 					 activation='relu')(model2_hidden5)
	model2_BN3 = layers.BatchNormalization()(model2_hidden5)
	model2_hidden6 = layers.MaxPool2D()(model2_BN3)

	model2_hidden7 = layers.Conv2D(128, kernel_size=(5, 5), strides=(1, 1),
						 activation='relu')(model2_hidden6)
	# model2_hidden8 = layers.Conv2D(128, kernel_size=(3, 3), strides=(2, 2),
	# 					 activation='relu')(model2_hidden7)
	model2_BN4 = layers.BatchNormalization()(model2_hidden7)
	model2_hidden8 = layers.MaxPool2D()(model2_BN4)

	model2_hidden9 = layers.Flatten()(model2_hidden8)
	# model2_hidden10 = layers.Dropout(0.2)(model2_hidden9)
	model2_hidden10 = layers.BatchNormalization()(model2_hidden9)
	model2_hidden11 = layers.Dense(128, activation='relu')(model2_hidden10)
	model2_output = layers.Dropout(0.2)(model2_hidden11)
	model2_output = layers.Dense(2, activation='softmax')(model2_output)

	model = Model(model2_input, model2_output)

	return model

def create_pause_model(hp):
	model_reg = tf.keras.Sequential()
	model_reg.add(layers.Input(shape=(11,)))
	model_reg.add(layers.BatchNormalization())
	for i in range(hp.Int('num_layers', 1, 4)):
		model_reg.add(layers.Dense(hp.Int('units_{}'.format(i),
								min_value=8,
								max_value=32,
								step=8), activation='relu'))
		model_reg.add(layers.BatchNormalization())
		model_reg.add(layers.Dropout(hp.Float('rate_{}'.format(i),
									min_value=0.0,
									max_value=0.5,
									step=0.1)))
	for i in range(hp.Int('num_layers_post', 1, 3)):
		model_reg.add(layers.Dense(hp.Int('units_post_{}'.format(i),
								min_value=8,
								max_value=32,
								step=8), activation='relu'))
	model_reg.add(layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(hp.Choice('l2', values=[1e-1, 1e-2, 1e-3])), activity_regularizer=tf.keras.regularizers.l1(hp.Choice('l1', values=[1e-1, 1e-2, 1e-3]))))
	model_reg.add(layers.ReLU(max_value=30))
	model_reg.compile(
		optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate',
					  values=[1e-2, 1e-3, 1e-4])),
		loss=tf.keras.losses.mean_squared_error)
	return model_reg

def create_compare_model(hp, model):

	model_reg = tf.keras.Sequential()
	model_reg.add(model)
	# model.add(layers.BatchNormalization())
	for i in range(hp.Int('num_layers', 1, 4)):
		model_reg.add(layers.Dense(hp.Int('units_{}'.format(i),
								min_value=8,
								max_value=32,
								step=8), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(hp.Choice('l2-pre', values=[1e-1, 1e-2, 1e-3])), activity_regularizer=tf.keras.regularizers.l1(hp.Choice('l1-pre', values=[1e-1, 1e-2, 1e-3]))))
		model_reg.add(layers.Dropout(hp.Float('rate_{}'.format(i),
									min_value=0.0,
									max_value=0.5,
									step=0.1)))
	model_reg.add(layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(hp.Choice('l2', values=[1e-1, 1e-2, 1e-3])), activity_regularizer=tf.keras.regularizers.l1(hp.Choice('l1', values=[1e-1, 1e-2, 1e-3]))))
	model_reg.add(layers.ReLU(max_value=30))

	model_reg.compile(
		optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate',
					  values=[1e-2, 1e-3, 1e-4])),
		loss=tf.keras.losses.mean_squared_error)
	return model_reg

data = dataset.prepare_data('../ADReSS-IS2020-data/train')
X_intervention, X_pause, X_spec, X_compare, X_reg_intervention, X_reg_pause, X_reg_compare = data[0:7]
y, y_reg, filenames_intervention, filenames_pause, filenames_spec, filenames_compare = data[7:]

feature_types = {
	'intervention': X_reg_intervention,
	'pause': X_reg_pause,
	'compare': X_reg_compare
}

results = {}
# model_types = ['compare']
model_types = ['pause'] # from scratch ran tmux 5
model_types = ['intervention'] # from scratch ran tmux 4
# model_types = ['intervention'] # from results/bagging_val_loss fold 4 ran in tmux 6
# model_types = ['pause'] # from results/bagging_val_loss fold 4 ran in tmux 5
# model_types = ['compare'] # from results/bagging_tuned_val_loss fold 1 ran in tmux 5

print('keras tuner using model type ', model_types[0], 'scratch')

## Train Intervention, models saved in `model_dir/intervention_{fold}.h5`
for m in model_types:
	m_results = train_n_folds(m, feature_types[m], y_reg, 5, 'tuner')
print(m_results)
