import glob, os, math, time
import numpy as np
np.random.seed(0)
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import dataset

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def train_n_folds(model_type, x, y, n_folds=5, model_dir='default'):

	train_scores = []
	val_scores = []
	fold = 0

	for train_index, val_index in KFold(n_folds).split(x):
		fold+=1

		x_train, x_val = x[train_index], x[val_index]
		y_train, y_val = y[train_index], y[val_index]
		# filenames_train, filenames_val = filenames[train_index], filenames[val_index]

		results = train_a_fold(model_type, x_train, y_train, x_val, y_val, fold, model_dir)
		train_score, val_score = results

		train_scores.append(train_score)
		val_scores.append(val_score)
	
	return train_scores, val_scores

def train_a_fold(model_type, x_train, y_train, x_val, y_val, fold, model_dir):

	print('Training fold {} of {}'.format(fold, model_type))

	if model_type == 'spectogram':
		raise Exception('Regression has not been done with spectograms')

	# model = tf.keras.models.load_model(os.path.join(model_dir, model_type, 'fold_{}.h5'.format(fold)))
	# model.pop()
	# for layer in model.layers:
	# 	layer.trainable = False

	if model_type == 'pause':
		model_reg = create_pause_model()
		epochs = 5000 # 3000
		batch_size = 8
		lr = 0.01

	elif model_type == 'intervention':
		model_reg = create_intervention_model()
		epochs = 2000
		batch_size = 8
		lr = 0.001

	elif model_type == 'compare':
		features_size = 21
		model_reg = create_compare_model(model)	
		epochs = 20000
		batch_size = 8
		lr = 0.01

		sc = StandardScaler()
		sc.fit(x_train)

		x_train = sc.transform(x_train)
		x_val = sc.transform(x_val)

		pca = PCA(n_components=features_size)
		pca.fit(x_train)

		x_train = pca.transform(x_train)
		x_val = pca.transform(x_val)

	checkpointer = tf.keras.callbacks.ModelCheckpoint(
			os.path.join(model_dir, model_type, 'scratch', 'fold_{}.h5'.format(fold)), monitor='val_loss', verbose=0, save_best_only=True,
			save_weights_only=False, mode='auto', save_freq='epoch')

	model_reg.compile(loss=tf.keras.losses.mean_squared_error, 
			optimizer=tf.keras.optimizers.Adam(lr=lr))

	model_reg.fit(x_train, y_train,
						batch_size=batch_size,
						epochs=epochs,
						verbose=1,
						callbacks=[checkpointer],
						validation_data=(x_val, y_val))

	model_reg = tf.keras.models.load_model(os.path.join(model_dir, model_type, 'scratch', 'fold_{}.h5'.format(fold))) 

	train_score = math.sqrt(model_reg.evaluate(x_train, y_train, verbose=0))

	val_score = math.sqrt(model_reg.evaluate(x_val, y_val, verbose=0))
	print('Fold Val accuracy:', val_score)

	return train_score, val_score

def create_intervention_model():
	model_reg = tf.keras.Sequential()
	model_reg.add(layers.Input((32,3)))
	model_reg.add(layers.LSTM(12))
	model_reg.add(layers.BatchNormalization())
	model_reg.add(layers.Dropout(0.4))
	model_reg.add(layers.Dense(16, activation='relu'))
	model_reg.add(layers.Dense(16, activation='relu'))
	model_reg.add(layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.1), activity_regularizer=tf.keras.regularizers.l1(0.001)))
	model_reg.add(layers.ReLU(max_value=30))
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

def create_pause_model():
	model_reg = tf.keras.Sequential()
	model_reg.add(layers.Input(shape=(11,)))
	model_reg.add(layers.BatchNormalization())
	model_reg.add(layers.Dense(16, activation='relu'))
	model_reg.add(layers.BatchNormalization())
	model_reg.add(layers.Dropout(0.2))
	model_reg.add(layers.Dense(16, activation='relu'))
	model_reg.add(layers.BatchNormalization())
	model_reg.add(layers.Dropout(0.5))
	model_reg.add(layers.Dense(32, activation='relu'))
	model_reg.add(layers.Dense(16, activation='relu'))
	model_reg.add(layers.Dense(8, activation='relu'))
	model_reg.add(layers.Dropout(0.3))
	model_reg.add(layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01), activity_regularizer=tf.keras.regularizers.l1(0.1)))
	model_reg.add(layers.ReLU(max_value=30))
	return model_reg

def create_compare_model(model):
	model_reg = tf.keras.Sequential()
	model_reg.add(model)
	model_reg.add(layers.Dense(16, activation='relu'))
	model_reg.add(layers.Dropout(0.4))
	model_reg.add(layers.Dense(8, activation='relu'))
	model_reg.add(layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01), activity_regularizer=tf.keras.regularizers.l1(0.01)))
	model_reg.add(layers.ReLU(max_value=30))
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


model_types = ['intervention'] # from scratch running in tmux 2
# model_types = ['pause'] # from scratch running in tmux 0; jumping around loss
# model_types = ['intervention'] # from results/bagging_val_loss based on fold 4 ran in tmux 6
# model_types = ['compare'] # from results/bagging_tuned_val_loss based on fold 1 ran in tmux 5
# model_types = ['pause'] # from results/bagging_val_loss based on fold 4 ran in tmux 5

model_dir = 'tuner'
# model_dir = Path(model_dir)
# model_dir.joinpath(model_types[0]).mkdir(parents=True, exist_ok=True)
# model_dir = str(model_dir)

## Train Intervention, models saved in `model_dir/intervention_{fold}.h5`
for m in model_types:
	t_results, v_results = train_n_folds(m, feature_types[m], y_reg, 108, model_dir)

print('trained ', model_types[0], 'from scratch', '\t')
for i,j in enumerate(t_results):
	print('Fold_{}'.format(i))
	print('Training score ',j)
	print('Val score ',v_results[i])

print('Training score mean', np.mean(t_results))
print('Training score std', np.std(t_results))
print('Val score mean', np.mean(v_results))
print('Val score std', np.std(v_results))


'''
Interventions

 |-Trial ID: ac784870dde38dfaddf0236485cfe3ca
 |-Score: 41.710889543805806
 |-Best step: 0
 > Hyperparameters:
 |-l1: 0.001
 |-l2: 0.01
 |-learning_rate: 0.01
 |-num_layers: 2
 |-rate_0: 0.2
 |-rate_1: 0.4
 |-rate_2: 0.1
 |-rate_3: 0.30000000000000004
 |-units_0: 24
 |-units_1: 24
 |-units_2: 16
 |-units_3: 16
[Trial summary]
 |-Trial ID: a4673a47068e132a6ac439e0ccece048
 |-Score: 41.79220145089286
 |-Best step: 0
 > Hyperparameters:
 |-l1: 0.01
 |-l2: 0.001
 |-learning_rate: 0.01
 |-num_layers: 4
 |-rate_0: 0.30000000000000004
 |-rate_1: 0.5
 |-rate_2: 0.0
 |-rate_3: 0.0
 |-units_0: 24
 |-units_1: 24
 |-units_2: 8
 |-units_3: 8

'''