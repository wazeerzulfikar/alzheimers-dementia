import glob, os, math, time
import numpy as np
np.random.seed(0)

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from kerastuner.tuners import RandomSearch

import dataset

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

def train_n_folds(model_type, x, y_class, y_reg, n_folds=5, model_dir='default', x2 = None):

	train_clf_accuracies_1, train_reg_losses_1 = [], []
	val_clf_accuracies_1, val_reg_losses_1 = [], []
	train_clf_accuracies_2, train_reg_losses_2 = [], []
	val_clf_accuracies_2, val_reg_losses_2 = [], []
	fold = 0

	for train_index, val_index in KFold(n_folds).split(x):
		fold+=1

		x_train, x_val = x[train_index], x[val_index]
		if x2 is not None:
			x2_train, x2_val = x2[train_index], x2[val_index]
		y_train_class, y_val_class = y_class[train_index], y_class[val_index]
		y_train_reg, y_val_reg = y_reg[train_index], y_reg[val_index]
		# filenames_train, filenames_val = filenames[train_index], filenames[val_index]
		if x2 is None:
			results = train_a_fold(model_type, x_train, y_train_class, y_train_reg, x_val, y_val_class, y_val_reg, fold, model_dir)
		else:
			results = train_a_fold(model_type, x_train, y_train_class, y_train_reg, x_val, y_val_class, y_val_reg, fold, model_dir, x2_train=x2_train, x2_val=x2_val)

		# train_reg_loss, train_clf_accuracy, val_reg_loss, val_clf_accuracy = results
		# train_reg_loss_1, train_clf_accuracy_1, val_reg_loss_1, val_clf_accuracy_1, train_reg_loss_2, train_clf_accuracy_2, val_reg_loss_2, val_clf_accuracy_2 = results
		train_reg_loss_1, train_clf_accuracy_1, val_reg_loss_1, val_clf_accuracy_1 = results
		
		train_clf_accuracies_1.append(train_clf_accuracy_1)
		val_clf_accuracies_1.append(val_clf_accuracy_1)
		train_reg_losses_1.append(train_reg_loss_1)
		val_reg_losses_1.append(val_reg_loss_1)
		# train_clf_accuracies_2.append(train_clf_accuracy_2)
		# val_clf_accuracies_2.append(val_clf_accuracy_2)
		# train_reg_losses_2.append(train_reg_loss_2)
		# val_reg_losses_2.append(val_reg_loss_2)

	return train_clf_accuracies_1, train_reg_losses_1, val_clf_accuracies_1, val_reg_losses_1#, train_clf_accuracies_2, train_reg_losses_2, val_clf_accuracies_2, val_reg_losses_2

def train_a_fold(model_type, x_train, y_train_class, y_train_reg, x_val, y_val_class, y_val_reg, fold, model_dir, x2_train=None, x2_val=None):

	print('Training fold {} of {}'.format(fold, model_type))

	if model_type == 'intervention':
		longest_speaker_length = 32
		model = create_intervention_model(longest_speaker_length)
		epochs = 400
		batch_size = 8
		lr = 0.01

	elif model_type == 'pause':
		n_features = 11
		model = create_pause_model(n_features)
		epochs = 600
		batch_size = 8
		lr = 0.01
		epsilon = 1e-07

	elif model_type == 'combined':
		model = create_combined_model(11, 32)
		epochs = 1500
		batch_size = 8
		lr = 0.005
		epsilon = 1e-07
		x_train = (x_train, x2_train)
		x_val = (x_val, x2_val)

	loss_functions = {
		"clf": tf.keras.losses.categorical_crossentropy,
		"reg": tf.keras.losses.mean_squared_error
	}
	loss_weights = {"clf": 1, "reg": 0.01}
	metrics = {
		"clf": ['categorical_accuracy']
		# "regression": [tf.keras.losses.mean_squared_error]
	}
	y_trains = {
		"clf": y_train_class,
		"reg": y_train_reg
	}
	y_vals = {
		"clf": y_val_class,
		"reg": y_val_reg
	}

	checkpointer_1 = tf.keras.callbacks.ModelCheckpoint(
			os.path.join(model_dir, model_type, 'fold_{}_{}.h5'.format(fold, mode_1)), monitor=qty_monitored_1, verbose=0, save_best_only=True,
			save_weights_only=False, mode=mode_1, save_freq='epoch')
	# checkpointer_2 = tf.keras.callbacks.ModelCheckpoint(
	# 		os.path.join(model_dir, model_type, 'fold_{}_{}.h5'.format(fold, mode_2)), monitor=qty_monitored_2, verbose=0, save_best_only=True,
	# 		save_weights_only=False, mode=mode_2, save_freq='epoch')

	model.compile(loss=loss_functions, loss_weights=loss_weights,
				  optimizer=tf.keras.optimizers.Adam(lr=lr),
				  metrics=metrics)

	model.fit(x_train, y_trains,
			  batch_size=batch_size,
			  epochs=epochs,
			  verbose=1,
			  callbacks=[checkpointer_1],
			  # callbacks=[checkpointer_1, checkpointer_2],
			  validation_data=(x_val, y_vals))

	model_1 = tf.keras.models.load_model(os.path.join(model_dir, model_type, 'fold_{}_{}.h5'.format(fold, mode_1)))
	# model_2 = tf.keras.models.load_model(os.path.join(model_dir, model_type, 'fold_{}_{}.h5'.format(fold, mode_2)))

	train_loss_1, train_clf_loss_1, train_reg_loss_1, train_clf_accuracy_1 = model_1.evaluate(x_train, y_trains, verbose=0)
	val_loss_1, val_clf_loss_1, val_reg_loss_1, val_clf_accuracy_1 = model_1.evaluate(x_val, y_vals, verbose=0)
	train_reg_loss_1, val_reg_loss_1 = math.sqrt(train_reg_loss_1), math.sqrt(val_reg_loss_1)

	print()
	print('Fold', fold, 'Classification Val Accuracy: {:.2f} %'.format(100*val_clf_accuracy_1))
	print('Fold', fold, 'Regression Val Score: {:.3f}'.format(val_reg_loss_1))
	print()

	# train_loss_2, train_clf_loss_2, train_reg_loss_2, train_clf_accuracy_2 = model_2.evaluate(x_train, y_trains, verbose=0)
	# val_loss_2, val_clf_loss_2, val_reg_loss_2, val_clf_accuracy_2 = model_2.evaluate(x_val, y_vals, verbose=0)
	# train_reg_loss_2, val_reg_loss_2 = math.sqrt(train_reg_loss_2), math.sqrt(val_reg_loss_2)

	# print()
	# print('Fold', fold, 'Classification Val Accuracy: {:.2f} %'.format(100*val_clf_accuracy_2))
	# print('Fold', fold, 'Regression Val Score: {:.3f}'.format(val_reg_loss_2))
	# print()

	return train_reg_loss_1, train_clf_accuracy_1, val_reg_loss_1, val_clf_accuracy_1#, train_reg_loss_2, train_clf_accuracy_2, val_reg_loss_2, val_clf_accuracy_2


def create_intervention_model(longest_speaker_length):
	
	input_data = Input(shape=(longest_speaker_length, 3))
	x = LSTM(16)(input_data)
	x = BatchNormalization()(x)
	
	x = Dense(16, activation='relu')(x)
	x = BatchNormalization()(x)
	# x = Dropout(0.2)(x)

	c = Dense(8, activation='relu')(x)
	c = BatchNormalization()(c)
	# c = Dropout(0.2)(c)
	classifier = Dense(2, activation='softmax', name='clf')(c)

	y = Dense(8, activation='relu')(x)
	y = BatchNormalization()(y)
	# y = Dropout(0.2)(y)
	# y = Dense(24, activation='relu')(y)
	# y = Dropout(0.4)(y)
	y = Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01), activity_regularizer=tf.keras.regularizers.l1(0.001))(y)
	regressor = ReLU(max_value=30, name='reg')(y)

	model = Model(inputs=input_data, outputs=[classifier, regressor])
	return model

def create_combined_model(n_features, longest_speaker_length):
	

	inv_input_data = Input(shape=(longest_speaker_length, 3))
	inv_x = LSTM(16)(inv_input_data)
	inv_x = BatchNormalization()(inv_x)
	
	inv_x = Dense(16, activation='relu')(inv_x)
	inv_x = BatchNormalization()(inv_x)
	inv_x = layers.Dropout(0.2)(inv_x)

	input_data = Input(shape=(n_features,))

	x = layers.BatchNormalization()(input_data)
	x = layers.Dense(24, activation='relu')(x)
	x = layers.BatchNormalization()(x)
	x = layers.Dropout(0.2)(x)

	x = layers.Dense(16, activation='relu')(x)
	x = layers.BatchNormalization()(x)
	x = layers.Dropout(0.2)(x)

	# x = layers.Dense(24, activation='relu')(x)
	# x = layers.BatchNormalization()(x)
	# x = layers.Dropout(0.2)(x)

	x = layers.Concatenate()([x, inv_x])
	x = layers.BatchNormalization()(x)
	x = layers.Dropout(0.2)(x)
	
	c = Dense(16, activation='relu')(x)
	c = BatchNormalization()(c)
	c = Dropout(0.2)(c)
	classifier = Dense(2, activation='softmax', name='clf', kernel_regularizer=tf.keras.regularizers.l2(0.01), activity_regularizer=tf.keras.regularizers.l1(0.01))(c)

	y = Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), activity_regularizer=tf.keras.regularizers.l1(0.01))(x)
	y = BatchNormalization()(y)
	y = Dropout(0.2)(y)
	# y = Dense(24, activation='relu')(y)
	# y = Dropout(0.4)(y)
	y = Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01), activity_regularizer=tf.keras.regularizers.l1(0.01))(y)
	regressor = ReLU(max_value=30, name='reg')(y)

	model = Model(inputs=[inv_input_data, input_data], outputs=[classifier, regressor])
	return model

def create_combined_old_model(n_features, longest_speaker_length):
	input_data = Input(shape=(n_features,))

	inv_input_data = Input(shape=(longest_speaker_length, 3))
	inv_x = LSTM(16)(inv_input_data)
	inv_x = BatchNormalization()(inv_x)
	
	inv_x = Dense(16, activation='relu')(inv_x)
	inv_x = BatchNormalization()(inv_x)

	x = layers.BatchNormalization()(input_data)
	x = layers.Dense(16, activation='relu')(x)
	x = layers.BatchNormalization()(x)
	x = layers.Dropout(0.2)(x)

	x = layers.BatchNormalization()(input_data)
	x = layers.Dense(24, activation='relu')(x)
	x = layers.BatchNormalization()(x)
	x = layers.Dropout(0.2)(x)

	x = layers.Dense(24, activation='relu')(x)
	x = layers.BatchNormalization()(x)
	x = layers.Dropout(0.2)(x)

	# x = layers.Dense(24, activation='relu')(x)
	# x = layers.BatchNormalization()(x)
	# x = layers.Dropout(0.2)(x)

	x = layers.Concatenate()([x, inv_x])
	
	c = Dense(16, activation='relu')(x)
	c = BatchNormalization()(c)
	c = Dropout(0.2)(c)
	classifier = Dense(2, activation='softmax', name='clf')(c)

	y = Dense(16, activation='relu')(x)
	y = BatchNormalization()(y)
	# y = Dropout(0.2)(y)
	# y = Dense(24, activation='relu')(y)
	# y = Dropout(0.4)(y)
	y = Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01), activity_regularizer=tf.keras.regularizers.l1(0.001))(y)
	regressor = ReLU(max_value=30, name='reg')(y)

	model = Model(inputs=[inv_input_data, input_data], outputs=[classifier, regressor])
	return model

def create_pause_model(n_features):
	
	input_data = Input(shape=(n_features,))
	x = layers.BatchNormalization()(input_data)
	x = layers.Dense(16, activation='relu')(x)
	x = layers.BatchNormalization()(x)
	x = layers.Dropout(0.2)(x)

	x = layers.Dense(24, activation='relu')(x)
	x = layers.BatchNormalization()(x)
	x = layers.Dropout(0.2)(x)

	x = layers.Dense(24, activation='relu')(x)
	x = layers.BatchNormalization()(x)
	x = layers.Dropout(0.2)(x)
	
	c = Dense(8, activation='relu')(x)
	c = BatchNormalization()(c)
	# c = Dropout(0.2)(c)
	classifier = Dense(2, activation='softmax', name='clf')(c)

	y = Dense(8, activation='relu')(x)
	y = BatchNormalization()(y)
	# y = Dropout(0.2)(y)
	# y = Dense(24, activation='relu')(y)
	# y = Dropout(0.4)(y)
	y = Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01), activity_regularizer=tf.keras.regularizers.l1(0.001))(y)
	regressor = ReLU(max_value=30, name='reg')(y)

	model = Model(inputs=input_data, outputs=[classifier, regressor])
	return model

qty_monitored_1, mode_1 = 'val_reg_loss', 'min'
qty_monitored_2, mode_2 = 'val_clf_categorical_accuracy', 'max'
data = dataset.prepare_data('../ADReSS-IS2020-data/train')
X_intervention, X_pause, X_spec, X_compare, X_reg_intervention, X_reg_pause, X_reg_compare = data[0:7]
y, y_reg, filenames_intervention, filenames_pause, filenames_spec, filenames_compare = data[7:]

if not np.array_equal(X_intervention, X_reg_intervention):
	raise Exception('X classification and regression different')

feature_types = {
	'intervention': X_intervention,
	'pause': X_pause
}

model_dir = 'multihead-models'
model_type = 'combined'
# train_clf_accuracies, train_reg_losses, val_clf_accuracies, val_reg_losses = train_n_folds(model_type, X_intervention, y, y_reg, n_folds=5, model_dir=model_dir)
results = train_n_folds(model_type, feature_types['intervention'], y, y_reg, n_folds=5, model_dir=model_dir, x2 = feature_types['pause'])
# train_clf_accuracies_1, train_reg_losses_1, val_clf_accuracies_1, val_reg_losses_1, train_clf_accuracies_2, train_reg_losses_2, val_clf_accuracies_2, val_reg_losses_2 = results
train_clf_accuracies_1, train_reg_losses_1, val_clf_accuracies_1, val_reg_losses_1 = results

print('\nQuantity monitored: {} \t Model type: {}\n'.format(qty_monitored_1, model_type))
for i,j in enumerate(train_clf_accuracies_1):
	print('Fold {}'.format(i))
	print('Classification Training Accuracy {:.2f} %'.format(100*j))
	print('Classification Validation Accuracy {:.2f} %'.format(100*val_clf_accuracies_1[i]))
	print('Regression Training Score {:.3f}'.format(train_reg_losses_1[i]))
	print('Regression Validation score {:.3f}'.format(val_reg_losses_1[i]))

print('\nMean over all folds:')
print('Classification Training Accuracy {:.2f} %'.format(100*np.mean(train_clf_accuracies_1)))
print('Classification Validation Accuracy {:.2f} %'.format(100*np.mean(val_clf_accuracies_1)))
print('Regression Training Score {:.3f}'.format(np.mean(train_reg_losses_1)))
print('Regression Validation score {:.3f}'.format(np.mean(val_reg_losses_1)))

# print('\nQuantity monitored: {} \t Model type: {}\n'.format(qty_monitored_2, model_type))
# for i,j in enumerate(train_clf_accuracies_2):
# 	print('Fold {}'.format(i))
# 	print('Classification Training Accuracy {:.2f} %'.format(100*j))
# 	print('Classification Validation Accuracy {:.2f} %'.format(100*val_clf_accuracies_2[i]))
# 	print('Regression Training Score {:.3f}'.format(train_reg_losses_2[i]))
# 	print('Regression Validation score {:.3f}'.format(val_reg_losses_2[i]))

# print('\nMean over all folds:')
# print('Classification Training Accuracy {:.2f} %'.format(100*np.mean(train_clf_accuracies_2)))
# print('Classification Validation Accuracy {:.2f} %'.format(100*np.mean(val_clf_accuracies_2)))
# print('Regression Training Score {:.3f}'.format(np.mean(train_reg_losses_2)))
# print('Regression Validation score {:.3f}'.format(np.mean(val_reg_losses_2)))

# print('\nQuantity monitored: {} \t Model type: {}\n'.format(qty_monitored, model_type))
# for i,j in enumerate(train_clf_accuracies):
# 	print('Fold {}'.format(i))
# 	print('Classification Training Accuracy {:.2f} %'.format(100*j))
# 	print('Classification Validation Accuracy {:.2f} %'.format(100*val_clf_accuracies[i]))
# 	print('Regression Training Score {:.3f}'.format(train_reg_losses[i]))
# 	print('Regression Validation score {:.3f}'.format(val_reg_losses[i]))

# print('\nMean over all folds:')
# print('Classification Training Accuracy {:.2f} %'.format(100*np.mean(train_clf_accuracies)))
# print('Classification Validation Accuracy {:.2f} %'.format(100*np.mean(val_clf_accuracies)))
# print('Regression Training Score {:.3f}'.format(np.mean(train_reg_losses)))
# print('Regression Validation score {:.3f}'.format(np.mean(val_reg_losses)))

