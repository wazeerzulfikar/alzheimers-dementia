import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow_probability as tfp

tfd = tfp.distributions

def create_intervention_model(task, longest_speaker_length, uncertainty):
	model = tf.keras.Sequential()
	model.add(layers.LSTM(16, input_shape=(longest_speaker_length, 3)))
	model.add(layers.BatchNormalization())
	model.add(layers.Dropout(0.2))
	model.add(layers.Dense(16, activation='relu'))
	model.add(layers.BatchNormalization())

	if task == 'classification':
		model.add(layers.Dense(2, activation='softmax'))

	elif task == 'regression':
		if uncertainty:
			model.add(layers.Dense(16, activation='relu'))
			model.add(layers.Dropout(0.2))
			model.add(layers.Dense(2))
			model.add(tfp.layers.DistributionLambda(
				lambda t: tfd.Normal(loc=t[..., :1], scale=tf.math.softplus(t[...,1:])+1e-6)))
		else:
			model.add(layers.Dense(16, activation='relu'))
			model.add(layers.Dense(8, activation='relu'))
			model.add(layers.Dense(1))
			model.add(layers.ReLU(max_value=30))

	return model

def create_pause_model(task, n_features, uncertainty):
	model = tf.keras.Sequential()
	model.add(layers.Input(shape=(n_features,)))
	model.add(layers.BatchNormalization())
	model.add(layers.Dense(16, activation='relu'))
	model.add(layers.BatchNormalization())
	model.add(layers.Dropout(0.2))
	model.add(layers.Dense(16, activation='relu'))
	model.add(layers.BatchNormalization())
	model.add(layers.Dropout(0.2))
	model.add(layers.Dense(24, activation='relu'))
	model.add(layers.BatchNormalization())
	model.add(layers.Dropout(0.1))
	model.add(layers.Dense(24, activation='relu'))
	model.add(layers.BatchNormalization())
	model.add(layers.Dropout(0.1))
	
	if task == 'classification':
		model.add(layers.Dense(2, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.01), activity_regularizer=tf.keras.regularizers.l1(0.01)))

	elif task == 'regression':
		if uncertainty:
			model.add(layers.Dense(16, activation='relu'))
			model.add(layers.Dropout(0.2))
			model.add(layers.Dense(2))
			model.add(tfp.layers.DistributionLambda(
				lambda t: tfd.Normal(loc=t[..., :1], scale=tf.math.softplus(t[...,1:])+1e-6)))
		else:
			model.add(layers.Dense(16, activation='relu'))
			model.add(layers.Dense(16, activation='relu'))
			model.add(layers.Dense(1))
			model.add(layers.ReLU(max_value=30))

	return model

def create_compare_model(task, features_size, uncertainty):

	model = tf.keras.Sequential()
	model.add(layers.Input(shape=(features_size,)))
	model.add(layers.Dense(24, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), activity_regularizer=tf.keras.regularizers.l1(0.01)))
	model.add(layers.BatchNormalization())
	model.add(layers.Dropout(0.2))

	if task == 'classification':
		model.add(layers.Dense(2, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.01), activity_regularizer=tf.keras.regularizers.l1(0.01)))

	elif task == 'regression':
		if uncertainty:
			model.add(layers.Dense(16, activation='relu'))
			model.add(layers.Dropout(0.2))
			model.add(layers.Dense(2))
			model.add(tfp.layers.DistributionLambda(
				lambda t: tfd.Normal(loc=t[..., :1], scale=tf.math.softplus(t[...,1:])+1e-6)))
		else:
			model.add(layers.Dense(8, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), activity_regularizer=tf.keras.regularizers.l1(0.01)))
			model.add(layers.Dense(8, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), activity_regularizer=tf.keras.regularizers.l1(0.01)))
			model.add(layers.Dropout(0.5))
			model.add(layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01), activity_regularizer=tf.keras.regularizers.l1(0.01)))
			model.add(layers.ReLU(max_value=30))
	return model

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

def create_silences_model(task, uncertainty):
	model2_input = layers.Input(shape=(800, 1),  name='silences_input')
	model2_input_BN = layers.BatchNormalization()(model2_input)

	model2_hidden1 = layers.Conv1D(16, kernel_size=3, strides=1,
						 activation='relu')(model2_input_BN)
	model2_BN1 = layers.BatchNormalization()(model2_hidden1)
	model2_hidden2 = layers.MaxPool1D()(model2_BN1)
	
	model2_hidden3 = layers.Conv1D(32, kernel_size=3, strides=1,
						 activation='relu')(model2_hidden2)
	model2_BN2 = layers.BatchNormalization()(model2_hidden3)
	model2_hidden4 = layers.MaxPool1D()(model2_BN2)

	model2_hidden5 = layers.Conv1D(64, kernel_size=5, strides=1,
						 activation='relu')(model2_hidden4)
	model2_BN3 = layers.BatchNormalization()(model2_hidden5)
	model2_hidden6 = layers.MaxPool1D()(model2_BN3)

	model2_hidden7 = layers.Conv1D(128, kernel_size=5, strides=1,
						 activation='relu')(model2_hidden6)
	model2_BN4 = layers.BatchNormalization()(model2_hidden7)
	model2_hidden8 = layers.MaxPool1D()(model2_BN4)

	model2_hidden9 = layers.Flatten()(model2_hidden8)
	model2_hidden10 = layers.BatchNormalization()(model2_hidden9)
	model2_hidden11 = layers.Dense(128, activation='relu')(model2_hidden10)
	model2_output = layers.Dropout(0.2)(model2_hidden11)
	if task=='classification':
		model2_output = layers.Dense(2, activation='softmax')(model2_output)
	elif task=='regression':
		if uncertainty:
			model2_output = layers.Dense(2)(model2_output)
			model2_output = layers.ReLU(max_value=30)(model2_output)
			model2_output = tfp.layers.DistributionLambda(
					lambda t: tfd.Normal(loc=t[..., :1], scale=tf.math.softplus(t[...,1:])+1e-6))(model2_output)

		else:	
			model2_output = layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01), activity_regularizer=tf.keras.regularizers.l1(0.01))(model2_output)
			model2_output = layers.ReLU(max_value=30)(model2_output)

	model = Model(model2_input, model2_output)

	return model