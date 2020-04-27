import glob, os, math, time
import numpy as np
np.random.seed(0)
p = np.random.permutation(108) # n_samples = 108

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

import dataset_features, dataset_utils

def prepare_data():
	################################## INTERVENTION ####################################

	longest_speaker_length = 32
	dataset_dir = '../ADReSS-IS2020-data/train'

	cc_files = sorted(glob.glob(os.path.join(dataset_dir, 'transcription/cc/*.cha')))
	all_speakers_cc = []
	for filename in cc_files:
		all_speakers_cc.append(dataset_features.get_intervention_features(filename, longest_speaker_length))

	cd_files = sorted(glob.glob(os.path.join(dataset_dir, 'transcription/cd/*.cha')))
	all_speakers_cd = []
	for filename in cd_files:
		all_speakers_cd.append(dataset_features.get_intervention_features(filename, longest_speaker_length))

	### Classification X and y values
	y_cc = np.zeros((len(all_speakers_cc), 2))
	y_cc[:,0] = 1

	y_cd = np.zeros((len(all_speakers_cd), 2))
	y_cd[:,1] = 1

	X_intervention = np.concatenate((all_speakers_cc, all_speakers_cd), axis=0).astype(np.float32)
	y_intervention = np.concatenate((y_cc, y_cd), axis=0).astype(np.float32)
	filenames_intervention = np.concatenate((cc_files, cd_files), axis=0)
	################################

	### Regression X and y values
	y_reg_cc = dataset_utils.get_regression_values(os.path.join(dataset_dir, 'cc_meta_data.txt'))
	y_reg_cd = dataset_utils.get_regression_values(os.path.join(dataset_dir, 'cd_meta_data.txt'))

	y_reg_intervention = np.concatenate((y_reg_cc, y_reg_cd), axis=0).astype(np.float32)
	X_reg_intervention = np.copy(X_intervention)
	#################################
	################################## INTERVENTION ####################################

	################################## PAUSE ####################################
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

	X_pause = np.concatenate((all_counts_cc, all_counts_cd), axis=0).astype(np.float32)

	### Regression y values
	y_reg_cc = dataset_utils.get_regression_values(os.path.join(dataset_dir, 'cc_meta_data.txt'))
	y_reg_cd = dataset_utils.get_regression_values(os.path.join(dataset_dir, 'cd_meta_data.txt'))

	y_reg_pause = np.concatenate((y_reg_cc, y_reg_cd), axis=0).astype(np.float32)
	#######################

	### Regression X values
	X_reg_pause = np.copy(X_pause)
	#######################

	### Classification y values
	y_cc = np.zeros((len(all_counts_cc), 2))
	y_cc[:,0] = 1

	y_cd = np.zeros((len(all_counts_cd), 2))
	y_cd[:,1] = 1

	y_pause = np.concatenate((y_cc, y_cd), axis=0).astype(np.float32)
	filenames_pause = np.concatenate((cc_transcription_files, cd_transcription_files), axis=0)
	################################## PAUSE ####################################

	################################## SPECTROGRAM ####################################
	dataset_dir = '../spectograms/'
	cc_files = sorted(glob.glob(os.path.join(dataset_dir, 'cc-images/*.png')))
	spectogram_size = (480, 640)
	X_cc = np.array([dataset_features.get_spectogram_features(f, spectogram_size[::-1]) for f in cc_files])
	y_cc = np.zeros((X_cc.shape[0], 2))
	y_cc[:,0] = 1

	cd_files = sorted(glob.glob(os.path.join(dataset_dir, 'cd-images/*.png')))
	X_cd = np.array([dataset_features.get_spectogram_features(f, spectogram_size[::-1]) for f in cd_files])
	y_cd = np.zeros((X_cd.shape[0], 2))
	y_cd[:,1] = 1

	X_spec = np.concatenate((X_cc, X_cd), axis=0).astype(np.float32)
	y_spec = np.concatenate((y_cc, y_cd), axis=0).astype(np.float32)
	filenames_spec = np.concatenate((cc_files, cd_files), axis=0)
	################################## SPECTROGRAM ####################################

	assert np.array_equal(y_intervention, y_pause) and np.array_equal(y_pause, y_spec) and np.array_equal(y_reg_intervention, y_reg_pause) and X_intervention.shape[0]==X_pause.shape[0] and X_intervention.shape[0]==X_spec.shape[0] and np.array_equal(filenames_intervention, filenames_pause), 'Data streams are different'
	print('Data streams verified')
	y = y_intervention
	y_reg = y_reg_intervention
	X_length = X_intervention.shape[0] # 108

	X_intervention, X_pause, X_spec = X_intervention[p], X_pause[p], X_spec[p]
	X_reg_intervention, X_reg_pause = X_reg_intervention[p], X_reg_pause[p]
	y, y_reg = y[p], y_reg[p]
	filenames_intervention, filenames_pause, filenames_spec = filenames_intervention[p], filenames_pause[p], filenames_spec[p]

	return X_intervention, X_pause, X_spec, X_reg_intervention, X_reg_pause, y, y_reg, filenames_intervention, filenames_pause, filenames_spec


def intervention(X, y, filenames, voting_type, longest_speaker_length=32, loocv=False):

	def create_model(longest_speaker_length):
		model = tf.keras.Sequential()
		model.add(layers.LSTM(8, input_shape=(longest_speaker_length, 3)))
		model.add(layers.BatchNormalization())
		model.add(layers.Dropout(0.2))
		model.add(layers.Dense(2, activation='softmax'))
		return model

	def training():

		if loocv==True:
			n_split = X.shape[0]
			model_dir = 'loocv-models-intervention'
		else:
			n_split = 5
			model_dir = '5-fold-models-intervention'

		epochs = 400
		batch_size = 8
		val_accuracies, val_losses = [], []
		train_accuracies, train_losses = [], []
		models = []
		fold = 0

		for train_index, val_index in KFold(n_split).split(X):
			fold+=1

			x_train, x_val = X[train_index], X[val_index]
			y_train, y_val = y[train_index], y[val_index]
			filenames_train, filenames_val = filenames[train_index], filenames[val_index]

			model = create_model(longest_speaker_length)

			checkpointer = tf.keras.callbacks.ModelCheckpoint(
				os.path.join(model_dir, voting_type, 'intervention_{}.h5'.format(fold)), monitor='val_loss', verbose=0, save_best_only=False,
				save_weights_only=False, mode='auto', save_freq='epoch'
			)

			model.compile(loss=tf.keras.losses.categorical_crossentropy,
						  optimizer=tf.keras.optimizers.Adam(lr=0.001),
						  metrics=['categorical_accuracy'])
			model.fit(x_train, y_train,
					  batch_size=batch_size,
					  epochs=epochs,
					  verbose=1,
					  validation_data=(x_val, y_val),
					  callbacks=[checkpointer])
			model = tf.keras.models.load_model(os.path.join(model_dir, voting_type, 'intervention_{}.h5'.format(fold)))
			models.append(model)

			val_pred = model.predict(x_val)
			for i in range(len(x_val)):
				print(filenames_val[i], np.argmax(val_pred[i])==np.argmax(y_val[i]), val_pred[i])

			train_score = model.evaluate(x_train, y_train, verbose=0)
			train_accuracies.append(train_score[1])
			train_losses.append(train_score[0])

			val_score = model.evaluate(x_val, y_val, verbose=0)
			print('Val accuracy:', val_score[1])
			val_accuracies.append(val_score[1])
			val_losses.append(val_score[0])
			print('Val mean till fold {} is {}'.format(fold, np.mean(val_accuracies)))

		return train_accuracies, val_accuracies, train_losses, val_losses, models

	train_accuracies, val_accuracies, train_losses, val_losses, models = training()
	return train_accuracies, val_accuracies, train_losses, val_losses, models


def pause(X, y, filenames, voting_type, loocv=False):

	def create_model():
		model = tf.keras.Sequential()
		model.add(layers.Input(shape=(11,)))
		model.add(layers.BatchNormalization())
		model.add(layers.Dense(16, activation='relu'))
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

	def training():

		if loocv==True:
			n_split = X.shape[0]
			model_dir = 'loocv-models-pause'
		else:
			n_split = 5
			model_dir = '5-fold-models-pause'

		epochs = 600
		batch_size = 8
		val_accuracies, val_losses = [], []
		train_accuracies, train_losses = [], []
		models = []
		fold = 0

		for train_index, val_index in KFold(n_split).split(X):
			fold+=1

			x_train, x_val = X[train_index], X[val_index]
			y_train, y_val = y[train_index], y[val_index]
			filenames_train, filenames_val = filenames[train_index], filenames[val_index]

			model = create_model()

			checkpointer = tf.keras.callbacks.ModelCheckpoint(
					os.path.join(model_dir, voting_type, 'pause_{}.h5'.format(fold)), monitor='val_loss', verbose=0, save_best_only=True,
					save_weights_only=False, mode='auto', save_freq='epoch')

			model.compile(loss=tf.keras.losses.categorical_crossentropy,
						  optimizer=tf.keras.optimizers.Adam(lr=0.001),
						  metrics=['categorical_accuracy'])
			model.fit(x_train, y_train,
					  batch_size=batch_size,
					  epochs=epochs,
					  verbose=1,
					  callbacks=[checkpointer],
					  validation_data=(x_val, y_val))
			model = tf.keras.models.load_model(os.path.join(model_dir, voting_type, 'pause_{}.h5'.format(fold)))
			models.append(model)

			val_pred = model.predict(x_val)
			for i in range(len(x_val)):
				print(filenames_val[i], np.argmax(val_pred[i])==np.argmax(y_val[i]), val_pred[i])

			train_score = model.evaluate(x_train, y_train, verbose=0)
			train_accuracies.append(train_score[1])
			train_losses.append(train_score[0])

			val_score = model.evaluate(x_val, y_val, verbose=0)
			print('Val accuracy:', val_score[1])
			val_accuracies.append(val_score[1])
			val_losses.append(val_score[0])
			print('Val mean till fold {} is {}'.format(fold, np.mean(val_accuracies)))

		return train_accuracies, val_accuracies, train_losses, val_losses, models

	train_accuracies, val_accuracies, train_losses, val_losses, models = training()
	return train_accuracies, val_accuracies, train_losses, val_losses, models

def spectrogram(X, y, filenames, voting_type, loocv=False):

	def create_model():
		input_shape_ = (480, 640, 3)
		model2_input = layers.Input(shape=input_shape_,  name='spectrogram_input')
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

	def training():

		if loocv==True:
			n_split = X.shape[0]
			model_dir = 'loocv-models-spectrogram'
		else:
			n_split = 5
			model_dir = '5-fold-models-spectrogram'

		epochs = 50
		batch_size = 8
		val_accuracies, val_losses = [], []
		train_accuracies, train_losses = [], []
		models = []
		fold = 0

		for train_index, val_index in KFold(n_split).split(X):
			fold+=1

			x_train, x_val = X[train_index], X[val_index]
			y_train, y_val = y[train_index], y[val_index]
			filenames_train, filenames_val = filenames[train_index], filenames[val_index]

			model = create_model()

			checkpointer = tf.keras.callbacks.ModelCheckpoint(
					os.path.join(model_dir, voting_type, 'spec_{}.h5'.format(fold)), monitor='val_loss', verbose=0, save_best_only=False,
					save_weights_only=False, mode='auto', save_freq='epoch'
				)
			model.compile(loss=tf.keras.losses.categorical_crossentropy,
						  optimizer=tf.keras.optimizers.Adam(lr=0.001, epsilon=0.1),
						  metrics=['categorical_accuracy'])
			model.fit(x_train, y_train,
					  batch_size=batch_size,
					  epochs=epochs,
					  verbose=1,
					  callbacks=[checkpointer],
					  validation_data=(x_val, y_val))
			model = tf.keras.models.load_model(os.path.join(model_dir, voting_type, 'spec_{}.h5'.format(fold)))
			models.append(model)					
			val_pred = model.predict(x_val)
			for i in range(len(x_val)):
				print(filenames_val[i], np.argmax(val_pred[i])==np.argmax(y_val[i]), val_pred[i])

			train_score = model.evaluate(x_train, y_train, verbose=0)
			train_accuracies.append(train_score[1])
			train_losses.append(train_score[0])

			val_score = model.evaluate(x_val, y_val, verbose=0)
			print('Val accuracy:', val_score[1])
			val_accuracies.append(val_score[1])
			val_losses.append(val_score[0])
			print('Val mean till fold {} is {}'.format(fold, np.mean(val_accuracies)))

		return train_accuracies, val_accuracies, train_losses, val_losses, models

	train_accuracies, val_accuracies, train_losses, val_losses, models = training()
	return train_accuracies, val_accuracies, train_losses, val_losses, models

def ensemble(voting_type, models=None):
	
	X_intervention, X_pause, X_spec, X_reg_intervention, X_reg_pause, y, y_reg, filenames_intervention, filenames_pause, filenames_spec = prepare_data()

	if models==None:
		train_accuracies_inv, val_accuracies_inv, train_losses_inv, val_losses_inv, models_inv = intervention(X_intervention, y, filenames_intervention, voting_type)
		train_accuracies_pause, val_accuracies_pause, train_losses_pause, val_losses_pause, models_pause = pause(X_pause, y, filenames_pause, voting_type)
		train_accuracies_spec, val_accuracies_spec, train_losses_spec, val_losses_spec, models_spec = spectrogram(X_spec, y, filenames_spec, voting_type)
	elif models=='load':
		models_inv, models_pause, models_spec = [], [], []
		train_accuracies_inv, val_accuracies_inv, train_accuracies_pause, val_accuracies_pause, train_accuracies_spec, val_accuracies_spec = [], [], [], [], [], []
		fold = 0
		n_split = 5
		for train_index, val_index in KFold(n_split).split(X_pause):
			fold+=1
			pause_train, pause_val = X_pause[train_index], X_pause[val_index]
			spec_train, spec_val = X_spec[train_index], X_spec[val_index]
			inv_train, inv_val = X_intervention[train_index], X_intervention[val_index]
			y_train, y_val = y[train_index], y[val_index]

			inv = tf.keras.models.load_model(os.path.join('5-fold-models-intervention', voting_type, 'intervention_{}.h5'.format(fold)))
			models_inv.append(inv)
			train_accuracies_inv.append(inv.evaluate(inv_train, y_train, verbose=0)[1])
			val_accuracies_inv.append(inv.evaluate(inv_val, y_val, verbose=0)[1])

			pause = tf.keras.models.load_model(os.path.join('5-fold-models-pause', voting_type, 'pause_{}.h5'.format(fold)))
			models_pause.append(pause)
			train_accuracies_pause.append(pause.evaluate(pause_train, y_train, verbose=0)[1])
			val_accuracies_pause.append(pause.evaluate(pause_val, y_val, verbose=0)[1])

			# spec = tf.keras.models.load_model(os.path.join('5-fold-models-spectrogram', voting_type, 'spec_{}.h5'.format(fold)))
			# models_spec.append(spec)
			# train_accuracies_spec.append(spec.evaluate(spec_train, y_train, verbose=0)[1])
			# val_accuracies_spec.append(spec.evaluate(spec_val, y_val, verbose=0)[1])		

	fold = 0
	n_split = 5

	train_accuracies_ensemble, val_accuracies_ensemble = [], []

	for train_index, val_index in KFold(n_split).split(X_pause):

		pause_train, pause_val = X_pause[train_index], X_pause[val_index]
		spec_train, spec_val = X_spec[train_index], X_spec[val_index]
		inv_train, inv_val = X_intervention[train_index], X_intervention[val_index]
		y_train, y_val = y[train_index], y[val_index]

		################################# TRAINING #################################
		pause_probs = models_pause[fold].predict(pause_train)
		# spec_probs = models_spec[fold].predict(spec_train)
		inv_probs = models_inv[fold].predict(inv_train)		

		if voting_type=='hard_voting':
			model_predictions = [[np.argmax(pause_probs[i]), np.argmax(spec_probs[i]), np.argmax(inv_probs[i])] for i in range(len(y_train))]
			voted_predictions = [max(set(i), key = i.count) for i in model_predictions]

		elif voting_type=='soft_voting':
			model_predictions = pause_probs + inv_probs
			voted_predictions = np.argmax(model_predictions, axis=-1)

		elif voting_type=='learnt_voting':
			model_predictions = np.concatenate((pause_probs, inv_probs), axis=-1)
			voter = LogisticRegression().fit(model_predictions, np.argmax(y_train, axis=-1))
			voted_predictions = voter.predict(model_predictions)

		train_accuracy = accuracy_score(np.argmax(y_train, axis=-1), voted_predictions)
		train_accuracies_ensemble.append(train_accuracy)

		################################ VALIDATION ################################
		pause_probs = models_pause[fold].predict(pause_val)
		# spec_probs = models_spec[fold].predict(spec_val)
		inv_probs = models_inv[fold].predict(inv_val)

		if voting_type=='hard_voting':
			model_predictions = [[np.argmax(pause_probs[i]), np.argmax(spec_probs[i]), np.argmax(inv_probs[i])] for i in range(len(y_val))]
			voted_predictions = [max(set(i), key = i.count) for i in model_predictions]
		elif voting_type=='soft_voting':
			model_predictions = pause_probs + inv_probs
			# for i,j in zip(model_predictions, y_val):
			# 	print(i, '\t', j)
			# exit()
			voted_predictions = np.argmax(model_predictions, axis=-1)
		elif voting_type=='learnt_voting':
			model_predictions = np.concatenate((pause_probs, inv_probs), axis=-1)
			voted_predictions = voter.predict(model_predictions)

		val_accuracy = accuracy_score(np.argmax(y_val, axis=-1), voted_predictions)		
		val_accuracies_ensemble.append(val_accuracy)

		fold+=1

	return train_accuracies_inv, val_accuracies_inv, train_accuracies_pause, val_accuracies_pause, train_accuracies_ensemble, val_accuracies_ensemble

def ensemble_boost_sampling(voting_type, loocv=False):

	def pause_training(x_train, y_train, x_val, y_val, fold):

		if loocv==True:
			n_split = X_pause.shape[0]
			model_dir = 'loocv-models-pause'
		else:
			n_split = 5
			model_dir = '5-fold-models-pause'

		epochs = 600
		batch_size = 8

		def create_model():
			model = tf.keras.Sequential()
			model.add(layers.Input(shape=(11,)))
			model.add(layers.BatchNormalization())
			model.add(layers.Dense(16, activation='relu'))
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

		model = create_model()

		checkpointer = tf.keras.callbacks.ModelCheckpoint(
				os.path.join(model_dir, voting_type, 'pause_{}.h5'.format(fold)), monitor='val_loss', verbose=0, save_best_only=True,
				save_weights_only=False, mode='auto', save_freq='epoch')

		model.compile(loss=tf.keras.losses.categorical_crossentropy,
					  optimizer=tf.keras.optimizers.Adam(lr=0.001),
					  metrics=['categorical_accuracy'])
		model.fit(x_train, y_train,
				  batch_size=batch_size,
				  epochs=epochs,
				  verbose=1,
				  callbacks=[checkpointer],
				  validation_data=(x_val, y_val))
		model = tf.keras.models.load_model(os.path.join(model_dir, voting_type, 'pause_{}.h5'.format(fold)))

		train_score = model.evaluate(x_train, y_train, verbose=0)
		train_accuracy = train_score[1]
		train_loss = train_score[0]

		val_score = model.evaluate(x_val, y_val, verbose=0)
		print('Val accuracy:', val_score[1])
		val_accuracy = val_score[1]
		val_loss = val_score[0]

		return train_accuracy, val_accuracy, train_loss, val_loss

	X_intervention, X_pause, X_spec, X_reg_intervention, X_reg_pause, y, y_reg, filenames_intervention, filenames_pause, filenames_spec = prepare_data()
	n_samples = X_pause.shape[0]
	n_split = 5

	# early_folds = list(range(n_samples % n_split))
	# late_folds = [i for i in list(range(n_split)) if i not in early_folds]
	# early_samples = n_samples // n_split + 1
	# late_samples = n_samples // n_split

	train_accuracies_inv, val_accuracies_inv, train_losses_inv, val_losses_inv, models_inv = intervention(X_intervention, y, filenames_intervention, voting_type)
	train_accuracies_pause, val_accuracies_pause, train_losses_pause, val_losses_pause = [], [], [], []

	for fold in range(1,6):

		inv = tf.keras.models.load_model(os.path.join('5-fold-models-intervention', voting_type, 'intervention_{}.h5'.format(fold)))
		inv_losses = []
		for idx, x in enumerate(X_intervention):
			loss = inv.evaluate(np.expand_dims(x, axis=0), np.expand_dims(y[idx], axis=0), verbose=0)[0]
			inv_losses.append(loss)
		inv_probs = [float(i)/sum(inv_losses) for i in inv_losses] # normalizing losses into probs to sum to 1
		train_index = np.random.choice(n_samples, (n_samples // n_split + 1)*(n_split-1), replace=False, p=inv_probs)
		val_index = np.array([i for i in np.arange(n_samples) if i not in train_index])
		pause_train, pause_val = X_pause[train_index], X_pause[val_index]
		y_train, y_val = y[train_index], y[val_index]
		train_accuracy, val_accuracy, train_loss, val_loss = pause_training(pause_train, y_train, pause_val, y_val, fold)
		train_accuracies_pause.append(train_accuracy)
		val_accuracies_pause.append(val_accuracy)
		train_losses_pause.append(train_loss)
		val_losses_pause.append(val_loss)

	train_accuracies_inv_, val_accuracies_inv_, train_accuracies_pause_, val_accuracies_pause_, train_accuracies_ensemble_, val_accuracies_ensemble_ = ensemble(voting_type, models='load')
	if np.array_equal(train_accuracies_inv, train_accuracies_inv_) and np.array_equal(val_accuracies_inv, val_accuracies_inv_):
		print('Interventions Accuracies verified')
	else:
		print('Interventions Accuracies different')
		exit()
	
	return train_accuracies_inv, val_accuracies_inv, train_accuracies_pause, val_accuracies_pause, train_accuracies_ensemble_, val_accuracies_ensemble_

# def ensemble_boost_input(voting_type, loocv=False):

def one_model(longest_speaker_length, loocv=False): # nan loss
	
	X_intervention, X_pause, X_spec, X_reg_intervention, X_reg_pause, y, y_reg, filenames_intervention, filenames_pause, filenames_spec = prepare_data()
	
	def create_model(longest_speaker_length):

		model1_input = layers.Input(shape=(longest_speaker_length, 3))
		x = layers.LSTM(8)(model1_input)
		x = layers.BatchNormalization()(x)
		x = layers.Dropout(0.2)(x)
		model1_output = layers.Flatten()(x)
		# model1 = Model(model1_input, model1_output)

		model3_input = layers.Input(shape=(11,))
		x = layers.BatchNormalization()(model3_input)
		x = layers.Dense(16, activation='relu')(x)
		x = layers.BatchNormalization()(x)
		x = layers.Dropout(0.5)(x)
		x = layers.Dense(32, activation='relu')(x)
		x = layers.BatchNormalization()(x)
		x = layers.Dropout(0.5)(x)
		x = layers.Dense(16, activation='relu')(x)
		x = layers.BatchNormalization()(x)
		x = layers.Dropout(0.5)(x)
		model3_output = layers.Flatten()(x)
		# model3 = Model(model3_input, model3_output)

		input_shape_ = (480, 640, 3)
		model2_input = layers.Input(shape=input_shape_,  name='spectrogram_input')
		x = layers.BatchNormalization()(model2_input)
		x = layers.Conv2D(16, kernel_size=(3, 3), strides=(1, 1),
							 activation='relu')(x)
		x = layers.BatchNormalization()(x)
		x = layers.MaxPool2D()(x)
		x = layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
							 activation='relu')(x)
		x = layers.BatchNormalization()(x)
		x = layers.MaxPool2D()(x)
		x = layers.Conv2D(64, kernel_size=(5, 5), strides=(1, 1),
							 activation='relu')(x)
		x = layers.BatchNormalization()(x)
		x = layers.MaxPool2D()(x)
		x = layers.Conv2D(128, kernel_size=(5, 5), strides=(1, 1),
							 activation='relu')(x)
		x = layers.BatchNormalization()(x)
		x = layers.MaxPool2D()(x)
		x = layers.Flatten()(x)
		x = layers.BatchNormalization()(x)
		x = layers.Dense(128, activation='relu')(x)
		x = layers.Dropout(0.2)(x)
		model2_output = layers.Flatten()(x)
		# model2 = Model(model2_input, model2_output)

		x = tf.keras.layers.Concatenate()([model1_output, model3_output, model2_output])
		x = layers.Dropout(0.2)(x)
		x = layers.Dense(32, activation='relu')(x)
		x = layers.Dropout(0.2)(x)
		model_output = layers.Dense(2, activation='relu')(x)
		model = Model(inputs=[model1_input, model3_input, model2_input], outputs=model_output)
		print(model.summary())

		return model

	def training():

		if loocv==True:
			n_split = X.shape[0]
			model_dir = 'loocv-models-one_model'
		else:
			n_split = 5
			model_dir = '5-fold-models-one_model'

		epochs = 50
		batch_size = 8
		val_accuracies, val_losses = [], []
		train_accuracies, train_losses = [], []
		models = []
		fold = 0

		for train_index, val_index in KFold(n_split).split(X_intervention):
			fold+=1

			pause_train, pause_val = X_pause[train_index], X_pause[val_index]
			spec_train, spec_val = X_spec[train_index], X_spec[val_index]
			inv_train, inv_val = X_intervention[train_index], X_intervention[val_index]
			y_train, y_val = y[train_index], y[val_index]

			model = create_model(longest_speaker_length)

			checkpointer = tf.keras.callbacks.ModelCheckpoint(
					os.path.join(model_dir, 'one_{}.h5'.format(fold)), monitor='val_loss', verbose=0, save_best_only=False,
					save_weights_only=False, mode='auto', save_freq='epoch'
				)
			model.compile(loss=tf.keras.losses.categorical_crossentropy,
						  optimizer=tf.keras.optimizers.Adam(lr=0.00001),
						  metrics=['categorical_accuracy'])
			model.fit([inv_train, pause_train, spec_train], y_train,
					  batch_size=batch_size,
					  epochs=epochs,
					  verbose=1,
					  callbacks=[checkpointer],
					  validation_data=([inv_val, pause_val, spec_val], y_val))
			model = tf.keras.models.load_model(os.path.join(model_dir, 'one_{}.h5'.format(fold)))
			models.append(model)					

			train_score = model.evaluate([inv_train, pause_train, spec_train], y_train, verbose=0)
			train_accuracies.append(train_score[1])
			train_losses.append(train_score[0])

			val_score = model.evaluate([inv_val, pause_val, spec_val], y_val, verbose=0)
			print('Val accuracy:', val_score[1])
			val_accuracies.append(val_score[1])
			val_losses.append(val_score[0])
			print('Val mean till fold {} is {}'.format(fold, np.mean(val_accuracies)))

		return train_accuracies, val_accuracies, train_losses, val_losses, models

	train_accuracies, val_accuracies, train_losses, val_losses, models = training()
	return train_accuracies, val_accuracies, train_losses, val_losses, models


train_accuracies_inv, val_accuracies_inv, train_accuracies_pause, val_accuracies_pause, train_accuracies_ensemble, val_accuracies_ensemble = ensemble_boost_sampling(voting_type='soft_voting')
# one_model(longest_speaker_length=32)
# ensemble(voting_type='soft_voting', models='load')
#### PRINTING
voting_type='soft_voting'
for fold in range(5):
	print('')
	print('-'*50)
	print('Fold {}'.format(fold))
	print('-'*50)
	print('Interventions :: \t Train Accuracy: {:.3f} \t Val Accuracy: {:.3f}'.format(train_accuracies_inv[fold], val_accuracies_inv[fold]))
	print('Pause :: \t Train Accuracy {:.3f} \t Val Accuracy: {:.3f}'.format(train_accuracies_pause[fold], val_accuracies_pause[fold]))
	# print('Spectrogram :: \t Train Accuracy: {:.3f} \t Val Accuracy: {:.3f}'.format(train_accuracies_spec[fold], val_accuracies_spec[fold]))
	print('Ensemble {} :: \t Train Accuracy: {:.3f} \t Val Accuracy: {:.3f}'.format(voting_type, train_accuracies_ensemble[fold], val_accuracies_ensemble[fold]))

print('')
print('-'*50)
print('Mean over all folds')
print('-'*50)
print('Interventions :: \t Train Accuracy: {:.3f} \t Val Accuracy: {:.3f}'.format(np.mean(train_accuracies_inv), np.mean(val_accuracies_inv)))
print('Pause :: \t Train Accuracy {:.3f} \t Val Accuracy: {:.3f}'.format(np.mean(train_accuracies_pause), np.mean(val_accuracies_pause)))
# print('Spectrogram :: \t Train Accuracy: {:.3f} \t Val Accuracy: {:.3f}'.format(np.mean(train_accuracies_spec), np.mean(val_accuracies_spec)))
print('Ensemble {} :: \t Train Accuracy: {:.3f} \t Val Accuracy: {:.3f}'.format(voting_type, np.mean(train_accuracies_ensemble), np.mean(val_accuracies_ensemble)))

