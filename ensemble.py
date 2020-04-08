'''
<<<<<<< HEAD
Ensemble model (using spectrogram and interventions) for AD classification.
=======
Convolutional and recurrent models for AD classification.
>>>>>>> 7e9afd57c019d6aee0862d02a894a9a1dd1af0cc
'''


import tensorflow as tf
from tensorflow.keras import layers, regularizers, models, preprocessing
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.models import Model
# from tensorflow.keras.preprocessing import ImageDataGenerator
from tensorflow.keras.callbacks import Callback, TensorBoard, ModelCheckpoint
# from keras_self_attention import SeqSelfAttention
# from attention_keras.layers.attention import AttentionLayer

import sys
# sys.path.insert(0, '../input/attention')
# from seq_self_attention import SeqSelfAttention

import os
import glob
import time

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="4"

import numpy as np
np.random.seed(0)
import cv2
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics


######################### TRANSCRIPTION DATA PREPROCESSING ########################

def process_interventions():
	print('------- CC ------')

	dataset_dir = '../ADReSS-IS2020-data/train/transcription/cc/'
	files = sorted(glob.glob(os.path.join(dataset_dir, '*.cha')))
	print(files)
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

	print()
	print('------- CD ------')

	dataset_dir = '../ADReSS-IS2020-data/train/transcription/cd/'
	files = sorted(glob.glob(os.path.join(dataset_dir, '*.cha')))
	print(files)
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

	X = np.concatenate((all_speakers_cc_binary, all_speakers_cd_binary), axis=0).astype(np.float32)
	y = np.concatenate((y_cc, y_cd), axis=0).astype(np.float32)

	# print(y)
	# p = np.random.permutation(len(X))
	p = np.random.RandomState(seed=42).permutation(len(X))

	X = X[p]
	y = y[p]
	X_shape = X.shape

	num_classes = 2

	return X, y, X_shape, num_classes




######################### SPECTROGRAMS DATA PREPROCESSING ###########################

def process_spectrograms():

	dataset_dir = ''
	dataset_dir = '../spectograms/'
	cc_files = sorted(glob.glob(os.path.join(dataset_dir, 'cc-images/*.png')))
	print()
	print(cc_files)
	X_cc = np.array([cv2.resize(cv2.imread(f), (640,480))/255. for f in cc_files])
	y_cc = np.zeros((X_cc.shape[0], 2))
	y_cc[:,0] = 1

	cd_files = sorted(glob.glob(os.path.join(dataset_dir, 'cd-images/*.png')))
	print()
	print(cd_files)

	X_cd = np.array([cv2.resize(cv2.imread(f), (640,480))/255. for f in cd_files])
	y_cd = np.zeros((X_cd.shape[0], 2))
	y_cd[:,1] = 1

	X = np.concatenate((X_cc, X_cd), axis=0).astype(np.float32)
	y = np.concatenate((y_cc, y_cd), axis=0).astype(np.float32)

	# p = np.random.permutation(len(X))
	p = np.random.RandomState(seed=42).permutation(len(X))

	X = X[p]
	y = y[p]
	X_shape = X.shape

	num_classes = 2

	return X, y, X_shape, num_classes


###################################### MODEL #########################################

def create_model(longest_speaker_length=40, num_classes=2,
	_type_='convolutional', _binning_=False):

	###################### INTERVENTIONS ########################

	model1_input = layers.Input(shape=(longest_speaker_length, 3), name='interventions_input')
	model1_BN = layers.BatchNormalization()(model1_input)
	model1_hidden = layers.LSTM(16, input_shape=(longest_speaker_length, 3))(model1_BN)
	# model1_output = layers.Dropout(0.2)(model1_hidden)
	model1_output = model1_hidden

	model1 = models.Model(model1_input, model1_output)

	###################### SPECTROGRAMS ###########################

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
	# model2_output = layers.Dropout(0.2)(model2_hidden11)
	model2_output = model2_hidden11

	model2 = models.Model(model2_input, model2_output)

	#################### MERGED MODEL #######################

	concatenated = layers.Concatenate()([model1_output, model2_output])
	# drop1 = layers.Dropout(0.5)(concatenated)
	drop1 = layers.BatchNormalization()(concatenated)
	# concatenated = concatenate([model1_output, model2_output])
	out = layers.Dense(2, activation='softmax', name='output_layer')(drop1)

	merged_model = models.Model([model1_input, model2_input], out)

	return merged_model


class DataGenerator():

	def __init__(self, X_spectrograms, X_interventions, y, batch_size):

		self.X_spectrograms = X_spectrograms
		self.X_interventions = X_interventions
		self.y = y
		self.batch_size = batch_size
		self.n_samples = self.X_interventions.shape[0]
		self.datagen = tf.keras.preprocessing.image.ImageDataGenerator(
			horizontal_flip=True)
			# samplewise_center=True,
			# samplewise_std_normalization=True)

	def random_crop(self, images, width_crop_size=320):
		height, width, channels = images.shape[1:]
		cropped_images = images
		for i in range(len(images)):
			if np.random.random()<0.5:
				img = images[i]
				pad_width_size = np.random.randint(0, width//4)
				pad_start = np.random.randint(0, width - pad_width_size)

				cropped_img_left = img[:, :pad_start, :]
				cropped_img_right = img[:, pad_start+pad_width_size:, :]
				padding = np.zeros((height, pad_width_size, channels))
				cropped_img = np.concatenate((cropped_img_left, padding, cropped_img_right), axis=1)

				cropped_images[i] = cropped_img

		# 		cv2.imwrite( 'crop{}.jpg'.format(i), cropped_img*255.)
		# exit()
		return cropped_images


	def __len__(self):
		return self.n_samples//self.batch_size

	def flow(self):
		while True:
			p = np.random.permutation(len(self.X_spectrograms))
			self.X_spectrograms = self.X_spectrograms[p]
			self.X_interventions = self.X_interventions[p]
			self.y = self.y[p]
			batch_n = 0

			for x_batch_spectograms, y_batch in self.datagen.flow(self.X_spectrograms, self.y, batch_size=self.batch_size):
				if batch_n>self.n_samples-self.batch_size:
					break
				x_batch_interventions = self.X_interventions[batch_n:batch_n+batch_size]
				batch_n += batch_size
				x_batch_spectograms = self.random_crop(x_batch_spectograms)
				yield ([x_batch_interventions, x_batch_spectograms], y_batch)

	def on_epoch_end():
		print('Epoch Done!')

print(create_model().summary())

n_split = 5
epochs = 75
batch_size = 8

training_accuracies, validation_accuracies, f1_scores = [], [], []
X_interventions, y_interventions, X_interventions_shape, num_classes1 = process_interventions()
X_spectrograms, y_spectrograms, X_spectrograms_shape, num_classes2 = process_spectrograms()


# print(y_interventions)
# print('###############################')
# print(num_classes1)
# print('###############################')
# print(y_spectrograms)
# print('###############################')
# print(num_classes2)
# print('###############################')

#################### CROSS VALIDATED MODEL TRAINING ########################

fold = 0
for train_index, val_index in KFold(n_split, shuffle=True, random_state=13).split(X_interventions):

	print('FOLD ###########', fold)
	x_train_interventions, x_val_interventions = X_interventions[train_index], X_interventions[val_index]
	y_train_interventions, y_val_interventions = y_interventions[train_index], y_interventions[val_index]

	x_train_spectrograms, x_val_spectrograms = X_spectrograms[train_index], X_spectrograms[val_index]
	y_train_spectrograms, y_val_spectrograms = y_spectrograms[train_index], y_spectrograms[val_index]
	
	# temp = x_train_spectrograms
	# print('#####################', x_train_spectrograms.shape)
	# datagen = preprocessing.image.ImageDataGenerator(horizontal_flip=True)
	# datagen.fit(x_train_spectrograms)
	# x_train_spectrograms = datagen.flow(x_train_spectrograms)
	# print([i for i in x_train_spectrograms])
	# print('#####################', x_train_spectrograms.shape)
	# if temp.all()==x_train_spectrograms.all():		print("FLIPS NOT DONE")

	if y_train_interventions.all()==y_train_spectrograms.all():		y_train = y_train_interventions
	if y_val_interventions.all()==y_val_spectrograms.all():		y_val = y_val_interventions

	model = create_model()
	datagen = DataGenerator(x_train_spectrograms, x_train_interventions, y_train, batch_size)

	timeString = time.strftime("%Y%m%d-%H%M%S", time.localtime())
	log_name = "{}".format(timeString)

	tensorboard = TensorBoard(log_dir="logs/{}".format(log_name), histogram_freq=1, write_graph=True, write_images=False)
	# mc = ModelCheckpoint('best_model_f{}.h5'.format(fold), monitor='val_acc', mode='max', save_best_only=True,
 #                             verbose=0)

	model.compile(loss=tf.keras.losses.categorical_crossentropy,
				  optimizer=tf.keras.optimizers.Adam(lr=0.001),
				  metrics=['categorical_accuracy'])

	# print(x_train_interventions.shape, x_train_spectrograms.shape, y_train.shape)
	# print(x_val_interventions.shape, x_val_spectrograms.shape, y_val.shape)

	print('Y validation counts', np.unique(np.argmax(y_val, axis=-1), return_counts=True))

	model.fit(datagen.flow(),
			  epochs=epochs,
			  steps_per_epoch=datagen.__len__(),
			  verbose=1,
			  callbacks=[tensorboard],
			  validation_data=([x_val_interventions, x_val_spectrograms], y_val))

	# model.fit([x_train_interventions, x_train_spectrograms], y_train,
	# 		  batch_size=batch_size,
	# 		  epochs=epochs,
	# 		  verbose=1,
	# 		  callbacks=[tensorboard],
	# 		  validation_data=([x_val_interventions, x_val_spectrograms], y_val))
	# model = load_model('best_model_f{}.h5'.format(fold))
	training_accuracy = model.evaluate([x_train_interventions, x_train_spectrograms], y_train, verbose=0)
	validation_accuracy = model.evaluate([x_val_interventions, x_val_spectrograms], y_val, verbose=0)

	predictions_probs = model.predict([x_val_interventions, x_val_spectrograms])
	predictions_ = np.argmax(predictions_probs, axis=-1)
	true_values_ = np.argmax(y_val, axis=-1)

	precision, recall, f1, _ = precision_recall_fscore_support(list(true_values_), list(predictions_))
	
	training_accuracies.append(training_accuracy[1])
	validation_accuracies.append(validation_accuracy[1])
	f1_scores.append(f1)

	print()
	# print('Predictions probabilities: ', predictions_probs)
	
	for i in range(predictions_.shape[0]):
		print('Prediction probability: ', np.max(predictions_probs[i]))
		print('Prediction: ', predictions_[i])
		print('True value : ', true_values_[i])
		print()
	print('Val accuracy: ', validation_accuracy[1], '\tVal F1: ', f1)
	print('Training Accuracies till now : ', training_accuracies, '\tValidation Accuracies till now : ', validation_accuracies, '\tF1 Scores till now : ', f1_scores)
	print()
	
	fold+=1
	# exit()

print('Validation Accuracy : ', np.mean(validation_accuracies), ' +- ', np.std(validation_accuracies))
print('F1 Score : ', np.mean(f1_scores), ' +- ', np.std(f1_scores))



