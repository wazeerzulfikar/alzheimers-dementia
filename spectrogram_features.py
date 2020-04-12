'''
Convolutional and recurrent models for AD classification.
'''


import tensorflow as tf
from tensorflow.keras import layers, regularizers, models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback, TensorBoard
# from keras_self_attention import SeqSelfAttention
# from attention_keras.layers.attention import AttentionLayer

import sys
# sys.path.insert(0, '../input/attention')
# from seq_self_attention import SeqSelfAttention

import os
import glob
import time

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import numpy as np
np.random.seed(42)

import cv2
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

import spectogram_augmentation
import dataset_features


dataset_dir = ''
dataset_dir = '../spectograms/'
cc_files = sorted(glob.glob(os.path.join(dataset_dir, 'cc-images/*.png')))
X_cc = np.array([dataset_features.get_spectogram_features(f) for f in cc_files])
y_cc = np.zeros((X_cc.shape[0], 2))
y_cc[:,0] = 1

cd_files = sorted(glob.glob(os.path.join(dataset_dir, 'cd-images/*.png')))
X_cd = np.array([dataset_features.get_spectogram_features(f) for f in cd_files])
y_cd = np.zeros((X_cd.shape[0], 2))
y_cd[:,1] = 1

X = np.concatenate((X_cc, X_cd), axis=0).astype(np.float32)
y = np.concatenate((y_cc, y_cd), axis=0).astype(np.float32)
filenames = np.concatenate((cc_files, cd_files), axis=0)

p = np.random.permutation(len(X))
X = X[p]
y = y[p]
filenames = filenames[p]

X_shape = X.shape

inp_shape = X_cc[0].shape
print('#####################')
print(inp_shape) # (480, 640, 3)
print('#####################')
num_classes = 2

class DataGenerator():

	def __init__(self, X_spectrograms, y, batch_size, split='train'):

		self.X_spectrograms = X_spectrograms
		# self.X_interventions = X_interventions
		self.y = y
		self.batch_size = batch_size
		self.n_samples = self.X_spectrograms.shape[0]
		if split=='train':
			self.datagen = tf.keras.preprocessing.image.ImageDataGenerator(
				horizontal_flip=True)

		elif split=='val':
			self.datagen = tf.keras.preprocessing.image.ImageDataGenerator(
				horizontal_flip=True)

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

		return cropped_images


	def get_n_batches(self):
		return self.n_samples//self.batch_size

	# def run_on_batch(self, batch, aug, prob=0.0):
	# 	aug_images = []
	# 	for img in batch:
	# 		if np.random.random()<prob:
	# 			aug_images.append(aug(img[None,...]))
	# 		else:
	# 			aug_images.append(img)

	def flow(self):
		while True:
			p = np.random.permutation(len(self.X_spectrograms))
			self.X_spectrograms = self.X_spectrograms[p]
			# self.X_interventions = self.X_interventions[p]
			self.y = self.y[p]
			batch_n = 0

			for x_batch_spectograms, y_batch in self.datagen.flow(self.X_spectrograms, self.y, batch_size=self.batch_size):
				if batch_n>self.n_samples-self.batch_size:
					break
				# x_batch_interventions = self.X_interventions[batch_n:batch_n+batch_size]
				batch_n += 1
				original = x_batch_spectograms

				# if np.random.random()<0.4:
				# 	x_batch_spectograms = spectogram_augmentation.augment_pitch_and_tempo(x_batch_spectograms)
				# if np.random.random()<0.4:
				# 	x_batch_spectograms = spectogram_augmentation.augment_freq_time_mask(x_batch_spectograms)
				yield (x_batch_spectograms, y_batch)

	def on_epoch_end():
		print('Epoch Done!')


def create_model(_type_ = 'convolutional'):

	if _type_=='convolutional':

		model = tf.keras.Sequential()
		model.add(layers.Input(inp_shape))
		# model.add(layers.Conv2D(16, kernel_size=(3, 3), strides=(1, 1),
		# 				 activation='relu'))
		model.add(layers.Conv2D(16, kernel_size=(3, 3), strides=(2, 2),
						 activation='relu'))
		model.add(layers.BatchNormalization())

		# model.add(layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
		# 				 activation='relu'))
		model.add(layers.Conv2D(32, kernel_size=(3, 3), strides=(2, 2),
						 activation='relu'))
		model.add(layers.BatchNormalization())

		# model.add(layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
		# 				 activation='relu'))
		model.add(layers.Conv2D(64, kernel_size=(3, 3), strides=(2, 2),
						 activation='relu'))
		model.add(layers.BatchNormalization())

		# model.add(layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1),
		# 				 activation='relu'))
		model.add(layers.Conv2D(128, kernel_size=(3, 3), strides=(2, 2),
						 activation='relu'))
		model.add(layers.BatchNormalization())

		# model.add(layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1),
		# 				 activation='relu'))
		model.add(layers.Conv2D(256, kernel_size=(3, 3), strides=(2, 2),
						 activation='relu'))
		model.add(layers.BatchNormalization())

		# model.add(layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 1),
		# 				 activation='relu'))
		# model.add(layers.Conv2D(512, kernel_size=(3, 3), strides=(2, 2),
		# 				 activation='relu'))
		# model.add(layers.BatchNormalization())

		# model.add(layers.Conv2D(512, kernel_size=(3, 3), strides=(2, 2),
		# 				 activation='relu'))
		# model.add(layers.BatchNormalization())

		# model.add(layers.Flatten())
		# model.add(layers.Dropout(0.5)) # 0.5
		model.add(layers.GlobalAveragePooling2D())
		model.add(layers.Dropout(0.5)) # 0.5
		model.add(layers.Dense(128, activation='relu'))
		model.add(layers.Dropout(0.5))
		model.add(layers.Dense(num_classes, activation='softmax'))

		# model.add(layers.Conv2D(num_classes, kernel_size=(1,1)))
		# model.add(layers.Activation('softmax'))

	if _type_=='convolutional_1':

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
		model2_hidden11 = layers.Dropout(0.2)(model2_hidden11)

		output = layers.Dense(num_classes, activation='softmax')(model2_hidden11)

		return tf.keras.models.Model(inputs=model2_input,outputs=output)

	if _type_=='recurrent':

		global X
		X = X.reshape(X_shape[0], X_shape[2], X_shape[1], X_shape[3])

		print(X.shape, y.shape) # (108, 640, 480, 3) (108, 2)

		# if _binning_=='True':

		
		print((X_cc[0].shape[1], X_cc[0].shape[0], X_cc[0].shape[2]))
		model = tf.keras.Sequential()
		model.add(layers.Input((X_cc[0].shape[1], X_cc[0].shape[0], X_cc[0].shape[2])))
		# model.add(layers.BatchNormalization())
		model.add(layers.TimeDistributed(layers.Flatten()))
		# model.add(layers.BatchNormalization())
		model.add(layers.Bidirectional(layers.GRU(16, kernel_regularizer=regularizers.l2(0.1),
				activity_regularizer=regularizers.l1(0.1), dropout=0.2)))
		model.add(layers.BatchNormalization())
		model.add(layers.Dropout(0.2))
		model.add(layers.Dense(num_classes, kernel_regularizer=regularizers.l2(0.1),
				activity_regularizer=regularizers.l1(0.1), activation='softmax'))

	if _type_=='old':
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
		model2_output = layers.Dense(num_classes, activation='softmax')(model2_output)

		model = models.Model(model2_input, model2_output)

	# if _type_=='self-attention':

	# 	print((X_cc[0].shape[1], X_cc[0].shape[0], X_cc[0].shape[2]))
	# 	model = tf.keras.Sequential()
	# 	model.add(layers.Input((X_cc[0].shape[1], X_cc[0].shape[0], X_cc[0].shape[2])))
	# 	model.add(layers.TimeDistributed(layers.Flatten()))
	# 	model.add(layers.LSTM(64, return_sequences=True))

	# 	model.add(SeqSelfAttention(attention_activation='sigmoid'))
	# 	model.add(layers.Flatten())

	# if _type_=='attention'

	# 	global X
	# 	X = X.reshape(X_shape[0], X_shape[2], X_shape[1], X_shape[3])
	# 	print(X.shape, y.shape)

	# 	encoder_inputs = layers.Input(shape=(X_cc[0].shape[1], X_cc[0].shape[0], X_cc[0].shape[2]))
	# 	encoder_time_distributed = layers.TimeDistributed(layers.Flatten()) 
	#     encoder = layers.Bidirectional(layers.LSTM(16, return_state=True, return_sequences=True, dropout=0.2))
	#     encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(encoder_time_distributed(encoder_inputs))

	#     state_h = layers.Concatenate()([forward_h, backward_h])
	#     state_c = layers.Concatenate()([forward_c, backward_c])
	#     encoder_states = [state_h, state_c]

	#     decoder_inputs = layers.Input(shape=(None, num_classes))
	#     decoder_dense = layers.Dense(num_classes, kernel_regularizer=regularizers.l2(0.1),
	# 			activity_regularizer=regularizers.l1(0.1), activation='softmax')
	#     decoder_outputs = decoder_dense(decoder_inputs, initial_state=encoder_states)

	# 	attn_layer = AttentionLayer(name='attention_layer')
	# 	attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])
	# 	model = models.Model

	return model


model = create_model(_type_='old')

print(model.summary())
print(X.shape, y.shape) # (108, 480, 640, 3)     (108, 2)


################################# CROSS VALIDATED MODEL TRAINING ################################

n_split = 5

epochs = 50
batch_size = 8

val_accuracies = []
train_accuracies = []
fold = 0

for train_index, val_index in KFold(n_split, shuffle=True).split(X):

	x_train, x_val = X[train_index], X[val_index]
	y_train, y_val = y[train_index], y[val_index]
	filenames_train, filenames_val = filenames[train_index], filenames[val_index]

	model = create_model(_type_='old')

	timeString = time.strftime("%Y%m%d-%H%M%S", time.localtime())
	log_name = "{}".format(timeString)

	# tensorboard = TensorBoard(log_dir="logs/{}".format(log_name), histogram_freq=1, write_graph=True, write_images=False)

	model.compile(loss=tf.keras.losses.categorical_crossentropy,
				  optimizer=tf.keras.optimizers.Adam(lr=0.001, epsilon=0.1),
				  metrics=['categorical_accuracy'])

	datagen = DataGenerator(x_train, y_train, batch_size)

	checkpointer = tf.keras.callbacks.ModelCheckpoint(
            'best_model_spec_2_{}.h5'.format(fold), monitor='val_loss', verbose=0, save_best_only=False,
            save_weights_only=False, mode='auto', save_freq='epoch'
        )
	model.fit(datagen.flow(),
			  epochs=epochs,
			  steps_per_epoch=datagen.get_n_batches(),
			  verbose=1,
			  callbacks=[checkpointer],
			  validation_data=(x_val, y_val))
	model = tf.keras.models.load_model('best_model_spec_2_{}.h5'.format(fold))

	print('_'*30)

	val_pred = model.predict(x_val)
	for i in range(len(x_val)):
		print(filenames_val[i], np.argmax(val_pred[i])==np.argmax(y_val[i]), val_pred[i])

	train_score = model.evaluate(x_train, y_train, verbose=0)
	# print('_'*30)
	# print(model.predict(x_train))
	# print(model.predict(x_val))

	train_accuracies.append(train_score[1])
	score = model.evaluate(x_val, y_val, verbose=0)
	print('Val accuracy:', score[1])
	val_accuracies.append(score[1])

	print('Train accuracies ', train_accuracies)
	print('Train mean', np.mean(train_accuracies))
	print('Train std', np.std(train_accuracies))

	print('Val accuracies ', val_accuracies)
	print('Val mean', np.mean(val_accuracies))
	print('Val std', np.std(val_accuracies))

	fold+=1

