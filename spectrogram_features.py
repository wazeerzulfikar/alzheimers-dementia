'''
Convolutional and recurrent models for AD classification.
'''


import tensorflow as tf
from tensorflow.keras import layers, regularizers
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
import cv2
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder


dataset_dir = ''
# dataset_dir = '../spectograms/'
cc_files = sorted(glob.glob(os.path.join(dataset_dir, 'cc-images/*.png')))
X_cc = np.array([cv2.resize(cv2.imread(f), (640,480))/255. for f in cc_files])
y_cc = np.zeros((X_cc.shape[0], 2))
y_cc[:,0] = 1

cd_files = sorted(glob.glob(os.path.join(dataset_dir, 'cd-images/*.png')))
X_cd = np.array([cv2.resize(cv2.imread(f), (640,480))/255. for f in cd_files])
y_cd = np.zeros((X_cd.shape[0], 2))
y_cd[:,1] = 1

X = np.concatenate((X_cc, X_cd), axis=0).astype(np.float32)
y = np.concatenate((y_cc, y_cd), axis=0).astype(np.float32)

np.random.seed(0)
p = np.random.permutation(len(X))
X = X[p]
y = y[p]
X_shape = X.shape

inp_shape = X_cc[0].shape
print('#####################')
print(inp_shape) # (480, 640, 3)
print('#####################')
num_classes = 2

################################ MODEL ################################

def create_model(_type_='convolutional', _binning_=False):

	if _type_=='convolutional':

		model = tf.keras.Sequential()
		model.add(layers.Input(inp_shape))
		model.add(layers.Conv2D(16, kernel_size=(3, 3), strides=(2, 2),
						 activation='relu'))
		model.add(layers.Conv2D(16, kernel_size=(3, 3), strides=(2, 2),
						 activation='relu'))
		model.add(layers.BatchNormalization())

		model.add(layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
						 activation='relu'))
		model.add(layers.Conv2D(32, kernel_size=(3, 3), strides=(2, 2),
						 activation='relu'))
		model.add(layers.BatchNormalization())

		model.add(layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
						 activation='relu'))
		model.add(layers.Conv2D(64, kernel_size=(3, 3), strides=(2, 2),
						 activation='relu'))
		model.add(layers.BatchNormalization())

		model.add(layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1),
						 activation='relu'))
		model.add(layers.Conv2D(128, kernel_size=(3, 3), strides=(2, 2),
						 activation='relu'))
		model.add(layers.BatchNormalization())

		model.add(layers.Flatten())
		model.add(layers.Dropout(0.5)) # 0.5
		model.add(layers.Dense(128, activation='relu'))
		model.add(layers.Dropout(0.2))
		model.add(layers.Dense(num_classes, activation='softmax'))

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

#######################################################################

model = create_model()
print(model.summary())
print(X.shape, y.shape) # (108, 480, 640, 3)     (108, 2)


################################# CROSS VALIDATED MODEL TRAINING ################################

n_split = 5
epochs = 50 # 25 # 50
batch_size = 8 # 8 # 16?

training_accuracies, validation_accuracies = [], []

for train_index, val_index in KFold(n_split, shuffle=True).split(X):

	x_train, x_val = X[train_index], X[val_index]
	y_train, y_val = y[train_index], y[val_index]
	model = create_model()

	timeString = time.strftime("%Y%m%d-%H%M%S", time.localtime())
	log_name = "{}".format(timeString)

	tensorboard = TensorBoard(log_dir="logs/{}".format(log_name), histogram_freq=1, write_graph=True, write_images=False)

	model.compile(loss=tf.keras.losses.categorical_crossentropy,
				  optimizer=tf.keras.optimizers.Adam(lr=0.001, epsilon=0.1),
				  metrics=['categorical_accuracy'])
	model.fit(x_train, y_train,
			  batch_size=batch_size,
			  epochs=epochs,
			  verbose=1,
			  callbacks=[tensorboard],
			  validation_data=(x_val, y_val))
	training_accuracy = model.evaluate(x_train, y_train, verbose=0)
	validation_accuracy = model.evaluate(x_val, y_val, verbose=0)
	training_accuracies.append(training_accuracy[1])
	validation_accuracies.append(validation_accuracy[1])

	print()
	print('Predictions : ', model.predict(x_val))
	print('True values : ', y_val)
	print('Val accuracy:', validation_accuracy[1])
	print('Training Accuracies till now : ', training_accuracies, '\tValidation Accuracies till now : ', validation_accuracies)
	print()

	# exit()

print(np.mean(validation_accuracies), '\t', np.std(validation_accuracies))

##################################################################################################