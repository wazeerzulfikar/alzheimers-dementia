import tensorflow as tf
# import tensorflow.keras as k
import keras
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


import numpy as np
import cv2
from sklearn.model_selection import cross_val_score

import glob
import os


def read_and_process_image(list_of_images):
	"""
	Returns two arrays:
		X is an array of images
		y is an rray of labels
	"""
	X = [] # images
	y = [] # labels

	for image in list_of_images:
		img = cv2.imread(image, cv2.IMREAD_COLOR)
		X.append(img)

	return img.shape


dataset_dir = 'cc-images/'
cc = sorted(glob.glob(os.path.join(dataset_dir, '*.png')))

dataset_dir = 'cd-images/'
cd = sorted(glob.glob(os.path.join(dataset_dir, '*.png')))

input_shape = read_and_process_image(cc)
num_classes = 2


model = tf.keras.Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# training
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[history])

scores = cross_val_score(model, X, y, cv=5)
print('SCORES: ', scores)

# evaluating and printing results
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])