'''
Counting the number of Investigator dialogues and using a hard threshold to classify AD.
The thresholds are looped over 1 to 9 to find best one.
'''

import glob
import os
import numpy as np
np.random.seed(42)

import time
import re
import math

import audio_length

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

from sklearn.model_selection import KFold
from sklearn import preprocessing

def clean_file(lines):
  return re.sub(r'[0-9]+[_][0-9]+', '', lines.replace("*INV:", "").replace("*PAR:", "")).strip().replace("\x15", "").replace("\n", "").replace("\t", " ").replace("[+ ", "[+").replace("[* ", "[*").replace("[: ", "[:").replace(" .", "").replace("'s", "").replace(" ?", "").replace(" !", "").replace(" ]", "]").lower()

def extra_clean(lines):
  lines = clean_file(lines)
  lines = lines.replace("[+exc]", "")
  lines = lines.replace("[+gram]", "")
  lines = lines.replace("[+es]", "")
  lines = re.sub(r'[&][=]*[a-z]+', '', lines) #remove all &=text
  lines = re.sub(r'\[[*][a-z]:[a-z][-|a-z]*\]', '', lines) #remove all [*char:char(s)]
  lines = re.sub(r'[^A-Za-z0-9\s_]+', '', lines) #remove all remaining symbols except underscore
  lines = re.sub(r'[_]', ' ', lines) #replace underscore with space
  return lines

def words_count(content):
  extra_cleaned = extra_clean(content).split(" ")
  return len(extra_cleaned) - extra_cleaned.count('')

def get_pauses_cnt(file):
  cnt = 0
  pauses_list = []
  pauses = re.findall(r'&[a-z]+', file) #find all utterances
  cnt += len(pauses)
  pauses_list.append(len(pauses))

  pauses = re.findall(r'<[a-z_\s]+>', file) #find all <text>
  cnt += len(pauses)
  pauses_list.append(len(pauses))

  pauses = re.findall(r'\[/+\]', file) #find all [/+]
  cnt += len(pauses)
  pauses_list.append(len(pauses))

  pauses = re.findall(r'\([\.]+\)', file) #find all (.*)
  cnt += len(pauses)
  pauses_list.append(len(pauses))

  pauses = re.findall(r'\+[\.]+', file) #find all +...
  cnt += len(pauses)
  pauses_list.append(len(pauses))

  pauses = re.findall(r'[m]*hm', file) #find all mhm or hm
  cnt += len(pauses)
  pauses_list.append(len(pauses))

  pauses = re.findall(r'\[:[a-z_\s]+\]', file) #find all [:word]
  cnt += len(pauses)
  pauses_list.append(len(pauses))

  pauses = re.findall(r'[a-z]*\([a-z]+\)[a-z]*', file) #find all wor(d)
  cnt += len(pauses)
  pauses_list.append(len(pauses))

  temp = re.sub(r'\[[*][a-z]:[a-z][-|a-z]*\]', '', file)
  pauses = re.findall(r'[a-z]+:[a-z]+', temp) #find all w:ord
  cnt += len(pauses)
  pauses_list.append(len(pauses))

  # print(pauses_list)
  return pauses_list

def prepare_data():
  dataset_dir = '../ADReSS-IS2020-data/train/transcription/cc/'
  cc_files = sorted(glob.glob(os.path.join(dataset_dir, '*.cha')))
  all_pause_counts_cc = []
  all_inv_counts_cc = []
  all_word_counts_cc = []
  for filename in cc_files:
    inv_count = 0
    with open(filename, 'r') as f:
      content = f.read()
      words_counter = words_count(content)
      clean_content = clean_file(content)
      pause_count = get_pauses_cnt(clean_content) # list    
      content = content.split('\n')
      speaker_cc = []

      for c in content:
        if 'INV' in c:
          speaker_cc.append('INV')
        if 'PAR' in c:
          speaker_cc.append('PAR')
      
      PAR_first_index = speaker_cc.index('PAR')
      PAR_last_index = len(speaker_cc) - speaker_cc[::-1].index('PAR') - 1 
      speaker_cc = speaker_cc[PAR_first_index:PAR_last_index]
      inv_count = speaker_cc.count('INV') # number
    all_word_counts_cc.append([words_counter/50])
    all_inv_counts_cc.append([inv_count])
    all_pause_counts_cc.append(pause_count)
    # print('{} has {} INVs'.format(filename.split('/')[-1], inv_count))

  dataset_dir = '../ADReSS-IS2020-data/train/Full_wave_enhanced_audio/cc/'
  files = sorted(glob.glob(os.path.join(dataset_dir, '*.wav')))
  all_audio_lengths_cc = [[i/10] for i in audio_length.audio_length(files)]
  all_pause_rates_cc, all_inv_rates_cc, all_word_rates_cc = [], [], []
  for idx, pause_counts in enumerate(all_pause_counts_cc):
    pause_rates = []
    for p in pause_counts:
      pause_rates.append(p/all_audio_lengths_cc[idx][0])
    all_pause_rates_cc.append(pause_rates)
  for inv, audio in zip(all_inv_counts_cc, all_audio_lengths_cc):
    all_inv_rates_cc.append([inv[0]/audio[0]])
  for w, audio in zip(all_word_counts_cc, all_audio_lengths_cc):
    all_word_rates_cc.append([w[0]/audio[0]])

  print('-'*100)

  dataset_dir = '../ADReSS-IS2020-data/train/transcription/cd/'
  cd_files = sorted(glob.glob(os.path.join(dataset_dir, '*.cha')))
  all_pause_counts_cd = []
  all_inv_counts_cd = []
  all_word_counts_cd = []
  for filename in cd_files:
    inv_count = 0
    with open(filename, 'r') as f:
      content = f.read()
      words_counter = words_count(content)
      clean_content = clean_file(content)
      pause_count = get_pauses_cnt(clean_content)  
      content = content.split('\n')
      speaker_cd = []

      for c in content:
        if 'INV' in c:
          speaker_cd.append('INV')
        if 'PAR' in c:
          speaker_cd.append('PAR')
      
      PAR_first_index = speaker_cd.index('PAR')
      PAR_last_index = len(speaker_cd) - speaker_cd[::-1].index('PAR') - 1 
      speaker_cd = speaker_cd[PAR_first_index:PAR_last_index]
      inv_count = speaker_cd.count('INV')
    all_word_counts_cd.append([words_counter/50])
    all_inv_counts_cd.append([inv_count])
    all_pause_counts_cd.append(pause_count)
    # print('{} has {} INVs'.format(filename.split('/')[-1], inv_count))

  dataset_dir = '../ADReSS-IS2020-data/train/Full_wave_enhanced_audio/cd/'
  files = sorted(glob.glob(os.path.join(dataset_dir, '*.wav')))
  all_audio_lengths_cd = [[i/10] for i in audio_length.audio_length(files)]
  all_pause_rates_cd, all_inv_rates_cd, all_word_rates_cd = [], [], []
  for idx, pause_counts in enumerate(all_pause_counts_cd):
    pause_rates = []
    for p in pause_counts:
      pause_rates.append(p/all_audio_lengths_cd[idx][0])
    all_pause_rates_cd.append(pause_rates)
  for inv, audio in zip(all_inv_counts_cd, all_audio_lengths_cd):
    all_inv_rates_cd.append([inv[0]/audio[0]])
  for w, audio in zip(all_word_counts_cd, all_audio_lengths_cd):
    all_word_rates_cd.append([w[0]/audio[0]])

  print('-'*100)

  # all_counts_cc = np.concatenate((all_inv_counts_cc, all_pause_counts_cc), axis=-1)
  # all_counts_cd = np.concatenate((all_inv_counts_cd, all_pause_counts_cd), axis=-1)

  all_counts_cc = np.concatenate((all_inv_rates_cc, all_pause_rates_cc, all_word_rates_cc), axis=-1)
  all_counts_cd = np.concatenate((all_inv_rates_cd, all_pause_rates_cd, all_word_rates_cd), axis=-1)

  # all_counts_cc = preprocessing.normalize(all_counts_cc, axis=0)
  # all_counts_cd = preprocessing.normalize(all_counts_cd, axis=0)

  X = np.concatenate((all_counts_cc, all_counts_cd), axis=0).astype(np.float32)

  ### Regression y values
  y_reg_cc = np.zeros((len(all_counts_cc), ))
  file = open('../ADReSS-IS2020-data/train/cc_meta_data.txt', 'r+')
  lines = file.readlines()[1:]
  for idx, line in enumerate(lines):
    token = line.split('; ')[-1].strip('\n')
    if token!='NA':   y_reg_cc[idx] = int(token)
    else:   y_reg_cc[idx] = 30
    

  y_reg_cd = np.zeros((len(all_counts_cd), ))
  file = open('../ADReSS-IS2020-data/train/cd_meta_data.txt', 'r+')
  lines = file.readlines()[1:]
  for idx, line in enumerate(lines):
    token = line.split('; ')[-1].strip('\n')
    y_reg_cd[idx] = int(token)

  y_reg = np.concatenate((y_reg_cc, y_reg_cd), axis=0).astype(np.float32)
  #######################

  ### Regression X values
  X_reg = np.copy(X)
  #######################

  ### Classification y values
  y_cc = np.zeros((len(all_counts_cc), 2))
  y_cc[:,0] = 1

  y_cd = np.zeros((len(all_counts_cd), 2))
  y_cd[:,1] = 1

  y = np.concatenate((y_cc, y_cd), axis=0).astype(np.float32)
  filenames = np.concatenate((cc_files, cd_files), axis=0)
  #######################

  p = np.random.permutation(len(X))
  X, X_reg = X[p], X_reg[p]
  y, y_reg = y[p], y_reg[p]
  filenames = filenames[p]

  return X, y, X_reg, y_reg, filenames


def create_model():
  # model = tf.keras.Sequential()
  # model.add(layers.Input(shape=(10,)))
  # model.add(layers.Dense(16, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
  # model.add(layers.Dense(32, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
  # model.add(layers.Dense(16, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
  # model.add(layers.Dropout(0.2))
  # model.add(layers.Dense(2, activation='softmax'))
  model = tf.keras.Sequential()
  model.add(layers.Input(shape=(11,)))
  # model.add(layers.Dropout(0.2))
  model.add(layers.BatchNormalization())
  model.add(layers.Dense(16, activation='relu'))
  # model.add(layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
  # model.add(layers.Dense(64, activation='relu'))
  # model.add(layers.Dense(128, activation='relu'))
  # model.add(layers.Dense(256, activation='sigmoid'))
  # model.add(layers.Dense(128, activation='relu'))
  # model.add(layers.Dense(64, activation='relu'))
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

def regression_baseline():

	_, _, _, y_reg = prepare_data()
	values = []
	for i in range(50):
		value = math.sqrt(np.mean(list(map(lambda x: (x-i)**2, list(y_reg)))))
		values.append(value)
	print(values)
	print()
	print(np.argmin(values), np.min(values)) # 23 	7.18279838173066


def regression(models):
	'''
	models is a list of loaded models
	'''

	_, _, X_reg, y_reg = prepare_data()
	fold = 0
	n_split = 5
	epochs = 1500
	batch_size = 8

	train_scores, val_scores = [], []
	all_train_predictions, all_val_predictions = [], []
	all_train_true, all_val_true = [], []

	for train_index, val_index in KFold(n_split).split(X_reg):

		# def root_mean_squared_error(y_true, y_pred):
		# 	return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

		x_train, x_val = X_reg[train_index], X_reg[val_index]
		y_train, y_val = y_reg[train_index], y_reg[val_index]

		model = models[fold]
		model.pop()
		for layer in model.layers:
			layer.trainable = False

		model_reg = tf.keras.Sequential()
		model_reg.add(model)
		model_reg.add(layers.Dense(16, activation='relu'))
		model_reg.add(layers.Dense(8, activation='relu'))
		# model_reg.add(layers.BatchNormalization())
		# model_reg.add(layers.Dropout(0.5))
		model_reg.add(layers.Dense(1, activation='relu'))

		# print(model_reg.summary())
		model_reg.compile(loss=tf.keras.losses.mean_squared_error, 
			optimizer=tf.keras.optimizers.Adam(lr=0.001))

		checkpointer = tf.keras.callbacks.ModelCheckpoint(
						'best_model_reg_{}.h5'.format(fold), monitor='val_loss', verbose=0, save_best_only=True,
						save_weights_only=False, mode='auto', save_freq='epoch')

		model_reg.fit(x_train, y_train,
							batch_size=batch_size,
							epochs=epochs,
							verbose=1,
							callbacks=[checkpointer],
							validation_data=(x_val, y_val))

		model_reg = tf.keras.models.load_model('best_model_reg_{}.h5'.format(fold))
		# models.append(model)
		train_score = math.sqrt(model_reg.evaluate(x_train, y_train, verbose=0))
		train_scores.append(train_score)
		val_score = math.sqrt(model_reg.evaluate(x_val, y_val, verbose=0))
		val_scores.append(val_score)
		train_predictions = model_reg.predict(x_train)
		all_train_predictions.append(train_predictions)
		val_predictions = model_reg.predict(x_val)
		all_val_predictions.append(val_predictions)
		all_train_true.append(y_train)
		all_val_true.append(y_val)
		fold+=1

	print()
	print('################### TRAIN VALUES ###################')
	for f in range(n_split):
		print()
		print('################### FOLD {} ###################'.format(f))
		print('True Values \t Predicted Values')
		for i in range(all_train_true[f].shape[0]):
			print(all_train_true[f][i], '\t\t', all_train_predictions[f][i,0])

	print()
	print('################### VAL VALUES ###################')
	for f in range(n_split):
		print()
		print('################### FOLD {} ###################'.format(f))
		print('True Values \t Predicted Values')
		for i in range(all_val_true[f].shape[0]):
			print(all_val_true[f][i], '\t\t', all_val_predictions[f][i,0])

	print()
	print('Train Scores ', train_scores)
	print('Train mean', np.mean(train_scores))
	print('Train std', np.std(train_scores))
	
	print()
	print('Val accuracies ', val_scores)
	print('Val mean', np.mean(val_scores))
	print('Val std', np.std(val_scores))

	# print('Train True values')
	# print(all_train_true)
	# print()

	# print('Train predicted values')
	# print(all_train_predictions)
	# print()

	# print('Val True values')
	# print(all_val_true)
	# print()

	# print('Val predicted values')
	# print(all_val_predictions)
	# print()


def training():

  n_split = 5
  epochs = 600
  batch_size = 8

  val_accuracies = []
  train_accuracies = []

  X, y, _, _, filenames = prepare_data()
  fold = 0
  models = []

  for train_index, val_index in KFold(n_split).split(X):

    x_train, x_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    filenames_train, filenames_val = filenames[train_index], filenames[val_index]


    model = create_model()

    # timeString = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    # log_name = "{}".format(timeString)
    # tensorboard = TensorBoard(log_dir="logs/{}".format(log_name), histogram_freq=1, write_graph=True, write_images=False)

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  metrics=['categorical_accuracy'])

    checkpointer = tf.keras.callbacks.ModelCheckpoint(
            'best_model_{}.h5'.format(fold), monitor='val_loss', verbose=0, save_best_only=True,
            save_weights_only=False, mode='auto', save_freq='epoch')

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=0,
              callbacks=[checkpointer],
              validation_data=(x_val, y_val))

    model = tf.keras.models.load_model('best_model_{}.h5'.format(fold))
    val_pred = model.predict(x_val)

    for i in range(len(x_val)):
        print(filenames_val[i], np.argmax(val_pred[i])==np.argmax(y_val[i]), val_pred[i])
    models.append(model)
    train_score = model.evaluate(x_train, y_train, verbose=0)

    train_accuracies.append(train_score[1])
    score = model.evaluate(x_val, y_val, verbose=0)
    print('Val accuracy:', score[1])
    val_accuracies.append(score[1])
    fold+=1
    print('Train accuracies ', train_accuracies)
    print('Train mean', np.mean(train_accuracies))
    print('Train std', np.std(train_accuracies))

    print('Val accuracies ', val_accuracies)
    print('Val mean', np.mean(val_accuracies))
    print('Val std', np.std(val_accuracies))
    # exit()





  return models

# models = training()

# models = [tf.keras.models.load_model('best_model_{}.h5'.format(fold)) for fold in range(5)]
# print(models)
# regression(models)

regression_baseline()




# thresholds = []

# for threshold in thresholds:
#     cc_pred = 0
#     cd_pred = 0
#     for cc, cd in zip(all_inv_counts_cc, all_inv_counts_cd):
#         if cc > threshold:
#             cc_pred+=1
#         if cd > threshold:
#             cd_pred+=1
#     print('Threshold of {} Investigator dialogues'.format(threshold))
#     print('Diagnosed {} healthy people'.format(cc_pred))

#     print('Diagnosed {} AD people'.format(cd_pred))


#     precision = cd_pred / (cc_pred + cd_pred)
#     recall = cd_pred / ((54-cd_pred) + cd_pred)
#     accuracy = (54-cc_pred+cd_pred)/108
#     f1_score = (2*precision*recall)/(precision+recall)
#     print('Accuracy {:.3f} '.format(accuracy))
#     print('F1 score {:.3f}'.format(f1_score))
#     print('Precision {:.3f}'.format(precision))
#     print('Recall {:.3f}'.format(recall))

#     print('----'*50)
