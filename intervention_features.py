'''
Counting the number of Investigator dialogues and using a hard threshold to classify AD.
The thresholds are looped over 1 to 9 to find best one.
'''

import glob
import os
import math
import numpy as np
np.random.seed(0)
n_ = 108
p = np.random.permutation(n_)

import tensorflow as tf

from tensorflow.keras import layers
from sklearn.model_selection import KFold
import time

import dataset_features, dataset_utils

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

X = np.concatenate((all_speakers_cc, all_speakers_cd), axis=0).astype(np.float32)
y = np.concatenate((y_cc, y_cd), axis=0).astype(np.float32)
filenames = np.concatenate((cc_files, cd_files), axis=0)
################################

### Regression X and y values
y_reg_cc = dataset_utils.get_regression_values(os.path.join(dataset_dir, 'cc_meta_data.txt'))
y_reg_cd = dataset_utils.get_regression_values(os.path.join(dataset_dir, 'cd_meta_data.txt'))

y_reg = np.concatenate((y_reg_cc, y_reg_cd), axis=0).astype(np.float32)
X_reg = np.copy(X)
#################################

X, X_reg = X[p], X_reg[p]
y, y_reg = y[p], y_reg[p]
filenames = filenames[p]

def create_model(longest_speaker_length, num_classes=2):
    model = tf.keras.Sequential()
    # model.add(layers.Input((longest_speaker_length)))
    # model.add(layers.TimeDistributed(layers.Flatten()))
    # model.add(layers.LSTM(128))
    model.add(layers.LSTM(8, input_shape=(longest_speaker_length, 3)))
    model.add(layers.BatchNormalization())
    # model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

def training(loocv=False):

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
    fold = 0

    # print(create_model(longest_speaker_length).summary())


    for train_index, val_index in KFold(n_split).split(X):

        fold+=1

        x_train, x_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        filenames_train, filenames_val = filenames[train_index], filenames[val_index]

        model = create_model(longest_speaker_length)

        # print(model.predict(x_val))
        checkpointer = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(model_dir, 'intervention_{}.h5'.format(fold)), monitor='val_loss', verbose=0, save_best_only=False,
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
        model = tf.keras.models.load_model(os.path.join(model_dir, 'intervention_{}.h5'.format(fold)))

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

    print()
    print('Train accuracies ', train_accuracies)
    print('Train mean', np.mean(train_accuracies))
    print('Train std', np.std(train_accuracies))
    print()
    print('Train losses ', train_losses)
    print('Train mean', np.mean(train_losses))
    print('Train std', np.std(train_losses))
    print()
    print('Val accuracies ', val_accuracies)
    print('Val mean', np.mean(val_accuracies))
    print('Val std', np.std(val_accuracies))
    print()
    print('Val losses ', val_losses)
    print('Val mean', np.mean(val_losses))
    print('Val std', np.std(val_losses))

def training_on_entire_dataset(X, y, longest_speaker_length):

    epochs = 400
    batch_size = 8

    model = create_model(longest_speaker_length)
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                      optimizer=tf.keras.optimizers.Adam(lr=0.001),
                      metrics=['categorical_accuracy'])

    checkpointer = tf.keras.callbacks.ModelCheckpoint(
            'best_model_intervention.h5', monitor='loss', verbose=0, save_best_only=True,
            save_weights_only=False, mode='auto', save_freq='epoch')

    model.fit(X, y,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              callbacks=[checkpointer])

    model = tf.keras.models.load_model('best_model_intervention.h5')
    train_loss, train_acc = model.evaluate(X, y, verbose=0)
    print('Train Loss: {}\t Train Accuracy: {}'.format(train_loss, train_acc))

def evaluate_models():
    '''
    Evaluate the saved models - 'best_model_intervention_1.h5', ... , 'best_model_intervention_5.h5'
    '''

    n_split = 5
    fold = 0
    val_accuracies = []
    for train_index, val_index in KFold(n_split).split(X):
        fold+=1
        x_val, y_val = X[val_index], y[val_index]
        model = tf.keras.models.load_model('best_model_intervention_{}.h5'.format(fold))
        val_score = model.evaluate(x_val, y_val, verbose=0)
        val_accuracies.append(val_score[1])
    print('Val accuracies ', val_accuracies) # [0.6363636, 0.90909094, 0.77272725, 0.71428573, 0.8095238]
    print('Val mean', np.mean(val_accuracies)) # 0.768
    print('Val std', np.std(val_accuracies)) # 0.09

def regression(loocv=False):

    if loocv==True:
        n_split = X_reg.shape[0]
        model_dir = 'loocv-models-intervention'
    else:
        n_split = 5
        model_dir = '5-fold-models-intervention'

    fold = 0
    epochs = 2000
    batch_size = 8

    train_scores, val_scores = [], []
    val_rounded_scores = []
    all_train_predictions, all_val_predictions = [], []
    all_train_true, all_val_true = [], []
    train_accuracies, val_accuracies = [], [] # classification

    for train_index, val_index in KFold(n_split).split(X_reg):

        fold+=1
        print('\n######################### FOLD {} #########################\n'.format(fold))
        x_train, x_val = X_reg[train_index], X_reg[val_index]
        y_train, y_val = y_reg[train_index], y_reg[val_index]

        model = tf.keras.models.load_model(os.path.join(model_dir, 'intervention_{}.h5'.format(fold)))
        train_accuracies.append(model.evaluate(X[train_index], y[train_index], verbose=0)[1])
        val_accuracies.append(model.evaluate(X[val_index], y[val_index], verbose=0)[1])
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
        # exit()
        model_reg.compile(loss=tf.keras.losses.mean_squared_error,
            optimizer=tf.keras.optimizers.Adam(lr=0.001))

        checkpointer = tf.keras.callbacks.ModelCheckpoint(
                        os.path.join(model_dir, 'intervention_reg_{}.h5'.format(fold)), monitor='val_loss', verbose=0, save_best_only=True,
                        save_weights_only=False, mode='auto', save_freq='epoch')
        timeString = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        log_name = "{}".format(timeString)
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(log_name), histogram_freq=1, write_graph=True, write_images=False)

        model_reg.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            callbacks=[checkpointer, tensorboard],
                            validation_data=(x_val, y_val))

        model_reg = tf.keras.models.load_model(os.path.join(model_dir, 'intervention_reg_{}.h5'.format(fold)))
        # models.append(model)
        train_score = math.sqrt(model_reg.evaluate(x_train, y_train, verbose=0))
        train_scores.append(train_score)

        val_score = math.sqrt(model_reg.evaluate(x_val, y_val, verbose=0))
        val_scores.append(val_score)

        train_predictions, val_predictions = model_reg.predict(x_train), model_reg.predict(x_val)
        all_train_predictions.append(train_predictions)
        all_val_predictions.append(val_predictions)
        all_train_true.append(y_train)
        all_val_true.append(y_val)
        val_rounded_scores.append(math.sqrt(np.mean(list(map(lambda i: (round(val_predictions[i,0]) - y_val[i])**2, list(range(val_predictions.shape[0])))))))


        print()
        print('Val score:', val_score)
        print('Val score mean till fold {} is {}'.format(fold, np.mean(val_scores)))
        print('Val rounded score mean till fold {} is {}'.format(fold, np.mean(val_rounded_scores)))
        print()

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
    print('Val Scores ', val_scores)
    print('Val mean', np.mean(val_scores))
    print('Val std', np.std(val_scores))

    print()
    print('Val rounded Scores ', val_rounded_scores)
    print('Val rounded mean', np.mean(val_rounded_scores))
    print('Val rounded std', np.std(val_rounded_scores))

    print()
    print('################# CLASSIFICATION #################')
    print('Train accuracies ', train_accuracies)
    print('Train mean', np.mean(train_accuracies))
    print('Train std', np.std(train_accuracies))

    print()
    print('Val accuracies ', val_accuracies)
    print('Val mean', np.mean(val_accuracies))
    print('Val std', np.std(val_accuracies))

regression(loocv=False)
# evaluate_models()
# training_on_entire_dataset(X, y, longest_speaker_length)
# training(loocv=False)

####################  Simple Thresholding ####################
# for threshold in range(10):
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
