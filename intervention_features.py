'''
Counting the number of Investigator dialogues and using a hard threshold to classify AD.
The thresholds are looped over 1 to 9 to find best one.
'''

import glob
import os
import numpy as np
np.random.seed(0)

import tensorflow as tf

from tensorflow.keras import layers
from sklearn.model_selection import KFold
import time

import dataset_features

longest_speaker_length = 32
dataset_dir = '../ADReSS-IS2020-data/train/transcription'

cc_files = sorted(glob.glob(os.path.join(dataset_dir, 'cc/*.cha')))
all_speakers_cc = []
for filename in cc_files:
    all_speakers_cc.append(dataset_features.get_intervention_features(filename, longest_speaker_length))

cd_files = sorted(glob.glob(os.path.join(dataset_dir, 'cd/*.cha')))
all_speakers_cd = []
for filename in cd_files:
    all_speakers_cd.append(dataset_features.get_intervention_features(filename, longest_speaker_length))


y_cc = np.zeros((len(all_speakers_cc), 2))
y_cc[:,0] = 1

y_cd = np.zeros((len(all_speakers_cd), 2))
y_cd[:,1] = 1

X = np.concatenate((all_speakers_cc, all_speakers_cd), axis=0).astype(np.float32)
y = np.concatenate((y_cc, y_cd), axis=0).astype(np.float32)
filenames = np.concatenate((cc_files, cd_files), axis=0)

p = np.random.permutation(len(X))
X = X[p]
y = y[p]
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


n_split = 5
epochs = 400
batch_size = 8
val_accuracies = []
train_accuracies = []
fold = 0

print(create_model(longest_speaker_length).summary())


for train_index, val_index in KFold(n_split).split(X):

    fold+=1

    x_train, x_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    filenames_train, filenames_val = filenames[train_index], filenames[val_index]

    model = create_model(longest_speaker_length)

    timeString = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    log_name = "{}".format(timeString)

    # print(model.predict(x_val))
    checkpointer = tf.keras.callbacks.ModelCheckpoint(
        'best_model_intervention_{}.h5'.format(fold), monitor='val_loss', verbose=0, save_best_only=False,
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
    model = tf.keras.models.load_model('best_model_intervention_{}.h5'.format(fold))

    val_pred = model.predict(x_val)

    for i in range(len(x_val)):
        print(filenames_val[i], np.argmax(val_pred[i])==np.argmax(y_val[i]), val_pred[i])

    train_score = model.evaluate(x_train, y_train, verbose=0)
    train_accuracies.append(train_score[1])

    val_score = model.evaluate(x_val, y_val, verbose=0)
    print('Val accuracy:', val_score[1])
    val_accuracies.append(val_score[1])

    print('Train accuracies ', train_accuracies)
    print('Train mean', np.mean(train_accuracies))
    print('Train std', np.std(train_accuracies))


    print('Val accuracies ', val_accuracies)
    print('Val mean', np.mean(val_accuracies))
    print('Val std', np.std(val_accuracies))


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

