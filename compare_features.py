
import glob
import os
import math
import numpy as np
import csv
import time
np.random.seed(0)
n_ = 108
p = np.random.permutation(n_)

import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import KFold
from tensorflow.keras.models import Model
from sklearn.feature_selection import f_classif, chi2, SelectKBest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import dataset_features, dataset_utils

# def select_bestK_features(X, y, n):
#     fs = SelectKBest(score_func=f_classif, k=n)
#     fs.fit(X, np.argmax(y, axis=1))
#     X_selected = fs.transform(X)
#     return X_selected, fs

# def pca_features(X, n):
#     sc = StandardScaler()
#     X_scaled = sc.fit_transform(X)
#     pca = PCA(n_components=n)
#     X_selected = pca.fit_transform(X_scaled)
#     return X_selected

# def lda_features(X, y, n):
#     sc = StandardScaler()
#     X_scaled = sc.fit_transform(X)
#     lda = LDA(n_components=n)
#     X_selected = lda.fit_transform(X_scaled, np.argmax(y, axis=1))
#     return X_selected

# def plot_selected_features_scores(X, y, n='all'):
#     _, fs = select_bestK_features(X, y, n)
#     pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
#     pyplot.show()

def prepare_data(features_size):
    dataset_dir = '../ADReSS-IS2020-data/train/'

    cc_files = sorted(glob.glob(os.path.join(dataset_dir, 'compare_features/compare_cc/*.csv')))
    all_speakers_cc = []
    for filename in cc_files:
        all_speakers_cc.append(dataset_features.get_compare_features(filename))

    cd_files = sorted(glob.glob(os.path.join(dataset_dir, 'compare_features/compare_cd/*.csv')))
    all_speakers_cd = []
    for filename in cd_files:
        all_speakers_cd.append(dataset_features.get_compare_features(filename))

    ### Classification X and y values
    y_cc = np.zeros((len(all_speakers_cc), 2))
    y_cc[:,0] = 1

    y_cd = np.zeros((len(all_speakers_cd), 2))
    y_cd[:,1] = 1

    ### Regression y values
    y_reg_cc = dataset_utils.get_regression_values(os.path.join(dataset_dir, 'cc_meta_data.txt'))
    y_reg_cd = dataset_utils.get_regression_values(os.path.join(dataset_dir, 'cd_meta_data.txt'))


    ### X and y
    X = np.concatenate((all_speakers_cc, all_speakers_cd), axis=0).astype(np.float32)
    y = np.concatenate((y_cc, y_cd), axis=0).astype(np.float32)
    # X, _ = select_bestK_features(X, y, features_size)
    # X = pca_features(X, y, features_size)

    # X = lda_features(X, y, features_size)

    X_reg = np.copy(X)
    y_reg = np.concatenate((y_reg_cc, y_reg_cd), axis=0).astype(np.float32)

    filenames = np.concatenate((cc_files, cd_files), axis=0)

    X, X_reg = X[p], X_reg[p]
    y, y_reg = y[p], y_reg[p]
    filenames = filenames[p]

    return X, y, X_reg, y_reg, filenames

def create_model(features_size):
    # INP = layers.Input(shape=(features_size,))
    # # BN1 = layers.BatchNormalization()(INP)

    # x = layers.Dense(8, activation='relu')(INP)
    # x = layers.BatchNormalization()(x)
    # x = layers.Dropout(0.2)(x)

    # x = layers.Dense(8, activation='relu')(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Dropout(0.2)(x)

    # # x = layers.Dense(8, activation='relu')(x)
    # # x = layers.BatchNormalization()(x)
    # # x = layers.Dropout(0.2)(x)

    # # D3 = layers.Dense(16, activation='relu',  activity_regularizer=tf.keras.regularizers.l1(0.01))(DP2)
    # # BN4 = layers.BatchNormalization()(D3)
    # # DP3 = layers.Dropout(0.2)(BN4)

    # D4 = layers.Dense(2, activation='softmax')(x)

    # model = Model(INP, D4)
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(features_size,)))
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(2, activation='softmax'))

    return model


def training(loocv=False):

    epochs = 600
    batch_size = 8
    

    val_accuracies = []
    train_accuracies = []

    X, y, _, _, filenames = prepare_data(features_size)

    if loocv==True:
        n_split = X.shape[0]
        model_dir = 'loocv-models-compare'
    else:
        n_split = 5
        model_dir = '5-fold-models-compare'

    fold = 0
    models = []

    for train_index, val_index in KFold(n_split).split(X):
        fold+=1

        x_train, x_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        filenames_train, filenames_val = filenames[train_index], filenames[val_index]

        sc = StandardScaler()
        sc.fit(x_train)

        x_train = sc.transform(x_train)
        x_val = sc.transform(x_val)

        pca = PCA(n_components=features_size)
        pca.fit(x_train)

        # e = pca.explained_variance_ratio_
        # for i in sorted(e)[::-1][:108]:
        #     print(i)
        # exit()

        x_train = pca.transform(x_train)
        x_val = pca.transform(x_val)

        model = create_model(features_size)

        model.compile(loss=tf.keras.losses.categorical_crossentropy,
                        optimizer=tf.keras.optimizers.Adam(lr=0.001),
                        metrics=['categorical_accuracy'])

        checkpointer = tf.keras.callbacks.ModelCheckpoint(
                os.path.join(model_dir, 'compare_{}.h5'.format(fold)), monitor='val_loss', verbose=0, save_best_only=True,
                save_weights_only=False, mode='auto', save_freq='epoch')

        model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    callbacks=[checkpointer],
                    validation_data=(x_val, y_val))

        model = tf.keras.models.load_model(os.path.join(model_dir, 'compare_{}.h5'.format(fold)))
        val_pred = model.predict(x_val)

        for i in range(len(x_val)):
            print(filenames_val[i], np.argmax(val_pred[i])==np.argmax(y_val[i]), val_pred[i])
        models.append(model)
        train_score = model.evaluate(x_train, y_train, verbose=0)

        train_accuracies.append(train_score[1])
        score = model.evaluate(x_val, y_val, verbose=0)
        print('Val accuracy:', score[1])
        val_accuracies.append(score[1])
        print('Val mean till fold {} is {}'.format(fold, np.mean(val_accuracies)))

    print('Train accuracies ', train_accuracies)
    print('Train mean', np.mean(train_accuracies))
    print('Train std', np.std(train_accuracies))

    print('Val accuracies ', val_accuracies)
    print('Val mean', np.mean(val_accuracies))
    print('Val std', np.std(val_accuracies))
    # exit()
    return models

def training_on_entire_dataset():

    epochs = 600
    batch_size = 8
    features_size = 6373
    # features_size = 1024
    X, y, _, _, filenames = prepare_data(features_size)

    model = create_model(features_size)
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                        optimizer=tf.keras.optimizers.Adam(lr=0.001),
                        metrics=['categorical_accuracy'])

    checkpointer = tf.keras.callbacks.ModelCheckpoint(
            'best_model_compare.h5', monitor='val_categorical_accuracy', verbose=0, save_best_only=True,
            save_weights_only=False, mode='max', save_freq='epoch')

    model.fit(X, y,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                callbacks=[checkpointer])

    model = tf.keras.models.load_model('best_model_compare.h5')
    train_loss, train_acc = model.evaluate(X, y, verbose=0)
    print('Train Loss: {}\t Train Accuracy: {}'.format(train_loss, train_acc))

def regression(loocv=False):

    if loocv==True:
        n_split = X_reg.shape[0]
        model_dir = 'loocv-models-compare'
    else:
        n_split = 5
        model_dir = '5-fold-models-compare'

    fold = 0
    epochs = 2000
    batch_size = 8

    train_scores, val_scores = [], []
    val_rounded_scores = []
    all_train_predictions, all_val_predictions = [], []
    all_train_true, all_val_true = [], []
    train_accuracies, val_accuracies = [], [] # classification

    X, y, X_reg, y_reg, filenames = prepare_data(features_size)

    for train_index, val_index in KFold(n_split).split(X_reg):

        fold+=1
        print('\n######################### FOLD {} #########################\n'.format(fold))
        x_reg_train, x_reg_val = X_reg[train_index], X_reg[val_index]
        y_reg_train, y_reg_val = y_reg[train_index], y_reg[val_index]

        sc = StandardScaler()
        sc.fit(x_reg_train)

        x_reg_train = sc.transform(x_reg_train)
        x_reg_val = sc.transform(x_reg_val)

        pca = PCA(n_components=features_size)
        pca.fit(x_reg_train)

        x_reg_train = pca.transform(x_reg_train)
        x_reg_val = pca.transform(x_reg_val)

        x_train, x_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        sc = StandardScaler()
        sc.fit(x_train)

        x_train = sc.transform(x_train)
        x_val = sc.transform(x_val)

        pca = PCA(n_components=features_size)
        pca.fit(x_train)

        x_train = pca.transform(x_train)
        x_val = pca.transform(x_val)        

        model = tf.keras.models.load_model(os.path.join(model_dir, 'compare_{}.h5'.format(fold)))
        train_accuracies.append(model.evaluate(x_train, y_train, verbose=0)[1])
        val_accuracies.append(model.evaluate(x_val, y_val, verbose=0)[1])
        model.pop()
        for layer in model.layers:
            layer.trainable = False

        model_reg = tf.keras.Sequential()
        model_reg.add(model)
        # model_reg.add(layers.Dense(16, activation='relu'))
        # model_reg.add(layers.Dense(8, activation='relu'))
        # model_reg.add(layers.Dense(8, activation='relu'))
        # model_reg.add(layers.Dense(8, activation='relu'))
        # model_reg.add(layers.BatchNormalization())
        # model_reg.add(layers.Dropout(0.5))
        model_reg.add(layers.Dense(1, activation='relu'))
 
        model_reg.compile(loss=tf.keras.losses.mean_squared_error,
            optimizer=tf.keras.optimizers.Adam(lr=0.001))

        checkpointer = tf.keras.callbacks.ModelCheckpoint(
                        os.path.join(model_dir, 'compare_reg_{}.h5'.format(fold)), monitor='val_loss', verbose=0, save_best_only=True,
                        save_weights_only=False, mode='auto', save_freq='epoch')
        timeString = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        log_name = "{}".format(timeString)
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(log_name), histogram_freq=1, write_graph=True, write_images=False)

        model_reg.fit(x_reg_train, y_reg_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            callbacks=[checkpointer, tensorboard],
                            validation_data=(x_reg_val, y_reg_val))

        model_reg = tf.keras.models.load_model(os.path.join(model_dir, 'compare_reg_{}.h5'.format(fold)))
        # models.append(model)
        train_score = math.sqrt(model_reg.evaluate(x_reg_train, y_reg_train, verbose=0))
        train_scores.append(train_score)

        val_score = math.sqrt(model_reg.evaluate(x_reg_val, y_reg_val, verbose=0))
        val_scores.append(val_score)

        train_predictions, val_predictions = model_reg.predict(x_reg_train), model_reg.predict(x_reg_val)
        all_train_predictions.append(train_predictions)
        all_val_predictions.append(val_predictions)
        all_train_true.append(y_reg_train)
        all_val_true.append(y_reg_val)
        val_rounded_scores.append(math.sqrt(np.mean(list(map(lambda i: (round(val_predictions[i,0]) - y_reg_val[i])**2, list(range(val_predictions.shape[0])))))))


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


features_size = 21
regression(loocv=False)

'''
Dense 8
Dense 1

Train Scores  [4.935066173546004, 6.373517498538949, 4.952092238025257, 5.214185435787439, 6.489117133710715]
Train mean 5.592795695921672
Train std 0.6927298791832278

Val Scores  [6.914622207659519, 7.644976603420425, 6.40931680356243, 6.7303712630780765, 10.279508959252839]
Val mean 7.595759167394658
Val std 1.401814864157051

Dense 8
Dense 8
Dense 8
Dense 1

Train Scores  [4.34013930171892, 6.317116381173984, 5.062877628897427, 5.088407730653666, 15.056717716515958]
Train mean 7.1730517517919905
Train std 3.9927129342241146

Val Scores  [5.258178879554644, 7.625046526657487, 6.090936099112891, 6.710476487548512, 16.098975183319123]
Val mean 8.356722635238532
Val std 3.9477262533789945


'''

# models = training(loocv=True)
# models = training()

# X, y, _, _, _ = prepare_data(2)
# print(X.shape)
# plot_selected_features_scores(X, y, 3000)
