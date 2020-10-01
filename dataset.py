import glob
import os
import math
import time
import re
import csv

import numpy as np
np.random.seed(0)
p = np.random.permutation(108) # n_samples = 108
p_subjects = np.random.RandomState(seed=0).permutation(242)
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

import utils

	
def get_pause_features(transcription_filename, audio_filename, audio_length_normalization=10):
    '''
    Pause features include word rate, pause rate of various kinds of pauses and utterances, and intervention rate
    '''
    audio_len =  utils.get_audio_length(audio_filename)/audio_length_normalization

    with open(transcription_filename, 'r') as f:
        content = f.read()
        word_rate = utils.words_count(content) / (50 * audio_len)
        pause_rates = utils.get_pauses_cnt(content) / audio_len
        inv_rate = utils.get_n_interventions(content) / audio_len

    pause_features = np.concatenate(([inv_rate], pause_rates, [word_rate]), axis=-1)

    return pause_features


def get_intervention_features(transcription_filename, max_length=40):
    '''
    Intervention features include one hot encoded sequence of speaker (par or inv) in the conversation
    '''
    speaker_dict = {
        'INV': [0 ,0 , 1],
        'PAR': [0, 1, 0],
        'padding': [1, 0, 0]
    }

    with open(transcription_filename, 'r') as f:
        content = f.read()
        content = content.split('\n')
        speakers = []

        for c in content:
            if 'INV' in c:
              speakers.append('INV')
            if 'PAR' in c:
              speakers.append('PAR')

        PAR_first_index = speakers.index('PAR')
        PAR_last_index = len(speakers) - speakers[::-1].index('PAR') - 1
        intervention_features = speakers[PAR_first_index:PAR_last_index]

    intervention_features = list(map(lambda x: speaker_dict[x], intervention_features))

    if len(intervention_features) > max_length:
        intervention_features = intervention_features[:max_length]
    else:
        pad_length = max_length - len(intervention_features)
        intervention_features =intervention_features+[speaker_dict['padding']]*pad_length

    return intervention_features


def get_spectogram_features(spectogram_filename):
    '''
    Spectogram features include MFCC which has been pregenerated for the audio file
    '''
    mel = np.load(spectogram_filename)
    # mel = feature_normalize(mel)
    mel = np.expand_dims(mel, axis=-1)
    return mel


def get_compare_features(compare_filename):
    compare_features = []
    with open(compare_filename, 'r') as file:
        content = csv.reader(file)
        for row in content:
            compare_features = row
    compare_features_floats = [float(item) for item in compare_features[1:-1]]
    return compare_features_floats


def prepare_data(dataset_dir, config):
	'''
	Prepare all data
	'''
	################################## SUBJECTS ################################

	subject_files = sorted(glob.glob(os.path.join(dataset_dir, 'transcription/*/*.cha')))
	subjects = np.array(sorted(list(set([re.split('[/-]', i)[-2] for i in subject_files]))))


	######################################################################


	################################## INTERVENTION ####################################


	cc_files = sorted(glob.glob(os.path.join(dataset_dir, 'transcription/cc/*.cha')))
	all_speakers_cc = []
	for filename in cc_files:
		all_speakers_cc.append(get_intervention_features(filename, config.longest_speaker_length))
		# print(get_intervention_features(filename, longest_speaker_length))
	
	
	cd_files = sorted(glob.glob(os.path.join(dataset_dir, 'transcription/cd/*.cha')))
	all_speakers_cd = []
	for filename in cd_files:
		all_speakers_cd.append(get_intervention_features(filename, config.longest_speaker_length))
		# print(get_intervention_features(filename, longest_speaker_length))
	
	# b = [len(i) for i in all_speakers_cc]
	# print(len(b), np.min(b), np.mean(b), np.max(b))
	# a = [len(i) for i in all_speakers_cd]
	# print(len(a), np.min(a), np.mean(a), np.max(a), len([i for i in a if i>np.max(b)]))
	# exit()
	y_cc = np.zeros((len(all_speakers_cc), 2))
	y_cc[:,0] = 1

	y_cd = np.zeros((len(all_speakers_cd), 2))
	y_cd[:,1] = 1

	X_intervention = np.concatenate((all_speakers_cc, all_speakers_cd), axis=0).astype(np.float32)
	y_intervention = np.concatenate((y_cc, y_cd), axis=0).astype(np.float32)
	filenames_intervention = np.concatenate((cc_files, cd_files), axis=0)

	y_reg_cc = utils.get_regression_values(os.path.join(dataset_dir, 'cc_meta_data.txt'))
	y_reg_cd = utils.get_regression_values(os.path.join(dataset_dir, 'cd_meta_data.txt'))

	y_reg_intervention = np.concatenate((y_reg_cc, y_reg_cd), axis=0).astype(np.float32)
	X_reg_intervention = np.copy(X_intervention)
	################################## INTERVENTION ####################################

	################################## PAUSE ####################################

	cc_transcription_files = sorted(glob.glob(os.path.join(dataset_dir, 'transcription/cc/*.cha')))
	if dataset_dir == '../DementiaBank':
		cc_audio_files = sorted(glob.glob(os.path.join(dataset_dir, 'Full_wave_enhanced_audio/cc/*.mp3')))
	else:
		cc_audio_files = sorted(glob.glob(os.path.join(dataset_dir, 'Full_wave_enhanced_audio/cc/*.wav')))

	all_counts_cc = []
	for t_f, a_f in zip(cc_transcription_files, cc_audio_files):
		pause_features = get_pause_features(t_f, a_f)
		all_counts_cc.append(pause_features)

	cd_transcription_files = sorted(glob.glob(os.path.join(dataset_dir, 'transcription/cd/*.cha')))
	if dataset_dir == '../DementiaBank':
		cd_audio_files = sorted(glob.glob(os.path.join(dataset_dir, 'Full_wave_enhanced_audio/cd/*.mp3')))
	else:
		cd_audio_files = sorted(glob.glob(os.path.join(dataset_dir, 'Full_wave_enhanced_audio/cd/*.wav')))

	all_counts_cd = [] 
	for t_f, a_f in zip(cd_transcription_files, cd_audio_files):
		pause_features = get_pause_features(t_f, a_f)
		all_counts_cd.append(pause_features)

	X_pause = np.concatenate((all_counts_cc, all_counts_cd), axis=0).astype(np.float32)

	y_reg_cc = utils.get_regression_values(os.path.join(dataset_dir, 'cc_meta_data.txt'))
	y_reg_cd = utils.get_regression_values(os.path.join(dataset_dir, 'cd_meta_data.txt'))

	y_reg_pause = np.concatenate((y_reg_cc, y_reg_cd), axis=0).astype(np.float32)

	X_reg_pause = np.copy(X_pause)

	y_cc = np.zeros((len(all_counts_cc), 2))
	y_cc[:,0] = 1

	y_cd = np.zeros((len(all_counts_cd), 2))
	y_cd[:,1] = 1

	y_pause = np.concatenate((y_cc, y_cd), axis=0).astype(np.float32)
	filenames_pause = np.concatenate((cc_transcription_files, cd_transcription_files), axis=0)
	################################## PAUSE ####################################

	################################## COMPARE ####################################
	cc_files = sorted(glob.glob(os.path.join(dataset_dir, 'compare/cc/*.csv')))
	X_cc = np.array([get_compare_features(f) for f in cc_files])
	y_cc = np.zeros((X_cc.shape[0], 2))
	y_cc[:,0] = 1

	cd_files = sorted(glob.glob(os.path.join(dataset_dir, 'compare/cd/*.csv')))
	X_cd = np.array([get_compare_features(f) for f in cd_files])
	y_cd = np.zeros((X_cd.shape[0], 2))
	y_cd[:,1] = 1

	X_compare = np.concatenate((X_cc, X_cd), axis=0).astype(np.float32)
	y_compare = np.concatenate((y_cc, y_cd), axis=0).astype(np.float32)

	X_reg_compare = np.copy(X_compare)

	y_reg_cc = utils.get_regression_values(os.path.join(dataset_dir, 'cc_meta_data.txt'))
	y_reg_cd = utils.get_regression_values(os.path.join(dataset_dir, 'cd_meta_data.txt'))

	y_reg_compare = np.concatenate((y_reg_cc, y_reg_cd), axis=0).astype(np.float32)

	filenames_compare = np.concatenate((cc_files, cd_files), axis=0)
	################################## COMPARE ####################################

	assert np.array_equal(y_intervention, y_pause) and X_intervention.shape[0]==X_pause.shape[0], '~ Data streams are different ~'
	print('~ Data streams verified ~')

	y = y_intervention
	y_reg = y_reg_intervention

	X_intervention = X_intervention[p]
	X_pause = X_pause[p]
	# X_spec = X_spec[p] 
	X_compare = X_compare[p]
	y = y[p]
	y_reg = y_reg[p]

	return {
		'intervention': X_intervention,
		'pause': X_pause,
		'compare': X_compare,
		'y_clf': y,
		'y_reg': y_reg,
		'subjects': subjects
	}


def prepare_test_data(dataset_dir, config):

	'''
	Prepare test data
	'''
	################################## SUBJECTS ################################

	subject_files = sorted(glob.glob(os.path.join(dataset_dir, 'transcription/*.cha')))
	subjects = np.array(sorted(list(set([re.split('[/-]', i)[-2] for i in subject_files]))))

	######################################################################

	################################## INTERVENTION ####################################

	transcription_files = sorted(glob.glob(os.path.join(dataset_dir, 'transcription/*.cha')))
	all_speakers = []
	for filename in transcription_files:
		all_speakers.append(get_intervention_features(filename, config.longest_speaker_length))	
	X_intervention = np.array(all_speakers)


	################################## PAUSE ####################################

	audio_files = sorted(glob.glob(os.path.join(dataset_dir, 'Full_wave_enhanced_audio/*.wav')))

	all_counts = []
	for t_f, a_f in zip(transcription_files, audio_files):
		pause_features = get_pause_features(t_f, a_f)
		all_counts.append(pause_features)
	X_pause = np.array(all_counts)


	################################## COMPARE ####################################
	compare_files = sorted(glob.glob(os.path.join(dataset_dir, 'compare/*.csv')))
	X_compare = np.array([get_compare_features(f) for f in compare_files])

	y = utils.get_classification_values(os.path.join(dataset_dir, 'meta_data.txt'))
	y_reg = utils.get_regression_values(os.path.join(dataset_dir, 'meta_data.txt'))


	assert X_intervention.shape[0]==X_pause.shape[0], '~ Data streams are different ~'
	print('~ Data streams verified ~')

	return {
		'intervention': X_intervention,
		'pause': X_pause,
		'compare': X_compare,
		'y_clf': y,
		'y_reg': y_reg,
		'subjects': subjects
	}
