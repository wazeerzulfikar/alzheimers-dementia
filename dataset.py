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

def prepare_data(dataset_dir):
	################################## INTERVENTION ####################################

	longest_speaker_length = 32

	cc_files = sorted(glob.glob(os.path.join(dataset_dir, 'transcription/cc/*.cha')))
	all_speakers_cc = []
	for filename in cc_files:
		all_speakers_cc.append(dataset_features.get_intervention_features(filename, longest_speaker_length))

	cd_files = sorted(glob.glob(os.path.join(dataset_dir, 'transcription/cd/*.cha')))
	all_speakers_cd = []
	for filename in cd_files:
		all_speakers_cd.append(dataset_features.get_intervention_features(filename, longest_speaker_length))

	y_cc = np.zeros((len(all_speakers_cc), 2))
	y_cc[:,0] = 1

	y_cd = np.zeros((len(all_speakers_cd), 2))
	y_cd[:,1] = 1

	X_intervention = np.concatenate((all_speakers_cc, all_speakers_cd), axis=0).astype(np.float32)
	y_intervention = np.concatenate((y_cc, y_cd), axis=0).astype(np.float32)
	filenames_intervention = np.concatenate((cc_files, cd_files), axis=0)

	y_reg_cc = dataset_utils.get_regression_values(os.path.join(dataset_dir, 'cc_meta_data.txt'))
	y_reg_cd = dataset_utils.get_regression_values(os.path.join(dataset_dir, 'cd_meta_data.txt'))

	y_reg_intervention = np.concatenate((y_reg_cc, y_reg_cd), axis=0).astype(np.float32)
	X_reg_intervention = np.copy(X_intervention)
	################################## INTERVENTION ####################################

	################################## PAUSE ####################################

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

	y_reg_cc = dataset_utils.get_regression_values(os.path.join(dataset_dir, 'cc_meta_data.txt'))
	y_reg_cd = dataset_utils.get_regression_values(os.path.join(dataset_dir, 'cd_meta_data.txt'))

	y_reg_pause = np.concatenate((y_reg_cc, y_reg_cd), axis=0).astype(np.float32)

	X_reg_pause = np.copy(X_pause)

	y_cc = np.zeros((len(all_counts_cc), 2))
	y_cc[:,0] = 1

	y_cd = np.zeros((len(all_counts_cd), 2))
	y_cd[:,1] = 1

	y_pause = np.concatenate((y_cc, y_cd), axis=0).astype(np.float32)
	filenames_pause = np.concatenate((cc_transcription_files, cd_transcription_files), axis=0)
	################################## PAUSE ####################################

	################################## SPECTROGRAM ####################################
	cc_files = sorted(glob.glob(os.path.join(dataset_dir, 'spectograms/cc/*.png')))
	spectogram_size = (480, 640)
	X_cc = np.array([dataset_features.get_old_spectogram_features(f) for f in cc_files])
	y_cc = np.zeros((X_cc.shape[0], 2))
	y_cc[:,0] = 1

	cd_files = sorted(glob.glob(os.path.join(dataset_dir, 'spectograms/cd/*.png')))
	X_cd = np.array([dataset_features.get_old_spectogram_features(f) for f in cd_files])
	y_cd = np.zeros((X_cd.shape[0], 2))
	y_cd[:,1] = 1

	X_spec = np.concatenate((X_cc, X_cd), axis=0).astype(np.float32)
	y_spec = np.concatenate((y_cc, y_cd), axis=0).astype(np.float32)
	filenames_spec = np.concatenate((cc_files, cd_files), axis=0)
	################################## SPECTROGRAM ####################################

	################################## COMPARE ####################################
	cc_files = sorted(glob.glob(os.path.join(dataset_dir, 'compare/cc/*.csv')))
	X_cc = np.array([dataset_features.get_compare_features(f) for f in cc_files])
	y_cc = np.zeros((X_cc.shape[0], 2))
	y_cc[:,0] = 1

	cd_files = sorted(glob.glob(os.path.join(dataset_dir, 'compare/cd/*.csv')))
	X_cd = np.array([dataset_features.get_compare_features(f) for f in cd_files])
	y_cd = np.zeros((X_cd.shape[0], 2))
	y_cd[:,1] = 1

	X_compare = np.concatenate((X_cc, X_cd), axis=0).astype(np.float32)
	y_compare = np.concatenate((y_cc, y_cd), axis=0).astype(np.float32)

	X_reg_compare = np.copy(X_compare)

	y_reg_cc = dataset_utils.get_regression_values(os.path.join(dataset_dir, 'cc_meta_data.txt'))
	y_reg_cd = dataset_utils.get_regression_values(os.path.join(dataset_dir, 'cd_meta_data.txt'))

	y_reg_compare = np.concatenate((y_reg_cc, y_reg_cd), axis=0).astype(np.float32)

	filenames_compare = np.concatenate((cc_files, cd_files), axis=0)
	################################## COMPARE ####################################

	assert np.array_equal(y_intervention, y_pause) and np.array_equal(y_pause, y_spec) and np.array_equal(y_reg_intervention, y_reg_pause) and np.array_equal(y_reg_intervention, y_reg_compare) and X_intervention.shape[0]==X_pause.shape[0] and X_intervention.shape[0]==X_spec.shape[0] and X_compare.shape[0]==X_spec.shape[0] and np.array_equal(filenames_intervention, filenames_pause), '~ Data streams are different ~'
	print('~ Data streams verified ~')
	y = y_intervention
	y_reg = y_reg_intervention
	X_length = X_intervention.shape[0] # 108

	X_intervention, X_pause, X_spec, X_compare = X_intervention[p], X_pause[p], X_spec[p], X_compare[p]
	X_reg_intervention, X_reg_pause, X_reg_compare = X_reg_intervention[p], X_reg_pause[p], X_reg_compare[p]
	y, y_reg = y[p], y_reg[p]
	filenames_intervention, filenames_pause, filenames_spec, filenames_compare = filenames_intervention[p], filenames_pause[p], filenames_spec[p], filenames_compare[p]

	return X_intervention, X_pause, X_spec, X_compare, X_reg_intervention, X_reg_pause, X_reg_compare, y, y_reg, filenames_intervention, filenames_pause, filenames_spec, filenames_compare
