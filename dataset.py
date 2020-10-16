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

import collections
import contextlib
import sys
import wave

import webrtcvad
import audioop

def read_wave(path):
	"""Reads a .wav file.
	Takes the path, and returns (PCM audio data, sample rate).
	"""
	with contextlib.closing(wave.open(path, 'rb')) as wf:
		num_channels = wf.getnchannels()
		assert num_channels == 1
		sample_width = wf.getsampwidth()
		assert sample_width == 2
		sample_rate = wf.getframerate()

		n_frames = wf.getnframes()
		data = wf.readframes(n_frames)

		converted = audioop.ratecv(data, sample_width, num_channels, sample_rate, 32000, None)
		return converted[0], 32000

class Frame(object):
	"""Represents a "frame" of audio data."""
	def __init__(self, bytes, timestamp, duration):
		self.bytes = bytes
		self.timestamp = timestamp
		self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
	"""Generates audio frames from PCM audio data.
	Takes the desired frame duration in milliseconds, the PCM data, and
	the sample rate.
	Yields Frames of the requested duration.
	"""
	n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
	offset = 0
	timestamp = 0.0
	duration = (float(n) / sample_rate) / 2.0
	while offset + n < len(audio):
		yield Frame(audio[offset:offset + n], timestamp, duration)
		timestamp += duration
		offset += n


def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
	"""Filters out non-voiced audio frames.
	Given a webrtcvad.Vad and a source of audio frames, yields only
	the voiced audio.
	Uses a padded, sliding window algorithm over the audio frames.
	When more than 90% of the frames in the window are voiced (as
	reported by the VAD), the collector triggers and begins yielding
	audio frames. Then the collector waits until 90% of the frames in
	the window are unvoiced to detrigger.
	The window is padded at the front and back to provide a small
	amount of silence or the beginnings/endings of speech around the
	voiced frames.
	Arguments:
	sample_rate - The audio sample rate, in Hz.
	frame_duration_ms - The frame duration in milliseconds.
	padding_duration_ms - The amount to pad the window, in milliseconds.
	vad - An instance of webrtcvad.Vad.
	frames - a source of audio frames (sequence or generator).
	Returns: A generator that yields PCM audio data.
	"""
	num_padding_frames = int(padding_duration_ms / frame_duration_ms)
	# We use a deque for our sliding window/ring buffer.
	ring_buffer = collections.deque(maxlen=num_padding_frames)
	# We have two states: TRIGGERED and NOTTRIGGERED. We start in the
	# NOTTRIGGERED state.
	triggered = False

	voiced_frames = []
	silenced_frames = []
	for frame in frames:
		is_speech = vad.is_speech(frame.bytes, sample_rate)
		if is_speech:
		    silenced_frames.append(5)
		else:
			silenced_frames.append(10)
    		        
	return silenced_frames


def get_pause_masks(file):
	frame_duration_ms = 30

	audio, sample_rate = read_wave(file)
	vad = webrtcvad.Vad()
	frames = frame_generator(frame_duration_ms, audio, sample_rate)
	frames = list(frames)
	segments = vad_collector(sample_rate, frame_duration_ms, 10, vad, frames)
	
	segments = np.asarray(segments)
	# segments = (segments - np.mean(segments))/np.std(segments) 
	# print(segments)
	return segments

def prepare_data_new(dataset_dir, config):
	'''
	Prepare all data
	'''
	################################## SUBJECTS ################################
	subject_files = sorted(glob.glob(os.path.join(dataset_dir, 'Full_wave_enhanced_audio/cc/*.wav')) + glob.glob(os.path.join(dataset_dir, 'Full_wave_enhanced_audio/cd/*.wav')))
	subjects = np.array(sorted(list(set([i.split('/')[-1][:-4] for i in subject_files]))))

	######################################################################

	################################## SILENCE MASK ####################################

	if 'ADReSS' in dataset_dir:
		cc_audio_files = sorted(glob.glob(os.path.join(dataset_dir, 'Full_wave_enhanced_audio/cc/*.wav')))
	else:
		cc_audio_files = sorted(glob.glob(os.path.join(dataset_dir, 'Full_wave_enhanced_audio/cc/*.mp3')))

	# can tweak this parameter for clipping the len of the mask
	max_len = 800

	all_counts_cc = []
	for a_f in cc_audio_files:
		pause_features = get_pause_masks(a_f)
		all_counts_cc.append(pause_features[:max_len])

	if dataset_dir == '../DementiaBank':
		cd_audio_files = sorted(glob.glob(os.path.join(dataset_dir, 'Full_wave_enhanced_audio/cd/*.mp3')))
	else:
		cd_audio_files = sorted(glob.glob(os.path.join(dataset_dir, 'Full_wave_enhanced_audio/cd/*.wav')))

	all_counts_cd = [] 
	for a_f in cd_audio_files:
		pause_features = get_pause_masks(a_f)
		all_counts_cd.append(pause_features[:max_len])

	all_counts_cd = np.asarray(all_counts_cd)
	all_counts_cc = np.asarray(all_counts_cc)
	# print("all_counts_cc : ", all_counts_cc.shape)
	# print("all_counts_cd : ", all_counts_cd.shape)

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
	
	print(X_pause.shape, y_pause.shape, y_reg_pause.shape)


	################################## SILENCE MASK ####################################
	return {
		# 'intervention': X_intervention,
		'silences': X_pause,
		# 'compare': X_compare,
		'y_clf': y_pause,
		'y_reg': y_reg_pause,
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
