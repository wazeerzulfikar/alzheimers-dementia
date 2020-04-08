import wave
import contextlib
import os
import glob
import numpy as np

def audio_length(files):
	'''
	Takes in a list of files and returns the corresponding list of audio lengths
	'''
	lengths = []
	for fname in files:
		with contextlib.closing(wave.open(fname,'r')) as f:
			frames = f.getnframes()
			rate = f.getframerate()
			duration = frames / float(rate) # returns duration in seconds
			lengths.append(duration)

	return lengths

dataset_dir = '../ADReSS-IS2020-data/train/Full_wave_enhanced_audio'
cc_files = sorted(glob.glob(os.path.join(dataset_dir, 'cc/*.wav')))
cd_files = sorted(glob.glob(os.path.join(dataset_dir, 'cd/*.wav')))

cc_lengths = audio_length(cc_files)
cd_lengths = audio_length(cd_files)

print('CC')
print(cc_lengths)
print(np.min(cc_lengths), np.mean(cc_lengths), np.max(cc_lengths))

print('\nCD')
print(cd_lengths)
print(np.min(cd_lengths), np.mean(cd_lengths), np.max(cd_lengths))


def get_substring(file):
	start = file.find('/S')+1
	return file[start : start+4]

dataset_dir = '../ADReSS-IS2020-data/train/Normalised_audio-chunks'
cc_files = sorted(glob.glob(os.path.join(dataset_dir, 'cc/*.wav')))
cd_files = sorted(glob.glob(os.path.join(dataset_dir, 'cd/*.wav')))

cc_filenames = list(map(lambda x: get_substring(x), cc_files))
cd_filenames = list(map(lambda x: get_substring(x), cd_files))

cc_uniques = sorted(list(set(cc_filenames)))
cd_uniques = sorted(list(set(cd_filenames)))

cc_threshold_lengths = []
for unique in cc_uniques:
	a = []
	for file, filename in zip(cc_files, cc_filenames):
		if filename==unique:
			a.append(file)
	cc_threshold_lengths.append(np.sum(audio_length(a)))

cd_threshold_lengths = []
for unique in cd_uniques:
	a = []
	for file, filename in zip(cd_files, cd_filenames):
		if filename==unique:
			a.append(file)
	cd_threshold_lengths.append(np.sum(audio_length(a)))

print('\nCC thresholded')
print(cc_threshold_lengths)
print(np.min(cc_threshold_lengths), np.mean(cc_threshold_lengths), np.max(cc_threshold_lengths))

print('\nCD thresholded')
print(cd_threshold_lengths)
print(np.min(cd_threshold_lengths), np.mean(cd_threshold_lengths), np.max(cd_threshold_lengths))

