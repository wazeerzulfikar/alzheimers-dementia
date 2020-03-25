'''
Generating the MEL spectrogram plots (640 x 480) for each full wave enhanced audio
'''

import glob
import os

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
print('hello')

n_fft = 2048
hop_length = 512
n_mels = 128

dataset_dir = 'train/Full_wave_enhanced_audio/cc/'
files = sorted(glob.glob(os.path.join(dataset_dir, '*.wav')))

for filename in files:
	print(filename)
	plt.clf()
	fig = plt.figure(frameon=False)

	ax = plt.Axes(fig, [0., 0., 1., 1.])
	ax.set_axis_off()
	fig.add_axes(ax)

	# ax.imshow(your_image, aspect='auto')
	# fig.savefig(fname, dpi)
	y, sr = librosa.load(filename)
	# trim silent edges
	whale_song, _ = librosa.effects.trim(y)
	librosa.display.waveplot(whale_song, sr=sr)

	S = librosa.feature.melspectrogram(whale_song, sr=sr, n_fft=n_fft, 
	                                   hop_length=hop_length, 
	                                   n_mels=n_mels)
	S_DB = librosa.power_to_db(S, ref=np.max)
	librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, 
	                         x_axis='time', y_axis='mel')
	# plt.colorbar(format='%+2.0f dB')
	plt.title(filename)

	fig.savefig(os.path.join(dataset_dir+'images/', filename.strip(".wav").strip("train/Full_wave_enhanced_audio/cd/")))
	print('saved')
	# plt.show()