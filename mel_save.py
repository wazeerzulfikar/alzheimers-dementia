'''
Generating the MEL spectrogram plots (640 x 480) for each full wave enhanced audio
'''

import glob
import os

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def mel_save(filename, save_path, save=False):
	'''
	filename - input .wav file
	save - if True, spectrogram is saved in save_path, otherwise returned from this function
	save_path is always required since while returning the image is saved, read and then deleted
	'''

	n_fft = 2048
	hop_length = 512
	n_mels = 128

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

	if save:
		fig.savefig(os.path.join(save_path, filename.strip(".wav").split('/')[-1]))
		print('Saved')
	else:
		fig.savefig(os.path.join(save_path, filename.strip(".wav").split('/')[-1]))
		print('Saved')
		img = mpimg.imread(os.path.join(save_path, filename.strip(".wav").split('/')[-1]+".png"))[:,:,:3].astype(np.float32)
		os.remove(os.path.join(save_path, filename.strip(".wav").split('/')[-1]+".png"))
		return img

# dataset_dir = '../ADReSS-IS2020-data/train/Full_wave_enhanced_audio'
# files = sorted(glob.glob(os.path.join(dataset_dir, 'cc/*.wav')))
# for filename in files:
# 	print(filename)
# 	# fig = mel_save(filename, '../ADReSS-IS2020-data/train/Full_wave_enhanced_audio/images/cc', True)
# 	fig = mel_save(filename, '../ADReSS-IS2020-data/train/Full_wave_enhanced_audio/images/cc', False)
# 	print(fig.shape)
# 	exit()