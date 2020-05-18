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
import cv2

from PIL import Image
import pickle 
import glob 

def feature_normalize(feature_data):
    mean = np.mean(feature_data, axis=0)
    std = np.std(feature_data, axis=0)
    N = feature_data.shape[0]
    S1 = np.sum(feature_data, axis=0)
    S2 = np.sum(feature_data ** 2, axis=0)
    mean=S1/N
    std=np.sqrt((N * S2 - (S1 * S1)) / (N * (N - 1)))
    mean = np.reshape(mean, [1, -1])
    std = np.reshape(std, [1, -1])
    feature_data=((feature_data-mean)/std)
    return feature_data

def cqt_extract(filename):

    wav, sr = librosa.load(filename)
    cqt=librosa.core.cqt(y=wav, hop_length=512,sr=sr, n_bins=80, bins_per_octave=12, window='hamming')
    cqt=cqt.T
    cqt=feature_normalize(cqt)
    cqt=np.log10(cqt).astype('float32')
    cqt = np.expand_dims(cqt, axis=-1)
    save_path = filename.replace('Full_wave_enhanced_audio', 'cqt')
    save_path = save_path.replace('.wav','')
    np.save(save_path, cqt)

# data_files = glob.glob('../ADReSS-IS2020-data/train/Full_wave_enhanced_audio/cc/*.wav')
# for e,f in enumerate(data_files):
#     cqt_extract(f)
#     print(e)

# data_files = glob.glob('../ADReSS-IS2020-data/train/Full_wave_enhanced_audio/cd/*.wav')
# for e,f in enumerate(data_files):
#     cqt_extract(f)
#     print(e)


# cqt_extract('../ADReSS-IS2020-data/train/Full_wave_enhanced_audio/cc/S001.wav')
# cqt_extract('../ADReSS-IS2020-data/train/Full_wave_enhanced_audio/cc/S002.wav')
# cqt_extract('../ADReSS-IS2020-data/train/Full_wave_enhanced_audio/cd/S079.wav')

def feature_normalize(feature_data):
    mean = np.mean(feature_data, axis=0)
    std = np.std(feature_data, axis=0)
    N = feature_data.shape[0]
    S1 = np.sum(feature_data, axis=0)
    S2 = np.sum(feature_data ** 2, axis=0)
    mean=S1/N
    std=np.sqrt((N * S2 - (S1 * S1)) / (N * (N - 1)))
    mean = np.reshape(mean, [1, -1])
    std = np.reshape(std, [1, -1])
    feature_data=((feature_data-mean)/std)
    return feature_data

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
    # S_DB = librosa.power_to_db(S, ref=np.max)
    # librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, 
    #                          x_axis='time', y_axis='mel')
    # # plt.colorbar(format='%+2.0f dB')
    # plt.title(filename)
    # print(S_DB.shape)
    # print(np.mean(S), np.std(S))

    # print(np.mean(S_DB), np.std(S_DB))
    # S_DB_normalized = feature_normalize(S_DB)

    save_path = filename.replace('Full_wave_enhanced_audio', 'spectograms_np_e')
    save_path = save_path.replace('.wav','')
    np.save(save_path, S)

    return

    if save:
        fig.savefig(os.path.join(save_path, filename.strip(".wav").split('/')[-1]))
        print('Saved')
    else:
        fig.savefig(os.path.join(save_path, filename.strip(".wav").split('/')[-1]))
        print('Saved')
        img = mpimg.imread(os.path.join(save_path, filename.strip(".wav").split('/')[-1]+".png"))[:,:,:3].astype(np.float32)
        os.remove(os.path.join(save_path, filename.strip(".wav").split('/')[-1]+".png"))
        return img

data_files = glob.glob('../ADReSS-IS2020-data/train/Full_wave_enhanced_audio/cc/*.wav')
for e,f in enumerate(data_files):
    mel_save(f,'')
    print(e)
data_files = glob.glob('../ADReSS-IS2020-data/train/Full_wave_enhanced_audio/cd/*.wav')
for e,f in enumerate(data_files):
    mel_save(f,'')
    print(e)
# mel_save('../ADReSS-IS2020-data/train/Full_wave_enhanced_audio/cc/S001.wav','../ADReSS-IS2020-data/train/Full_wave_enhanced_audio/cc/S001.wav')
