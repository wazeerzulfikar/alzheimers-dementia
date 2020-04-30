'''
Filter human voice range from the audio files using 
1) butterworth bandpass
2) FT and IFT

Independent component analysis
'''

import glob
import os

import numpy as np
from pydub import AudioSegment, effects, scipy_effects

from scipy.signal import butter, lfilter

# def butter_bandpass(lowcut, highcut, fs, order=5):
# 	nyq = 0.5 * fs 
# 	low = lowcut / nyq
# 	high = highcut / nyq
# 	b, a = butter(order, [low, high], btype='band')
# 	return b, a 

# def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
# 	b, a = butter_bandpass(lowcut, highcut, fs, order=order)
# 	y = lfilter(b, a, data)
# 	return y


lowcut = 60
highcut = 300

wav_file = '../ADReSS-IS2020-data/train/Full_wave_enhanced_audio/cc/S003.wav'

audio = AudioSegment.from_file(wav_file, 'wav')
# print(audio.band_pass_filter)
# print(dir(effects))
filtered_audio = scipy_effects.band_pass_filter(audio, lowcut, highcut, order=5)  
filtered_audio.export('S003_filtered.wav', format="wav")
