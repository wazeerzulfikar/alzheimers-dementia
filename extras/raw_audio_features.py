import librosa
import glob
import os

import numpy as np

TARGET_SR = 1000

print('------- CC ------')


dataset_dir = '../ADReSS-IS2020-data/train/Full_wave_enhanced_audio/cc/'
files = sorted(glob.glob(os.path.join(dataset_dir, '*.wav')))
total_files = len(files)
cc_audios = []
for e, f in enumerate(files):
    print('{} / {} cc loaded'.format(e+1, total_files))
    audio, sr = librosa.load(f, sr = TARGET_SR, mono=True)
    print(len(audio))
    audio = np.reshape(audio, (-1, 1))
    audio = (audio - np.mean(audio)) / np.std(audio)
    cc_audios.append(audio)


dataset_dir = '../ADReSS-IS2020-data/train/Full_wave_enhanced_audio/cd/'
files = sorted(glob.glob(os.path.join(dataset_dir, '*.wav')))
total_files = len(files)
cd_audios = []
for f in files:
    print('{} / {} cd loaded'.format(e+1, total_files))
    audio, sr = librosa.load(f, sr = TARGET_SR, mono=True)
    print(sr)
    print(len(audio))

    audio = np.reshape(audio, (-1, 1))
    audio = (audio - np.mean(audio)) / np.std(audio)
    cd_audios.append(audio)


longest_speakers_cc = max(cc_audios, key=lambda x: len(x))
longest_speakers_cd = max(cd_audios, key=lambda x: len(x))

longest_speaker_length = max(len(longest_speakers_cc), len(longest_speakers_cd))
print(longest_speaker_length)