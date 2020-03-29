'''
Converting .wav audio files for volume normalization. 2 methods explored below:
1) match all instances' dB to mean dB od the dataset
2) boost audio of every instance to respective instance max volume
'''

import glob
import os

import numpy as np
from pydub import AudioSegment, effects 

def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

volumes = []

dataset_dir = 'train/Full_wave_enhanced_audio/cc/'
files = sorted(glob.glob(os.path.join(dataset_dir, '*.wav')))
for file in files:
	sound = AudioSegment.from_file(file, "wav")
	volumes.append(sound.dBFS)

dataset_dir = 'train/Full_wave_enhanced_audio/cd/'
files = sorted(glob.glob(os.path.join(dataset_dir, '*.wav')))
for file in files:
	sound = AudioSegment.from_file(file, "wav")
	volumes.append(sound.dBFS)

print(np.min(volumes), np.mean(volumes), np.max(volumes))
print(len(volumes))

target_db = np.mean(volumes)	

export_dir = 'train/Audio_normalized_full_wave_enhanced_audio/cc/'
dataset_dir = 'train/Full_wave_enhanced_audio/cc/'
files = sorted(glob.glob(os.path.join(dataset_dir, '*.wav')))
for file in files:
	filename = file.strip(dataset_dir).strip('.wav')
	sound = AudioSegment.from_file(file, "wav")
	# normalized_sound = match_target_amplitude(sound, target_db) # match to mean (method1)
	normalized_sound = effects.normalize(sound) # match to highest volume i instance (method2)

	normalized_sound.export(os.path.join(export_dir, filename+'.wav'), format="wav")

export_dir = 'train/Audio_normalized_full_wave_enhanced_audio/cd/'
dataset_dir = 'train/Full_wave_enhanced_audio/cd/'
files = sorted(glob.glob(os.path.join(dataset_dir, '*.wav')))
for file in files:
	filename = file.strip(dataset_dir).strip('.wav')
	sound = AudioSegment.from_file(file, "wav")
	# normalized_sound = match_target_amplitude(sound, target_db) # match to mean
	normalized_sound = effects.normalize(sound)  

	normalized_sound.export(os.path.join(export_dir, filename+'.wav'), format="wav")

dataset_dir = 'train/Audio_normalized_full_wave_enhanced_audio/cc/'
files = sorted(glob.glob(os.path.join(dataset_dir, '*.wav')))
for file in files:
	sound = AudioSegment.from_file(file, "wav")
	print(file)
	print(sound.dBFS)
	print()

dataset_dir = 'train/Audio_normalized_full_wave_enhanced_audio/cd/'
files = sorted(glob.glob(os.path.join(dataset_dir, '*.wav')))
for file in files:
	sound = AudioSegment.from_file(file, "wav")
	print(file)
	print(sound.dBFS)
	print()

### S002 and S068 lowest sound in CC

dataset_dir = 'train/Full_wave_enhanced_audio/cc/'
file = sorted(glob.glob(os.path.join(dataset_dir, 'S002.wav')))[0]
filename = file.strip(dataset_dir).strip('.wav')
sound = AudioSegment.from_file(file, "wav")
print("Original dB: ", sound.dBFS)
normalized_sound = match_target_amplitude(sound, 0)
normalized_sound.export(os.path.join(dataset_dir, filename+'_n.wav'), format="wav")
normalized_sound = effects.normalize(sound)  
normalized_sound.export(os.path.join(dataset_dir, filename+'_n_2.wav'), format="wav")

file = sorted(glob.glob(os.path.join(dataset_dir, 'S002_n.wav')))[0]
sound = AudioSegment.from_file(file, "wav")
print("Now dB: ", sound.dBFS)
file = sorted(glob.glob(os.path.join(dataset_dir, 'S002_n_2.wav')))[0]
sound = AudioSegment.from_file(file, "wav")
print("Now dB: ", sound.dBFS)