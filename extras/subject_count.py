import os, glob
import numpy as np

dataset_dir = '../../Documents/dementia/DementiaBank'
cc_files = sorted(glob.glob(os.path.join(dataset_dir, 'transcription/cc/*.cha')))
cc_files_mod = list(map(lambda x: x.split('.')[-2].split('/')[-1], cc_files))
cd_files = sorted(glob.glob(os.path.join(dataset_dir, 'transcription/cd/*.cha')))
cd_files_mod = list(map(lambda x: x.split('.')[-2].split('/')[-1], cd_files))
print('Total Control speech samples: ', len(cc_files))
print('Total Dementia speech samples: ', len(cd_files))

val, counts = np.unique(list(map(lambda x: x.split('-')[0], cc_files)), return_counts=True)
print('Total Control subjects: ', len(val))

val, counts = np.unique(list(map(lambda x: x.split('-')[0], cd_files)), return_counts=True)
print('Total Dementia subjects: ', len(val))


cc_files_audio = sorted(glob.glob(os.path.join(dataset_dir, 'Full_wave_enhanced_audio/cc/*.mp3')))
cd_files_audio = sorted(glob.glob(os.path.join(dataset_dir, 'Full_wave_enhanced_audio/cd/*.mp3')))

cc_files_audio_mod = list(map(lambda x: x.split('.')[-2].split('/')[-1], cc_files_audio))
cd_files_audio_mod = list(map(lambda x: x.split('.')[-2].split('/')[-1], cd_files_audio))

# print(cc_files_mod)
# print(cc_files_audio_mod)
print(np.array_equal(cc_files_mod, cc_files_audio_mod))
print(np.array_equal(cd_files_mod, cd_files_audio_mod))

control_type, control_subjects = [], []
for idx, transcription_filename in enumerate(cc_files):
	with open(transcription_filename, 'r') as f:
		# print(transcription_filename)
		content = f.readlines()
		type_ = content[5].split('|')[5]
		if type_ == 'Control':
			control_type.append(type_)
			control_subjects.append(cc_files_mod[idx].split('-')[0])
		else:
			os.remove(transcription_filename)
			os.remove(cc_files_audio[idx])
# exit()
print()
print('Control speech samples: ', len(control_type))
print('Control subjects: ', len(list(set(control_subjects))))

dementia_type, dementia_subjects = [], []
for idx, transcription_filename in enumerate(cd_files):
	with open(transcription_filename, 'r') as f:
		# print(transcription_filename)
		content = f.readlines()
		type_ = content[5].split('|')[5]
		# print(type_)
		if type_ == 'ProbableAD':
			dementia_type.append(type_)
			dementia_subjects.append(cd_files_mod[idx].split('-')[0])
		elif type_ == 'PossibleAD':
			dementia_type.append(type_)
			dementia_subjects.append(cd_files_mod[idx].split('-')[0])
		else:
			os.remove(transcription_filename)
			os.remove(cd_files_audio[idx])

print('Total AD speech samples: ', len(dementia_type))
print('Total AD subjects: ', len(list(set(dementia_subjects))))