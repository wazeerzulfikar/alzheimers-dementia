import glob
import os
import re
from sklearn.model_selection import KFold
import numpy as np
np.random.seed(0)

dataset_dir = '../DementiaBank/train'
cc_files = sorted(glob.glob(os.path.join(dataset_dir, 'transcription/cc/*.cha')))

cd_files = sorted(glob.glob(os.path.join(dataset_dir, 'transcription/cd/*.cha')))

all_files = np.array(cc_files + cd_files)
print(len(cc_files))
print(len(cd_files))
print(len(all_files))

subjects = np.array(sorted(list(set([re.split('[/-]', i)[-2] for i in all_files]))))
print('n subjects', len(subjects))

subject_n_features = {}
for s in subjects:
	count = 0
	for f in all_files:
		if s in f:
			count+=1

	subject_n_features[s] = count

n_folds = 10

split_reference = 'subjects'
# split_reference = 'samples'

print(f'Split by {split_reference}')

if split_reference == 'subjects':
	p = np.random.permutation(len(subjects))
	subjects = subjects[p]
	fold = 1
	for train_index, val_index in KFold(n_folds).split(subjects):
		train_subjects, val_subjects = subjects[train_index], subjects[val_index]
		train_sample_count = sum([subject_n_features[s] for s in train_subjects])
		val_sample_count = sum([subject_n_features[s] for s in val_subjects])

		print(f'Fold {fold}')
		print(f'Train count {train_sample_count}')
		print(f'Val count {val_sample_count}')
		print('--'*20)
		fold+=1

if split_reference == 'samples':
	p = np.random.permutation(len(all_files))
	all_files = all_files[p]

	fold = 1
	for train_index, val_index in KFold(n_folds).split(all_files):
		train_samples, val_samples = all_files[train_index], all_files[val_index]

		train_subjects = set([re.split('[/-]', i)[-2] for i in train_samples])
		val_subjects = set([re.split('[/-]', i)[-2] for i in val_samples])
		intersection = len([s for s in train_subjects if s in val_subjects])

		train_subjects_count = len(train_subjects)
		val_subjects_count = len(val_subjects)

		print(f'Fold {fold}')
		print(f'Train subjects {train_subjects_count}')
		print(f'Val subjects {val_subjects_count}')
		print(f'Intersection {intersection}')

		print('--'*20)
		fold+=1