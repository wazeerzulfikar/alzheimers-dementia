import os, math
import glob

import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
np.random.seed(0)

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="6"

# Local imports
import dataset

def evaluate(dataset_dir, model_dir, model_types, voting_type='soft_voting', dataset_split='k_fold', n_split = 5):
	print('Loading data...')

	data = dataset.prepare_data(dataset_dir)
	intervention_features, pause_features, compare_features = data[4:7]
	y_class, y, filenames = data[7], data[8], data[9]

	feature_types = {
		'intervention': intervention_features,
		'pause': pause_features,
		'compare': compare_features,
		'intervention-scratch': intervention_features,
		'pause-scratch': pause_features
	}

	compare_feature_size = 21

	print('Loading models from {}'.format(model_dir))
	saved_model_types = {}

	for m in model_types:
		model_files = sorted(glob.glob(os.path.join(model_dir, '{}/reg/*.h5'.format(m))))
		saved_models = list(map(lambda x: tf.keras.models.load_model(x), model_files))

		saved_model_types[m] = saved_models

	print()
	print('Loading models from {}'.format(model_dir))
	print('Using {} on {}'.format(voting_type, dataset_split))
	print('Models evaluated ', model_types)

	train_scores = []
	val_scores = []

	if dataset_split == 'k_fold':
		fold = 0

		for train_index, val_index in KFold(n_split).split(y):
			compare_train, compare_val = compare_features[train_index], compare_features[val_index]
			y_train, y_val = y[train_index], y[val_index]

			sc = StandardScaler()
			sc.fit(compare_train)

			compare_train = sc.transform(compare_train)
			compare_val = sc.transform(compare_val)

			pca = PCA(n_components=compare_feature_size)
			pca.fit(compare_train)

			compare_train = pca.transform(compare_train)
			compare_val = pca.transform(compare_val)


			if len(model_types) == 1:
				m = model_types[0]
				if m == 'compare':
					train_score = get_individual_score(saved_model_types[m][fold], compare_train, y_train)
					val_score = get_individual_score(saved_model_types[m][fold], compare_val, y_val)
				elif m == 'combined':
					train_score = get_individual_score(saved_model_types[m][fold], 
						(feature_types['intervention'][train_index], feature_types['pause'][train_index]), y_train)
					val_score = get_individual_score(saved_model_types[m][fold],
					 	(feature_types['intervention'][val_index], feature_types['pause'][val_index]), y_val)
				else:
					train_score = get_individual_score(saved_model_types[m][fold], feature_types[m][train_index], y_train)
					val_score = get_individual_score(saved_model_types[m][fold], feature_types[m][val_index], y_val)
			else:
				models = []
				features = []
				for m in model_types:
					models.append(saved_model_types[m][fold])
					if m == 'compare':
						features.append(compare_train)
					elif m == 'combined':
						features.append((feature_types['intervention'][train_index], feature_types['pause'][train_index]))
					else:	
						features.append(feature_types[m][train_index])

				train_score, learnt_voter = get_ensemble_score(models, features, y_train, voting_type)

				features = []
				for m in model_types:
					if m == 'compare':
						features.append(compare_val)
					elif m == 'combined':
						features.append((feature_types['intervention'][val_index], feature_types['pause'][val_index]))
					else:	
						features.append(feature_types[m][val_index])

				val_score, _ = get_ensemble_score(models, features, y_val, voting_type, learnt_voter=learnt_voter)

			train_scores.append(train_score)
			val_scores.append(val_score)

			print('Fold {} Training Score {:.3f}'.format(fold+1, train_score))
			print('Fold {} Val Score {:.3f}'.format(fold+1, val_score))
			fold+=1

		print('Train score mean: {:.3f}'.format(np.mean(train_scores)))
		print('Train score std: {:.3f}'.format(np.std(train_scores)))
		if len(val_scores) > 0:
			print('Val score mean: {:.3f}'.format(np.mean(val_scores)))
			print('Val score std: {:.3f}'.format(np.std(val_scores)))

def get_individual_score(model, feature, y):
	# preds = list(map(lambda i:i[0], model.predict(feature)))
	y = np.array(y)
	preds = model.predict(feature)

	print(len([i for i in preds if i>=26]))
	score = mean_squared_error(np.expand_dims(y, axis=-1), preds, squared=False)
	return score

def get_ensemble_score(models, features, y, voting_type, num_classes=2, learnt_voter=None):
	preds = []
	for model, feature in zip(models, features):
		probs = model.predict(feature)

		preds.append(probs)
	preds = np.stack(preds, axis=1) # 86,3,1

	if voting_type=='soft_voting':
		voted_predictions = np.mean(preds, axis=1)

	score = mean_squared_error(np.expand_dims(y, axis=-1), voted_predictions, squared=False)
	
	return score, learnt_voter

