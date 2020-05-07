import os
import glob

import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
np.random.seed(0)

# Local imports
import dataset

def test(test_filename, train_dataset_dir, test_dataset_dir, model_dir, model_types, voting_type='soft_voting', select_fold=None):
	'''
	Learnt voting not available
	'''
	print('Loading data...')

	train_compare_features = dataset.prepare_data(train_dataset_dir)[3]

	test_data = dataset.prepare_test_data(test_dataset_dir)
	intervention_features, pause_features, spectogram_features, compare_features, filenames = test_data
	feature_types = {
		'intervention': intervention_features,
		'pause': pause_features,
		'spectogram': spectogram_features,
		'compare': compare_features,
		'intervention-scratch': intervention_features,
		'pause-scratch': pause_features,
		'multi-head-intervention': intervention_features
	}

	compare_feature_size = 21

	print('Loading models from {}'.format(model_dir))
	saved_model_types = {}

	for m in model_types:
		model_files = sorted(glob.glob(os.path.join(model_dir, '{}/reg/*.h5'.format(m))))
		saved_models = list(map(lambda x: tf.keras.models.load_model(x), model_files))

		saved_model_types[m] = saved_models


	# Soft voting from all folds
	n_split = 5
	if select_fold == None:
		all_predictions = []
		fold = 0
		for train_index, val_index in KFold(n_split).split(train_compare_features):
			compare_train = train_compare_features[train_index]
			sc = StandardScaler()
			sc.fit(compare_train)
			compare_train = sc.transform(compare_train)
			compare_features_t = sc.transform(compare_features)
			pca = PCA(n_components=compare_feature_size)
			pca.fit(compare_train)
			compare_features_t = pca.transform(compare_features_t)

			models = []
			features = []
			for m in model_types:
				models.append(saved_model_types[m][fold])
				if m == 'compare':
					features.append(compare_features_t)
				else:	
					features.append(feature_types[m])

			fold_predictions = get_ensemble_predictions(models, features, voting_type)
			fold_predictions = np.squeeze(fold_predictions)
			all_predictions.append(fold_predictions)

			fold+=1

		all_predictions = np.stack(all_predictions, axis=1) # 86,5,1
		ensemble_predictions = np.mean(all_predictions, axis=1)

		# print(np.unique(ensemble_predictions, return_counts=1))
		print(list(ensemble_predictions))
		print(len([i for i in list(ensemble_predictions) if i>=26]))
	else:
		print('Using fold {} in {} through {}'.format(select_fold, model_dir, voting_type))
		print('Models used ', model_types)
		models = []
		features = []
		fold = 1
		for train_index, val_index in KFold(n_split).split(train_compare_features):
			compare_train = train_compare_features[train_index]
			if fold == select_fold:
				break
			fold+=1
		sc = StandardScaler()
		sc.fit(compare_train)
		compare_train = sc.transform(compare_train)
		compare_features_t = sc.transform(compare_features)
		pca = PCA(n_components=compare_feature_size)
		pca.fit(compare_train)
		compare_features_t = pca.transform(compare_features_t)
		
		for m in model_types:
			models.append(saved_model_types[m][select_fold - 1])
			if m == 'compare':
				features.append(compare_features_t)
			else:	
				features.append(feature_types[m])

		ensemble_predictions = get_ensemble_predictions(models, features, voting_type)
		ensemble_predictions = np.squeeze(ensemble_predictions)
		# print(np.unique(ensemble_predictions, return_counts=1))
		print(list(ensemble_predictions))
		print(len([i for i in list(ensemble_predictions) if i>=26]))

	np.save(test_filename.replace('.txt',''),ensemble_predictions)

	test_template = os.path.join(test_dataset_dir, 'test_results.txt')
	with open(test_template, 'r') as template, open(test_filename, 'w') as test_f:
		content = template.read().split('\n')
		# Write column names
		test_f.write(content[0]+'\n')
		for e,line in enumerate(content[1:-1]):
			new_line = line+' '+str(ensemble_predictions[e])
			test_f.write(new_line+'\n')

def get_individual_predictions(model, feature):
	return model.predict(feature, verbose=0)

def get_ensemble_predictions(models, features, voting_type='soft_voting'):
	preds = []
	for model, feature in zip(models, features):
		feature = np.array(feature)
		# print(feature.shape)
		# feature = np.expand_dims(feature, axis=0)
		# print(feature.shape)
		probs = model.predict(feature)
		# feature = np.array(feature)
		if len(probs) == 2:
			probs = probs[1]
		preds.append(probs)
	preds = np.stack(preds, axis=1)

	if voting_type=='soft_voting':
		voted_predictions = np.mean(preds, axis=1)


	return voted_predictions
	