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

def evaluate(dataset_dir, model_dir, model_types, voting_type='hard_voting', dataset_split='full_dataset', n_split = 5):
	print('Loading data...')

	data = dataset.prepare_data(dataset_dir)
	intervention_features, pause_features, spectogram_features, compare_features = data[0:4]
	y, filenames = data[7], data[9]

	feature_types = {
		'intervention': intervention_features,
		'pause': pause_features,
		'spectogram': spectogram_features,
		'compare': compare_features
	}

	compare_feature_size = 21

	print('Loading models from {}'.format(model_dir))
	saved_model_types = {}

	for m in model_types:
		model_files = sorted(glob.glob(os.path.join(model_dir, '{}/*.h5'.format(m))))
		saved_models = list(map(lambda x: tf.keras.models.load_model(x), model_files))

		saved_model_types[m] = saved_models

	print('Using {} on {}'.format(voting_type, dataset_split))
	print('Models evaluated ', model_types)

	train_accuracies = []
	val_accuracies = []

	if dataset_split == 'full_dataset': # compare features need to be projected
		if len(model_types) == 1:
			m = model_types[0]
			accuracy = get_individual_accuracy(saved_model_types[m][0], feature_types[m], y)
		else:
			models = []
			features = []
			for m in model_types:
				models.append(saved_model_types[m][0])
				features.append(feature_types[m])

			accuracy, learnt_voter = get_ensemble_accuracy(models, features, y, voting_type)

		print('Full dataset Accuracy {:.3f}'.format(accuracy))

	elif dataset_split == 'k_fold':
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
				train_accuracy = get_individual_accuracy(saved_model_types[m][fold], feature_types[m][train_index], y_train)
				val_accuracy = get_individual_accuracy(saved_model_types[m][fold], feature_types[m][val_index], y_val)
			else:

				models = []
				features = []
				for m in model_types:
					models.append(saved_model_types[m][fold])
					if m == 'compare':
						features.append(compare_train)
					else:	
						features.append(feature_types[m][train_index])

				train_accuracy, learnt_voter = get_ensemble_accuracy(models, features, y_train, voting_type)

				features = []
				for m in model_types:
					if m == 'compare':
						features.append(compare_val)
					else:	
						features.append(feature_types[m][val_index])

				val_accuracy, _ = get_ensemble_accuracy(models, features, y_val, voting_type, learnt_voter=learnt_voter)

			train_accuracies.append(train_accuracy)
			val_accuracies.append(val_accuracy)

			print('Fold {} Training Accuracy {:.3f}'.format(fold+1, train_accuracy))
			print('Fold {} Val Accuracy {:.3f}'.format(fold+1, val_accuracy))
			fold+=1

		print('Train accuracy mean: {:.3f}'.format(np.mean(train_accuracies)))
		print('Train accuracy std: {:.3f}'.format(np.std(train_accuracies)))
		if len(val_accuracies) > 0:
			print('Val accuracy mean: {:.3f}'.format(np.mean(val_accuracies)))
			print('Val accuracy std: {:.3f}'.format(np.std(val_accuracies)))


def get_individual_accuracy(model, feature, y):
	return model.evaluate(feature, y, verbose=0)[1]

def get_ensemble_accuracy(models, features, y, voting_type, num_classes=2, learnt_voter=None):
	probs = []
	for model, feature in zip(models, features):
		pred = model.predict(feature)
		probs.append(pred)
	probs = np.stack(probs, axis=1)

	if voting_type=='hard_voting':
		model_predictions = np.argmax(probs, axis=-1)
		model_predictions = np.squeeze(model_predictions)
		voted_predictions = [max(set(i), key = list(i).count) for i in model_predictions]
	elif voting_type=='soft_voting':
		model_predictions = np.sum(probs, axis=1)
		voted_predictions = np.argmax(model_predictions, axis=-1)
	elif voting_type=='learnt_voting':
		model_predictions = np.reshape(probs, (len(y), -1))
		if learnt_voter is None:
			learnt_voter = LogisticRegression().fit(model_predictions, np.argmax(y, axis=-1))
		# print('Voter coef ', voter.coef_)
		voted_predictions = learnt_voter.predict(model_predictions)

	accuracy = accuracy_score(np.argmax(y, axis=-1), voted_predictions)
	return accuracy, learnt_voter

