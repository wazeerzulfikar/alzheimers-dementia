import os, math
import glob

import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
np.random.seed(0)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="6"

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
					else:	
						features.append(feature_types[m][train_index])

				train_score, learnt_voter = get_ensemble_score(models, features, y_train, voting_type)

				features = []
				for m in model_types:
					if m == 'compare':
						features.append(compare_val)
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
	preds = list(map(lambda i:i[0], model.predict(feature)))
	print(len([i for i in preds if i>=26]))
	# print(model.predict(feature))
	return math.sqrt(model.evaluate(feature, y, verbose=0))

def get_ensemble_score(models, features, y, voting_type, num_classes=2, learnt_voter=None):
	preds = []
	for model, feature in zip(models, features):
		pred = model.predict(feature)
		preds.append(pred)
	preds = np.stack(preds, axis=1) # 86,3,1

	if voting_type=='soft_voting':
		voted_predictions = np.mean(preds, axis=1)
	elif voting_type=='learnt_voting':
		model_predictions = np.reshape(preds, (len(y), -1)) # 86,3
		if learnt_voter is None:
			learnt_voter = LogisticRegression().fit(model_predictions, np.expand_dims(y, axis=-1))
		# print('Voter coef ', voter.coef_)
		voted_predictions = learnt_voter.predict(model_predictions)

	score = mean_squared_error(np.expand_dims(y, axis=-1), voted_predictions, squared=False)
	return score, learnt_voter




# def infer(dataset_dir, model_dir):
# 	print('Loading data...')

# 	data = dataset.prepare_data(dataset_dir)
# 	intervention_features, pause_features, compare_features = data[4:7]
# 	y, filenames = data[8], data[9]

# 	print('Loading models from {}'.format(model_dir))
# 	pause_model_files = sorted(glob.glob(os.path.join(model_dir, 'pause/reg/*.h5')))
# 	pause_models = list(map(lambda x: tf.keras.models.load_model(x), pause_model_files))

# 	# spec_model_files = sorted(glob.glob(os.path.join(model_dir,'spectogram/reg/*.h5')))
# 	# spec_models = list(map(lambda x: tf.keras.models.load_model(x), spec_model_files))

# 	compare_model_files = sorted(glob.glob(os.path.join(model_dir,'compare/reg/*.h5')))
# 	compare_models = list(map(lambda x: tf.keras.models.load_model(x), compare_model_files))

# 	intervention_model_files = sorted(glob.glob(os.path.join(model_dir,'intervention/reg/*.h5')))
# 	intervention_models = list(map(lambda x: tf.keras.models.load_model(x), intervention_model_files))

# 	fold = 0
# 	n_split = 5

# 	# dataset_split = 'no_split'
# 	dataset_split = 'k_fold'

# 	voting_type = 'hard_voting'
# 	# voting_type = 'soft_voting'
# 	# voting_type = 'learnt_voting'

# 	print('Voting type {}'.format(voting_type))

# 	train_scores = []
# 	val_scores = []

# 	if dataset_split == 'no_split': # compare features need to be projected
# 		pause_probs = pause_models[0].predict(pause_features)
# 		# spec_probs = spec_models[0].predict(spectogram_features)
# 		inv_probs = intervention_models[0].predict(intervention_features)
# 		compare_probs = compare_models[0].predict(compare_features) ########## compare_features should be projected

# 		if voting_type=='hard_voting':
# 				model_predictions = [[np.argmax(pause_probs[i]), np.argmax(inv_probs[i]),  np.argmax(compare_probs[i])] for i in range(len(y_train))]
# 				voted_predictions = [max(set(i), key = i.count) for i in model_predictions]

# 		elif voting_type=='soft_voting':
# 			# model_predictions = pause_probs + spec_probs + inv_probs + compare_probs
# 			model_predictions = pause_probs + inv_probs + compare_probs
# 			# model_predictions = pause_probs + inv_probs
# 			voted_predictions = np.argmax(model_predictions, axis=-1)

# 		elif voting_type=='learnt_voting':
# 			# model_predictions = np.concatenate((pause_probs, spec_probs, inv_probs, compare_probs), axis=-1)
# 			model_predictions = np.concatenate((pause_probs, inv_probs, compare_probs), axis=-1)
# 			voter = LogisticRegression(C=0.1).fit(model_predictions, np.argmax(y_train, axis=-1))
# 			# print('Voter coef ', voter.coef_)
# 			voted_predictions = voter.predict(model_predictions)

# 			for i in range(len(model_predictions)):
# 				if voted_predictions[i]!= np.argmax(y, axis=-1)[i]:
# 					print(filenames[i])

# 		train_accuracy = accuracy_score(np.argmax(y, axis=-1), voted_predictions)
# 		train_accuracies.append(train_accuracy)
# 		print('Fold {} Training Accuracy {:.3f}'.format(fold, train_accuracy))

# 	elif dataset_split == 'k_fold':
		
# 		for train_index, val_index in KFold(n_split).split(pause_features):
# 			pause_train, pause_val = pause_features[train_index], pause_features[val_index]
# 			spec_train, spec_val = spectogram_features[train_index], spectogram_features[val_index]
# 			inv_train, inv_val = intervention_features[train_index], intervention_features[val_index]
# 			compare_train, compare_val = compare_features[train_index], compare_features[val_index]

# 			sc = StandardScaler()
# 			sc.fit(compare_train)

# 			compare_train = sc.transform(compare_train)
# 			compare_val = sc.transform(compare_val)

# 			pca = PCA(n_components=compare_feature_size)
# 			pca.fit(compare_train)

# 			compare_train = pca.transform(compare_train)
# 			compare_val = pca.transform(compare_val)

# 			y_train, y_val = y[train_index], y[val_index]

# 			filenames_train, filenames_val = filenames[train_index], filenames[val_index]

# 			pause_preds = pause_models[fold].predict(pause_train)
# 			# spec_preds = spec_models[fold].predict(spec_train)
# 			inv_preds = intervention_models[fold].predict(inv_train)
# 			compare_preds = compare_models[fold].predict(compare_train)

# 			if voting_type=='soft_voting':
# 				voted_predictions = (pause_preds + inv_preds + compare_preds)/3
# 				# voted_predictions = (pause_preds + inv_preds)/2

# 			elif voting_type=='learnt_voting':
# 				model_predictions = np.concatenate((pause_preds, inv_preds, compare_preds), axis=-1)
# 				voter = LogisticRegression().fit(model_predictions, np.argmax(y_train, axis=-1))
# 				# print('Voter coef ', voter.coef_)
# 				voted_predictions = voter.predict(model_predictions)

# 			train_score = mean_squared_error(y_train, voted_predictions, squared=False)
# 			train_scores.append(train_score)
# 			print('Fold {} Training Accuracy {:.3f}'.format(fold+1, train_score))

# 			pause_preds = pause_models[fold].predict(pause_val)
# 			# spec_preds = spec_models[fold].predict(spec_val)
# 			inv_preds = intervention_models[fold].predict(inv_val)
# 			compare_preds = compare_models[fold].predict(compare_val)

# 			if voting_type=='soft_voting':
# 				voted_predictions = (pause_preds + inv_preds + compare_preds)/3
# 				# voted_predictions = (pause_preds + inv_preds)/2

# 			elif voting_type=='learnt_voting':
# 				model_predictions = np.concatenate((pause_preds, inv_preds, compare_preds), axis=-1)
# 				voted_predictions = voter.predict(model_predictions)
			
# 			val_score = mean_squared_error(y_val, voted_predictions, squared=False)
# 			val_scores.append(val_score)
# 			print('Fold {} Val Accuracy {:.3f}'.format(fold+1, val_score))
# 			fold+=1


# 	print('Train score mean: {:.3f}'.format(np.mean(train_scores)))
# 	print('Train score std: {:.3f}'.format(np.std(train_scores)))
# 	if len(val_scores) > 0:
# 		print('Val score mean: {:.3f}'.format(np.mean(val_scores)))
# 		print('Val score std: {:.3f}'.format(np.std(val_scores)))

