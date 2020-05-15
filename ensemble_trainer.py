import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


import dataset
import trainer

def bagging_ensemble_training(dataset_dir, model_dir, model_types, n_splits):

	data = dataset.prepare_data(dataset_dir)
	X_intervention, X_pause, X_spec, X_compare, y, y_reg, subjects = data

	feature_types = {
		'intervention': X_intervention,
		'pause': X_pause,
		'spectogram': X_spec,
		'compare': X_compare
	}

	results = {}

	## Train Intervention, models saved in `model_dir/intervention_{fold}.h5`
	for m in model_types:
		m_results = trainer.train_n_folds(m, subjects, feature_types[m], y, n_splits, model_dir, split_reference='samples')

	results[m] = m_results
	return results

def boosted_train_a_fold(
	booster_model_type, 
	X1,
	y,
	boosted_model_type,
	X2,
	fold, 
	model_dir,
	n_split=5
	):

	booster_model = tf.keras.models.load_model(os.path.join(model_dir, booster_model_type, 'fold_{}.h5'.format(fold)))

	# special stuff for compare features
	if booster_model_type == 'compare':
		compare_feature_size = 21
		fold_ = 0
		for train_index, val_index in KFold(n_split).split(X1):
			compare_train = X1[train_index]
			if fold_ == fold:
				break
			fold_+=1
		sc = StandardScaler()
		sc.fit(compare_train)
		X1 = sc.transform(X1)
		pca = PCA(n_components=compare_feature_size)
		pca.fit(compare_train)
		X1 = pca.transform(X1)

	booster_losses = []
	for idx, x in enumerate(X1):
		loss = booster_model.evaluate(np.expand_dims(x, axis=0), np.expand_dims(y[idx], axis=0), verbose=0)[0]
		booster_losses.append(loss)

	boost_probs = [float(i)/sum(booster_losses) for i in booster_losses] # normalizing losses into probs to sum to 1
	train_index = np.random.choice(len(y), 86, replace=False, p=boost_probs)
	val_index = np.array([i for i in np.arange(len(y)) if i not in train_index])

	x_train, x_val = X2[train_index], X2[val_index]
	y_train, y_val = y[train_index], y[val_index]

	results = trainer.train_a_fold(boosted_model_type, x_train, y_train, x_val, y_val, fold, model_dir)

	return results

def boosted_ensemble_training(dataset_dir, model_dir, model_types, n_splits=5):
	'''
	Training order is same as model_types
	'''
	
	data = dataset.prepare_data(dataset_dir)
	X_intervention, X_pause, X_spec, X_compare, y, y_reg, subjects = data

	feature_types = {
		'intervention': X_intervention,
		'pause': X_pause,
		'spectogram': X_spec,
		'compare': X_compare
	}

	results = {}

	# Train model type, models saved in `model_dir/{model_type}/fold_{fold}.h5`
	m_results = trainer.train_n_folds(model_types[0], feature_types[model_types[0]], y, n_splits, model_dir)
	results[model_types[0]] = m_results

	if len(model_types) > 1:
		for m_i in range(1, len(model_types)):
			m_results = []
			for fold in range(1, n_splits+1):
				booster_features = feature_types[model_types[m_i-1]]
				boosted_features = feature_types[model_types[m_i]]
				m_results_fold = boosted_train_a_fold(model_types[m_i-1], booster_features, y, 
					model_types[m_i], boosted_features, fold, model_dir)
				m_results.append(m_results_fold)

			results[model_types[m_i]] = m_results
	
	return results