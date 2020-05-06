import os, math
import numpy as np
import tensorflow as tf

import dataset
import trainer_reg

def bagging_ensemble_training(dataset_dir, model_dir, model_types, n_splits):

	data = dataset.prepare_data(dataset_dir)
	X_intervention, X_pause, X_spec, X_compare, X_reg_intervention, X_reg_pause, X_reg_compare = data[0:7]
	y, y_reg, filenames_intervention, filenames_pause, filenames_spec, filenames_compare = data[7:]

	feature_types = {
		'intervention': X_reg_intervention,
		'pause': X_reg_pause,
		'compare': X_reg_compare
	}

	results = {}

	for m in model_types:
		m_results = trainer_reg.train_n_folds(m, feature_types[m], y_reg, n_splits, model_dir)

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
	):

	booster_model = tf.keras.models.load_model(os.path.join(model_dir, booster_model_type, 'reg', 'fold_{}.h5'.format(fold)))

	booster_losses = []
	for idx, x in enumerate(X1):
		loss = math.sqrt(booster_model.evaluate(np.expand_dims(x, axis=0), np.expand_dims(y[idx], axis=0), verbose=0))
		booster_losses.append(loss)

	boost_probs = [float(i)/sum(booster_losses) for i in booster_losses] # normalizing losses into probs to sum to 1
	train_index = np.random.choice(len(y), 86, replace=False, p=boost_probs)
	val_index = np.array([i for i in np.arange(len(y)) if i not in train_index])

	x_train, x_val = X2[train_index], X2[val_index]
	y_train, y_val = y[train_index], y[val_index]

	results = trainer_reg.train_a_fold(boosted_model_type, x_train, y_train, x_val, y_val, fold, model_dir)

	return results

def boosted_ensemble_training(dataset_dir, model_dir, n_splits=5):
	'''
	Training order is same as model_types
	'''

	data = dataset.prepare_data(dataset_dir)
	X_intervention, X_pause, X_spec, X_compare, X_reg_intervention, X_reg_pause, X_reg_compare = data[0:7]
	y, y_reg, filenames_intervention, filenames_pause, filenames_spec, filenames_compare = data[7:]

	feature_types = {
		'intervention': X_reg_intervention,
		'pause': X_reg_pause,
		'compare': X_reg_compare
	}

	results = {}

	m_results = trainer.train_n_folds(model_types[0], feature_types[model_types[0]], y_reg, n_splits, model_dir)
	results[model_types[0]] = m_results
	if len(model_types) > 1:
		for m_i in range(1, len(model_types)):
			for fold in range(1, n_splits+1):
				booster_features = feature_types[model_types[m_i-1]]
				boosted_features = feature_types[model_types[m_i]]
				m_results_fold = boosted_train_a_fold(model_types[m_i-1], booster_features, y_reg, 
					model_types[m_i], boosted_features, fold, model_dir)
				m_results.append(m_results_fold)

			results[m] = m_results
	
	return results