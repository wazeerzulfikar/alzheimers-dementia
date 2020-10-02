import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pickle import load

import dataset
import trainer

def bagging_ensemble_training(data, config):

	results = {}

	for model_type in config.model_types:
		model_results = trainer.train_n_folds(model_type, data, config)
		results[model_type] = model_results

	return results

def boosted_train_a_fold(
	data,
	booster_model_type, 
	boosted_model_type,
	fold,
	config
	):

	X1 = data[booster_model_type]
	X2 = data[boosted_model_type]

	if config.task == 'classification':
		y = data['y_clf']
	elif config.task == 'regression':
		y = data['y_reg']

	if config.uncertainty:
		def negloglik(y, p_y):
			return -p_y.log_prob(y)
		booster_model = tf.keras.models.load_model(os.path.join(config.model_dir, booster_model_type, 'fold_{}.h5'.format(fold)),
			custom_objects={'negloglik': negloglik})
	else:
		booster_model = tf.keras.models.load_model(os.path.join(config.model_dir, booster_model_type, 'fold_{}.h5'.format(fold)))

	# special stuff for compare features
	if booster_model_type == 'compare':
		sc = load(open(os.path.join(config.model_dir, 'compare/scaler_{}.pkl'.format(fold)), 'rb'))
		pca = load(open(os.path.join(config.model_dir, 'compare/pca_{}.pkl'.format(fold)), 'rb'))
		X1 = sc.transform(X1)
		X1 = pca.transform(X1)

	booster_losses = []
	for idx, x in enumerate(X1):
		if config.uncertainty:
			if config.boosting_type=='stddev':
				loss = booster_model(np.expand_dims(x, axis=0)).stddev().numpy()[0][0]
			elif config.boosting_type=='rmse':
				loss = np.abs(y[idx] - booster_model(np.expand_dims(x, axis=0)).mean().numpy()[0][0])
		else:
			loss = booster_model.evaluate(np.expand_dims(x, axis=0), np.expand_dims(y[idx], axis=0), verbose=0)
		booster_losses.append(loss)

	boost_probs = [float(i)/sum(booster_losses) for i in booster_losses] # normalizing losses into probs to sum to 1
	train_index = np.random.choice(len(y), size=int(config.split_ratio*len(boost_probs)), replace=False, p=boost_probs)
	val_index = np.array([i for i in np.arange(len(y)) if i not in train_index])

	x_train, x_val = X2[train_index], X2[val_index]
	y_train, y_val = y[train_index], y[val_index]
	booster_losses = np.array(booster_losses)[train_index]

	results = trainer.train_a_fold(boosted_model_type, x_train, y_train, x_val, y_val, fold, 
		sample_weight=booster_losses, config=config)

	return results

def boosted_ensemble_training(data, config):
	'''
	Training order is same as model_types
	'''

	results = {}
	model_types = config.model_types

	# Train model type, models saved in `model_dir/{model_type}/fold_{fold}.h5`
	m_results = trainer.train_n_folds(model_types[0], data, config)
	results[model_types[0]] = m_results

	if len(model_types) > 1:
		for m_i in range(1, len(model_types)):
			m_results = []
			for fold in range(1, config.n_folds+1):
				m_results_fold = boosted_train_a_fold(data, model_types[m_i-1], model_types[m_i], fold, config)
				m_results.append(m_results_fold)

			results[model_types[m_i]] = m_results
	
	return results