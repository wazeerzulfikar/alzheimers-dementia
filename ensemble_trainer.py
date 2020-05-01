import os
import numpy as np
import tensorflow as tf

import dataset
import trainer

def bagging_ensemble_training(dataset_dir, model_dir, n_splits):

	data = dataset.prepare_data(dataset_dir)
	X_intervention, X_pause, X_spec, X_compare, X_reg_intervention, X_reg_pause, X_reg_compare = data[0:7]
	y, y_reg, filenames_intervention, filenames_pause, filenames_spec, filenames_compare = data[7:]

	## Train Intervention, models saved in `model_dir/intervention_{fold}.h5`
	intervention_results = trainer.train_n_folds('intervention', X_intervention, y, n_splits, model_dir)
	pause_results = trainer.train_n_folds('pause', X_pause, y, n_splits, model_dir)
	# spectogram_results = trainer.train_n_folds('spectogram', X_spec, y, n_splits, model_dir)
	compare_results = trainer.train_n_folds('compare', X_compare, y, n_splits, model_dir)

	return {
		'intervention_results' : intervention_results, 
		'pause_results': pause_results, 
		# 'spectogram_results' : spectogram_results,
		'compare_results' : compare_results
	}

def boosted_train_a_fold(
	booster_model_type, 
	X1,
	y,
	boosted_model_type,
	X2,
	fold, 
	model_dir,
	):

	booster_model = tf.keras.models.load_model(os.path.join(model_dir, booster_model_type, 'fold_{}.h5'.format(fold)))

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

def boosted_ensemble_training(dataset_dir, model_dir, n_splits=5):
	
	data = dataset.prepare_data(dataset_dir)
	X_intervention, X_pause, X_spec, X_compare, X_reg_intervention, X_reg_pause, X_reg_compare = data[0:7]
	y, y_reg, filenames_intervention, filenames_pause, filenames_spec, filenames_compare = data[7:]

	## Train Intervention, models saved in `model_dir/intervention/fold_{fold}.h5`
	intervention_results = trainer.train_n_folds('intervention', X_intervention, y, n_splits, model_dir)

	# INTERVENTION -> PAUSE
	pause_results = []
	for fold in range(1, n_splits+1):

		pause_results_fold = boosted_train_a_fold('intervention', X_intervention, y, 'pause', X_pause, fold, model_dir)
		pause_results.append(pause_results_fold)

	# PAUSE -> SPECTROGRAM
	spectogram_results = []
	for fold in range(1, n_splits+1):
		spectogram_results_fold = boosted_train_a_fold('pause', X_pause, y, 'spectogram', X_spec, fold, model_dir)
		spectogram_results.append(spectogram_results_fold)
	
	return {
		'intervention_results' : intervention_results, 
		'pause_results': pause_results, 
		'spectogram_results' : spectogram_results
	}