from models import build_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.model_selection import KFold
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
import os, math

def evaluate_n(config, data):
	
	if config.mod_split=='none':
		evaluate_normal(config, data)
	   
	elif config.mod_split=='human':
		evaluate_human(config, data)

	else:
		evaluate_kl(config, data)

def standard_scale(x_train, x_test):
	scalar = StandardScaler()
	scalar.fit(x_train)
	x_train = scalar.transform(x_train)
	x_test = scalar.transform(x_test)
	return x_train, x_test

def evaluate_normal(config, data):
	X = data['X']
	y = data['y']

	final_train_score, final_val_score = [], []
	final_train_rmse, final_val_rmse = [], []
	best_epochs = []
	
	kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=42)
	fold=1
	for train_index, test_index in kf.split(X, [0]*len(X)):
		x_train = np.asarray(X[train_index])
		y_train = np.asarray(y[train_index])
		x_val = np.asarray(X[test_index])
		y_val = np.asarray(y[test_index])

		x_train, x_val = standard_scale(x_train, x_val)

		fold_train_score, fold_val_score = [], []
		mus_train, mus_val, sigmas_train, sigmas_val = [], [], [], []
		train_preds, val_preds = [], []
		for model_number in range(config.n_models):
			model, history = train_a_fold(fold, model_number+1, config, x_train, y_train, x_val, y_val)
			
			if config.build_model=='point':
				# train_score = model.evaluate(x_train, y_train, verbose=0)
				train_preds.append(np.squeeze(model.predict(x_train)))
				val_preds.append(np.squeeze(model.predict(x_val)))
				# val_score = model.evaluate(x_val, y_val, verbose=0)
				# train_score = math.sqrt(train_score)
				# val_score = math.sqrt(val_score)
				# fold_train_score.append(train_score)
				# fold_val_score.append(val_score)

			if config.build_model=='gaussian':
				pred_train = model(x_train)
				pred_val = model(x_val)
				val_score = np.min(history.history['val_loss'])
				train_score = history.history['loss'][np.argmin(history.history['val_loss'])]
				mu_val = pred_val.mean()
				sigma_val = pred_val.stddev()
				mus_val.append(mu_val.numpy())
				sigmas_val.append(sigma_val.numpy())
				mu_train = pred_train.mean()
				sigma_train = pred_train.stddev()
				mus_train.append(mu_train.numpy())
				sigmas_train.append(sigma_train.numpy())

			best_epoch = np.argmin(history.history['val_loss'])+1
			print('\nFold ', fold, ' Model number:', model_number+1, ' Train score:', train_score, ' Val score:', val_score, ' Best epoch:', best_epoch, '\n')

		if config.build_model=='point':
			final_train_score.append(mean_squared_error(y_train, np.mean(train_preds), squared=False))
			final_val_score.append(mean_squared_error(y_val, np.mean(val_preds), squared=False))


		if config.build_model=='gaussian':
			mus_train, sigmas_train = np.concatenate(mus_train, axis=-1), np.concatenate(sigmas_train, axis=-1)
			ensemble_mu_train = np.mean(mus_train, axis=-1).reshape(-1,1)
			ensemble_sigma_train = np.sqrt(np.mean(np.square(sigmas_train) + np.square(mus_train), axis=-1).reshape(-1,1) - np.square(ensemble_mu_train))
			
			mus_val, sigmas_val = np.concatenate(mus_val, axis=-1), np.concatenate(sigmas_val, axis=-1)
			ensemble_mu_val = np.mean(mus_val, axis=-1).reshape(-1,1)
			ensemble_sigma_val = np.sqrt(np.mean(np.square(sigmas_val) + np.square(mus_val), axis=-1).reshape(-1,1) - np.square(ensemble_mu_val))

			tfd = tfp.distributions
			ensemble_dist_train = tfd.Normal(loc=ensemble_mu_train, scale=ensemble_sigma_train)
			ensemble_dist_val = tfd.Normal(loc=ensemble_mu_val, scale=ensemble_sigma_val)

			ensemble_true_train_log_probs = ensemble_dist_train.log_prob(y_train).numpy()
			final_train_score.append(np.mean(-ensemble_true_train_log_probs))
			ensemble_true_val_log_probs = ensemble_dist_val.log_prob(y_val).numpy()
			final_val_score.append(np.mean(-ensemble_true_val_log_probs))
			final_train_rmse.append(mean_squared_error(y_train, ensemble_mu_train, squared=False))
			final_val_rmse.append(mean_squared_error(y_val, ensemble_mu_val, squared=False))

		# best_epochs.append(best_epoch)    
		fold+=1
	# print("\nBest epochs : ", best_epochs)
	if config.build_model=='point':
		print("\nTrain RMSE : ", final_train_score)
		print("Val RMSE : ", final_val_score)
		print("Train RMSE mean : ", np.mean(final_train_score), "+/-", np.std(final_train_score))
		print("Val RMSE mean : ", np.mean(final_val_score), "+/-", np.std(final_val_score))
	if config.build_model=='gaussian':
		print("\nTrain NLL : ", final_train_score)
		print("Val NLL : ", final_val_score)
		print("Train NLL mean : ", np.mean(final_train_score), "+/-", np.std(final_train_score))
		print("Val NLL mean : ", np.mean(final_val_score), "+/-", np.std(final_val_score))
		print("\nTrain RMSE : ", final_train_rmse)
		print("Val RMSE : ", final_val_rmse)		
		print("Train RMSE mean : ", np.mean(final_train_rmse), "+/-", np.std(final_train_rmse))
		print("Val RMSE mean : ", np.mean(final_val_rmse), "+/-", np.std(final_val_rmse))		


def train_a_fold(fold, model_number, config, x_train, y_train, x_val, y_val):
	
	model, loss = build_model(config)

	if config.build_model=='point':
		epochs = 200
		lr = 0.05 # 0.005
		batch_size = 32
	elif config.build_model=='gaussian':
		# epochs = 300
		# lr = 0.05
		# batch_size = 32
		epochs = 40
		lr = 0.1
		batch_size = 100

	model.compile(loss=loss, optimizer=Adam(learning_rate=lr))
	checkpoint_filepath = os.path.join(config.model_dir, config.dataset, config.build_model, 'fold_{}_model_{}.h5'.format(fold, model_number))
	checkpoints = ModelCheckpoint(checkpoint_filepath,
								  monitor='val_loss', 
								  verbose=0, 
								  save_best_only=True,
								  save_weights_only=False,
								  mode='auto',
								  save_freq='epoch')

	history = model.fit(x_train, y_train,
				  epochs=epochs,
				  batch_size=batch_size,
				  verbose=1,
				  callbacks=[checkpoints],
				  validation_data=(x_val, y_val))

	if config.build_model=='point':
		model = load_model(checkpoint_filepath)
	elif config.build_model=='gaussian':
		model.load_weights(checkpoint_filepath)

	return model, history


'''
Train RMSE :  [2.818101485645493, 2.500808308234439, 2.6760598506297364, 2.3116475797111287, 2.5202441947228102, 2.516431549841327, 2.5353926243252887, 2.467800544598327, 2.304326575224174, 4.011069650488212, 2.673344289465624, 2.5214935096684696, 2.5578219828372957, 3.5642766761143383, 2.756177426581398, 2.4518696080307576, 2.3235185832888114, 2.534690094251268, 3.040788602126909, 2.485732208879986]
Val RMSE :  [2.4629534186976643, 2.526248241878259, 2.2614667345883506, 4.762374800391421, 3.4070693201631217, 2.411270687032414, 2.9461113441295708, 1.9431623034854273, 3.3365886422216753, 4.8446793848344925, 2.5847270384735106, 2.3810181281095515, 1.8048404567584992, 3.2763241360548827, 2.9627035680180747, 2.1840387062676636, 2.2934069958435908, 2.0932610344914755, 2.8321399581496003, 3.6538315705054862]
Train RMSE mean :  2.67857976723329     RMSE std dev :  0.41315396129133275
Val RMSE mean :  2.8484108235047367     RMSE std dev :  0.8172386232908562

'''