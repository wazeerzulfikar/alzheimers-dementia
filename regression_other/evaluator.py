from models import build_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.model_selection import KFold
import numpy as np
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import os


def evaluate_n(config, data):

	if config.mod_split == 'none':
		evaluate_normal(config, data)
	   
	elif config.mod_split == 'human':
		evaluate_human(config, data)

	else:
		evaluate_kl(config, data)

def standard_scale(x_train, x_test):
	scalar = StandardScaler()
	scalar.fit(x_train)
	x_train = scalar.transform(x_train)
	x_test = scalar.transform(x_test)
	return x_train, x_test

def evaluate_human(config, data):
	X1 = data['X1']
	X2 = data['X2']
	y = data['y']

	final_train1_score, final_val1_score = [], []
	final_train2_score, final_val2_score = [], []

	final_train1_rmse, final_val1_rmse = [], []
	final_train2_rmse, final_val2_rmse = [], []

	final_train_score, final_val_score = [], []
	final_train_rmse, final_val_rmse = [], []


	kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=42)
	fold=1
	for train_index, test_index in kf.split(X1, [0]*len(X1)):
	### starts here
		x_train1 = np.asarray(X1[train_index])
		x_val1 = np.asarray(X1[test_index])
		x_train2 = np.asarray(X2[train_index])
		x_val2 = np.asarray(X2[test_index])

		y_train = np.asarray(y[train_index])
		y_val = np.asarray(y[test_index])

		x_train1, x_val1 = standard_scale(x_train1, x_val1)
		x_train2, x_val2 = standard_scale(x_train2, x_val2)

		fold_train1_score, fold_val1_score = [], []
		fold_train2_score, fold_val2_score = [], []
		fold_train_score, fold_val_score = [], []
		
		mus_train1, mus_val1, sigmas_train1, sigmas_val1 = [], [], [], []
		mus_train2, mus_val2, sigmas_train2, sigmas_val2 = [], [], [], []
		mus_train, mus_val, sigmas_train, sigmas_val = [], [], [], []
		
		train_preds1, val_preds1 = [0]*len(x_train1), [0]*len(x_val1)
		train_preds2, val_preds2 = [0]*len(x_train2), [0]*len(x_val2)

		best_epochs1, best_epochs2 = [], []
		for model_number in range(config.n_models):
			
			if config.build_model=='point':
				model1, history1 = train_a_fold(fold, model_number+1, config, x_train1, y_train, x_val1, y_val)
			
				model2, history2 = train_a_fold(fold, model_number+1, config, x_train2, y_train, x_val2, y_val)
			
				# train_score = model.evaluate(x_train, y_train, verbose=0)
				train_preds1 += model1.predict(x_train1)[:, 0]
				val_preds1 += model1.predict(x_val1)[:, 0]
				
				train_preds2 += model2.predict(x_train2)[:, 0]
				val_preds2 += model2.predict(x_val2)[:, 0]
				
				# val_score = model.evaluate(x_val, y_val, verbose=0)
				# train_score = math.sqrt(train_score)
				# val_score = math.sqrt(val_score)
				# fold_train_score.append(train_score)
				# fold_val_score.append(val_score)

			if config.build_model=='gaussian':
				model1, history1 = train_a_fold(fold, model_number+1, config, x_train1, y_train, x_val1, y_val)
				pred_train1 = model1(x_train1)
				pred_val1 = model1(x_val1)
				val_score1 = np.min(history1.history['val_loss'])
				train_score1 = history1.history['loss'][np.argmin(history1.history['val_loss'])]
				mu_val1 = pred_val1.mean()
				sigma_val1 = pred_val1.stddev()
				mus_val1.append(mu_val1.numpy())
				sigmas_val1.append(sigma_val1.numpy())
				mu_train1 = pred_train1.mean()
				sigma_train1 = pred_train1.stddev()
				mus_train1.append(mu_train1.numpy())
				sigmas_train1.append(sigma_train1.numpy())

				model2, history2 = train_a_fold(fold, model_number+1, config, x_train2, y_train, x_val2, y_val)
				pred_train2 = model2(x_train2)
				pred_val2 = model2(x_val2)
				val_score2 = np.min(history2.history['val_loss'])
				train_score2 = history2.history['loss'][np.argmin(history2.history['val_loss'])]
				mu_val2 = pred_val2.mean()
				sigma_val2 = pred_val2.stddev()
				mus_val2.append(mu_val2.numpy())
				sigmas_val2.append(sigma_val2.numpy())
				mu_train2 = pred_train2.mean()
				sigma_train2 = pred_train2.stddev()
				mus_train2.append(mu_train2.numpy())
				sigmas_train2.append(sigma_train2.numpy())

				mus_train.append((mu_train1.numpy() + mu_train2.numpy())/2)
				mus_val.append((mu_val1.numpy() + mu_val2.numpy())/2)
				sigmas_train.append((sigma_train1.numpy() + sigma_train2.numpy())/2)
				sigmas_val.append((sigma_val1.numpy() + sigma_val2.numpy())/2)

			best_epochs1.append(np.argmin(history1.history['val_loss'])+1)
			best_epochs2.append(np.argmin(history2.history['val_loss'])+1)
			# print('\nFold ', fold, ' Model number:', model_number+1, ' Train score:', train_score, ' Val score:', val_score, ' Best epoch:', best_epoch, '\n')

		if config.build_model=='point':
			train_preds1 /= 5
			val_preds1 /= 5
			train_preds1 = np.asarray(train_preds1)
			val_preds1 = np.asarray(val_preds1)

			train_preds2 /= 5
			val_preds2 /= 5
			train_preds2 = np.asarray(train_preds2)
			val_preds2 = np.asarray(val_preds2)

			final_train1_score.append(mean_squared_error(y_train, train_preds1, squared=False))
			final_val1_score.append(mean_squared_error(y_val, val_preds1, squared=False))

			final_train2_score.append(mean_squared_error(y_train, train_preds2, squared=False))
			final_val2_score.append(mean_squared_error(y_val, val_preds2, squared=False))

			final_train_score.append(mean_squared_error(y_train, (train_preds1 + train_preds2)/2, squared=False))
			final_val_score.append(mean_squared_error(y_val, (val_preds1 + val_preds2)/2, squared=False))
			
			print('\nFold ', fold, ' Model1 Train score:', final_train1_score[-1], ' Val score:', final_val1_score[-1], ' Best epoch:', best_epochs1[-1], '\n')
			print('\nFold ', fold, ' Model2 Train score:', final_train2_score[-1], ' Val score:', final_val2_score[-1], ' Best epoch:', best_epochs2[-1], '\n')
			print('\nFold ', fold, ' Ensemble Train score:', final_train_score[-1], ' Val score:', final_val_score[-1], '\n')

		if config.build_model=='gaussian':
			mus_train1, sigmas_train1 = np.concatenate(mus_train1, axis=-1), np.concatenate(sigmas_train1, axis=-1)
			ensemble_mu_train1 = np.mean(mus_train1, axis=-1).reshape(-1,1)
			ensemble_sigma_train1 = np.sqrt(np.mean(np.square(sigmas_train1) + np.square(mus_train1), axis=-1).reshape(-1,1) - np.square(ensemble_mu_train1))
			
			mus_train2, sigmas_train2 = np.concatenate(mus_train2, axis=-1), np.concatenate(sigmas_train2, axis=-1)
			ensemble_mu_train2 = np.mean(mus_train2, axis=-1).reshape(-1,1)
			ensemble_sigma_train2 = np.sqrt(np.mean(np.square(sigmas_train2) + np.square(mus_train2), axis=-1).reshape(-1,1) - np.square(ensemble_mu_train2))
			
			mus_train, sigmas_train = np.concatenate(mus_train, axis=-1), np.concatenate(sigmas_train, axis=-1)
			ensemble_mu_train = np.mean(mus_train, axis=-1).reshape(-1,1)
			ensemble_sigma_train = np.sqrt(np.mean(np.square(sigmas_train) + np.square(mus_train), axis=-1).reshape(-1,1) - np.square(ensemble_mu_train))
			
			mus_val1, sigmas_val1 = np.concatenate(mus_val1, axis=-1), np.concatenate(sigmas_val1, axis=-1)
			ensemble_mu_val1 = np.mean(mus_val1, axis=-1).reshape(-1,1)
			ensemble_sigma_val1 = np.sqrt(np.mean(np.square(sigmas_val1) + np.square(mus_val1), axis=-1).reshape(-1,1) - np.square(ensemble_mu_val1))

			mus_val2, sigmas_val2 = np.concatenate(mus_val2, axis=-1), np.concatenate(sigmas_val2, axis=-1)
			ensemble_mu_val2 = np.mean(mus_val2, axis=-1).reshape(-1,1)
			ensemble_sigma_val2 = np.sqrt(np.mean(np.square(sigmas_val2) + np.square(mus_val2), axis=-1).reshape(-1,1) - np.square(ensemble_mu_val2))

			mus_val, sigmas_val = np.concatenate(mus_val, axis=-1), np.concatenate(sigmas_val, axis=-1)
			ensemble_mu_val = np.mean(mus_val, axis=-1).reshape(-1,1)
			ensemble_sigma_val = np.sqrt(np.mean(np.square(sigmas_val) + np.square(mus_val), axis=-1).reshape(-1,1) - np.square(ensemble_mu_val))

			tfd = tfp.distributions

			ensemble_dist_train1 = tfd.Normal(loc=ensemble_mu_train1, scale=ensemble_sigma_train1)
			ensemble_dist_val1 = tfd.Normal(loc=ensemble_mu_val1, scale=ensemble_sigma_val1)
			ensemble_true_train1_log_probs = ensemble_dist_train1.log_prob(y_train).numpy()
			final_train1_score.append(np.mean(-ensemble_true_train1_log_probs))
			ensemble_true_val1_log_probs = ensemble_dist_val1.log_prob(y_val).numpy()
			final_val1_score.append(np.mean(-ensemble_true_val1_log_probs))
			final_train1_rmse.append(mean_squared_error(y_train, ensemble_mu_train1, squared=False))
			final_val1_rmse.append(mean_squared_error(y_val, ensemble_mu_val1, squared=False))

			ensemble_dist_train2 = tfd.Normal(loc=ensemble_mu_train2, scale=ensemble_sigma_train2)
			ensemble_dist_val2 = tfd.Normal(loc=ensemble_mu_val2, scale=ensemble_sigma_val2)
			ensemble_true_train2_log_probs = ensemble_dist_train2.log_prob(y_train).numpy()
			final_train2_score.append(np.mean(-ensemble_true_train2_log_probs))
			ensemble_true_val2_log_probs = ensemble_dist_val2.log_prob(y_val).numpy()
			final_val2_score.append(np.mean(-ensemble_true_val2_log_probs))
			final_train2_rmse.append(mean_squared_error(y_train, ensemble_mu_train2, squared=False))
			final_val2_rmse.append(mean_squared_error(y_val, ensemble_mu_val2, squared=False))

			ensemble_dist_train = tfd.Normal(loc=ensemble_mu_train, scale=ensemble_sigma_train)
			ensemble_dist_val = tfd.Normal(loc=ensemble_mu_val, scale=ensemble_sigma_val)
			ensemble_true_train_log_probs = ensemble_dist_train.log_prob(y_train).numpy()
			final_train_score.append(np.mean(-ensemble_true_train_log_probs))
			ensemble_true_val_log_probs = ensemble_dist_val.log_prob(y_val).numpy()
			final_val_score.append(np.mean(-ensemble_true_val_log_probs))
			final_train_rmse.append(mean_squared_error(y_train, ensemble_mu_train, squared=False))
			final_val_rmse.append(mean_squared_error(y_val, ensemble_mu_val, squared=False))

			print('\nFold ', fold, ' Model1 Train RMSE:', final_train1_rmse[-1], ' Val RMSE:', final_val1_rmse[-1], ' Train score:', final_train1_score[-1], ' Val score:', final_val1_score[-1], ' Best epoch:', best_epochs1[-1], '\n')
			print('\nFold ', fold, ' Model2 Train RMSE:', final_train2_rmse[-1], ' Val RMSE:', final_val2_rmse[-1], ' Train score:', final_train2_score[-1], ' Val score:', final_val2_score[-1], ' Best epoch:', best_epochs2[-1], '\n')
			print('\nFold ', fold, ' Ensemble Train RMSE:', final_train_score[-1], ' Val RMSE:', final_val_score[-1], ' Train score:', final_train_score[-1], ' Val score:', final_val_score[-1], '\n')

		# best_epochs.append(best_epoch)    
		fold+=1
	# print("\nBest epochs : ", best_epochs)
	if config.build_model=='point':
		print("Train1 RMSE mean : ", np.mean(final_train1_score), "+/-", np.std(final_train1_score))
		print("Val1 RMSE mean : ", np.mean(final_val1_score), "+/-", np.std(final_val1_score))
		print("Train2 RMSE mean : ", np.mean(final_train2_score), "+/-", np.std(final_train2_score))
		print("Val2 RMSE mean : ", np.mean(final_val2_score), "+/-", np.std(final_val2_score))
		print("Ensemble Train RMSE mean : ", np.mean(final_train_score), "+/-", np.std(final_train_score))
		print("Ensemble Val RMSE mean : ", np.mean(final_val_score), "+/-", np.std(final_val_score))
	if config.build_model=='gaussian':
		print("\nModel 1\nTrain NLL : ", final_train1_score)
		print("Val NLL : ", final_val1_score)
		print("Train NLL mean : ", np.mean(final_train1_score), "+/-", np.std(final_train1_score))
		print("Val NLL mean : ", np.mean(final_val1_score), "+/-", np.std(final_val1_score))
		print("\nTrain RMSE : ", final_train1_rmse)
		print("Val RMSE : ", final_val1_rmse)		
		print("Train RMSE mean : ", np.mean(final_train1_rmse), "+/-", np.std(final_train1_rmse))
		print("Val RMSE mean : ", np.mean(final_val1_rmse), "+/-", np.std(final_val1_rmse))

		print("\nModel2\nTrain NLL : ", final_train2_score)
		print("Val NLL : ", final_val2_score)
		print("Train NLL mean : ", np.mean(final_train2_score), "+/-", np.std(final_train2_score))
		print("Val NLL mean : ", np.mean(final_val2_score), "+/-", np.std(final_val2_score))
		print("\nTrain RMSE : ", final_train2_rmse)
		print("Val RMSE : ", final_val2_rmse)		
		print("Train RMSE mean : ", np.mean(final_train2_rmse), "+/-", np.std(final_train2_rmse))
		print("Val RMSE mean : ", np.mean(final_val2_rmse), "+/-", np.std(final_val2_rmse))

		print("\nEnsemble\nTrain NLL : ", final_train_score)
		print("Val NLL : ", final_val_score)
		print("Train NLL mean : ", np.mean(final_train_score), "+/-", np.std(final_train_score))
		print("Val NLL mean : ", np.mean(final_val_score), "+/-", np.std(final_val_score))
		print("\nTrain RMSE : ", final_train_rmse)
		print("Val RMSE : ", final_val_rmse)		
		print("Train RMSE mean : ", np.mean(final_train_rmse), "+/-", np.std(final_train_rmse))
		print("Val RMSE mean : ", np.mean(final_val_rmse), "+/-", np.std(final_val_rmse))		


	### ends here




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
		train_preds, val_preds = [0]*len(x_train), [0]*len(x_val)
		best_epochs = []
		for model_number in range(config.n_models):
			model, history = train_a_fold(fold, model_number+1, config, x_train, y_train, x_val, y_val)
			
			if config.build_model=='point':
				# train_score = model.evaluate(x_train, y_train, verbose=0)
				train_preds += model.predict(x_train)[:, 0]
				val_preds += model.predict(x_val)[:, 0]
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
			best_epochs.append(best_epoch)
			# print('\nFold ', fold, ' Model number:', model_number+1, ' Train score:', train_score, ' Val score:', val_score, ' Best epoch:', best_epoch, '\n')

		if config.build_model=='point':
			train_preds /= 5
			val_preds /= 5
			train_preds = np.asarray(train_preds)
			val_preds = np.asarray(val_preds)
			print(train_preds.shape, val_preds.shape)
			final_train_score.append(mean_squared_error(y_train, train_preds, squared=False))
			final_val_score.append(mean_squared_error(y_val, val_preds, squared=False))
			print('\nFold ', fold, ' Train score:', final_train_score[-1], ' Val score:', final_val_score[-1], ' Best epoch:', best_epochs[-1], '\n')


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

	# if config.build_model=='point':
	# 	epochs = 200
	# 	lr = 0.05 # 0.005
	# 	batch_size = 32
	# elif config.build_model=='gaussian':
	# 	# epochs = 300
	# 	# lr = 0.05
	# 	# batch_size = 32
	# 	epochs = 40
	# 	lr = 0.1
	# 	batch_size = 100

	epochs = config.epochs
	lr = config.learning_rate
	batch_size = config.batch_size

	model.compile(loss=loss, optimizer=Adam(learning_rate=lr))
	checkpoint_filepath = os.path.join(config.model_dir, config.dataset, config.expt_name, 'fold_{}_model_{}.h5'.format(fold, model_number))
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
				  verbose=0,
				  callbacks=[checkpoints],
				  validation_data=(x_val, y_val))

	if config.build_model=='point':
		model = load_model(checkpoint_filepath)
	elif config.build_model=='gaussian':
		model.load_weights(checkpoint_filepath)

	return model, history