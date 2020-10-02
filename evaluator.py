import os
import glob

import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, mean_squared_error
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pickle import load
np.random.seed(0)

# Local imports
import dataset

def evaluate(data, test_data, config):
	print('Loading data...')

	if config.task == 'classification':
		y = data['y_clf']
		y_test = test_data['y_clf']

	elif config.task == 'regression':
		y = data['y_reg']
		y_test = test_data['y_reg']

	model_dir = config.model_dir
	model_types = config.model_types
	voting_type = config.voting_type
	dataset_split = config.dataset_split
	n_folds = config.n_folds
	task = config.task
	fold=0

	saved_model_types = {}

	if config.uncertainty:

		def negloglik(y, p_y):
			return -p_y.log_prob(y)

		for m in model_types:
			model_files = sorted(glob.glob(os.path.join(model_dir, '{}/*.h5'.format(m))))
			saved_models = list(map(lambda x: tf.keras.models.load_model(x, custom_objects={'negloglik': negloglik})
				, model_files))
			saved_model_types[m] = saved_models
	else:
		for m in model_types:
			model_files = sorted(glob.glob(os.path.join(model_dir, '{}/*.h5'.format(m))))
			saved_models = list(map(lambda x: tf.keras.models.load_model(x), model_files))
			saved_model_types[m] = saved_models

	print('Loading models from {}'.format(model_dir))
	print('Using {} on {}'.format(voting_type, dataset_split))
	print('Models evaluated ', model_types)

	train_accuracies = []
	val_accuracies = []
	test_accuracies = []

	average_uncertainties = []
	average_entropies = []

	if dataset_split == 'full_dataset': # compare features need to be projected

		numpy_seeds = [913293, 653261, 84754, 645, 13451235]

		for i in range(config.n_folds):
			np.random.seed(numpy_seeds[i])

			p = np.random.permutation(len(y))
			for m in model_types:
				data[m] = data[m][p]
			y = y[p]

			n_train = int(config.split_ratio * len(y))
			compare_train, compare_val = data['compare'][:n_train], data['compare'][n_train:]
			compare_test = test_data['compare']
			y_train, y_val = y[:n_train], y[n_train:]

			sc = load(open(os.path.join(config.model_dir, 'compare/scaler_{}.pkl'.format(fold+1)), 'rb'))
			pca = load(open(os.path.join(config.model_dir, 'compare/pca_{}.pkl'.format(fold+1)), 'rb'))

			compare_train = sc.transform(compare_train)
			compare_train = pca.transform(compare_train)
			compare_val = sc.transform(compare_val)
			compare_val = pca.transform(compare_val)
			compare_test = sc.transform(compare_test)
			compare_test = pca.transform(compare_test)

			if len(model_types) == 1:
				m = model_types[0]
				if m == 'compare':
					print('Fold {}'.format(fold+1))
					print('Train')
					train_accuracy = get_individual_accuracy(task, saved_model_types[m][fold], compare_train, y_train, config, fold=fold)
					print('Val')
					val_accuracy = get_individual_accuracy(task, saved_model_types[m][fold], compare_val, y_val, config, fold=fold)
					print('Test')
					test_accuracy = get_individual_accuracy(task, saved_model_types[m][fold], compare_test, y_test, config, fold=fold)
				else:
					print('Fold {}'.format(fold+1))
					print('Train')
					train_accuracy = get_individual_accuracy(task, saved_model_types[m][fold], data[m][:n_train], y_train, config, fold=fold)
					print('Val')
					val_accuracy = get_individual_accuracy(task, saved_model_types[m][fold], data[m][n_train:], y_val, config, fold=fold)
					print('Test')
					test_accuracy = get_individual_accuracy(task, saved_model_types[m][fold], test_data[m], y_test, config, fold=fold)
			else:
				models = []
				features = []
				for m in model_types:
					models.append(saved_model_types[m][fold])
					if m == 'compare':
						features.append(compare_train)
					else:	
						features.append(data[m][:n_train])

				print('Fold {}'.format(fold+1))
				print('Train')
				train_accuracy, learnt_voter, _ = get_ensemble_accuracy(task, models, features, y_train, config)

				print('Val')
				features = []
				for m in model_types:
					if m == 'compare':
						features.append(compare_val)
					else:	
						features.append(data[m][n_train:])
				val_accuracy, _, _ = get_ensemble_accuracy(task, models, features, y_val, config, learnt_voter=learnt_voter,  fold=fold)

				print('Test')
				features = []
				for m in model_types:
					if m == 'compare':
						features.append(compare_test)
					else:	
						features.append(test_data[m])
				test_accuracy, _, average_results = get_ensemble_accuracy(task, models, features, y_test, config, learnt_voter=learnt_voter,  fold=fold, plot=config.plot)

				print('----'*10)

			train_accuracies.append(train_accuracy)
			val_accuracies.append(val_accuracy)
			test_accuracies.append(test_accuracy)

			if config.uncertainty  and len(config.model_types) > 1:
				average_uncertainties.append(average_results[0])
				average_entropies.append(average_results[1])

			fold+=1


	if dataset_split == 'k_fold':
		
		for train_index, val_index in KFold(n_folds).split(y):
			compare_train, compare_val = data['compare'][train_index], data['compare'][val_index]
			y_train, y_val = y[train_index], y[val_index]

			sc = StandardScaler()
			sc.fit(compare_train)

			compare_train = sc.transform(compare_train)
			compare_val = sc.transform(compare_val)

			pca = PCA(n_components=config.compare_features_size)
			pca.fit(compare_train)

			compare_train = pca.transform(compare_train)
			compare_val = pca.transform(compare_val)


			if len(model_types) == 1:
				m = model_types[0]
				if m == 'compare':
					print('Fold {}'.format(fold+1))
					print('Train')
					train_accuracy = get_individual_accuracy(task, saved_model_types[m][fold], compare_train, y_train, fold=fold)
					print('Val')
					val_accuracy = get_individual_accuracy(task, saved_model_types[m][fold], compare_val, y_val, fold=fold)

				else:
					print('Fold {}'.format(fold+1))
					print('Train')
					train_accuracy = get_individual_accuracy(task, saved_model_types[m][fold], data[m][train_index], y_train, fold=fold)
					print('Val')
					val_accuracy = get_individual_accuracy(task, saved_model_types[m][fold], data[m][val_index], y_val, fold=fold)

			else:

				models = []
				features = []
				for m in model_types:
					models.append(saved_model_types[m][fold])
					if m == 'compare':
						features.append(compare_train)
					else:	
						features.append(data[m][train_index])

				print('Fold {}'.format(fold+1))
				print('Train')
				train_accuracy, learnt_voter = get_ensemble_accuracy(task, models, features, y_train, config)

				print('Val')
				features = []
				for m in model_types:
					if m == 'compare':
						features.append(compare_val)
					else:	
						features.append(data[m][val_index])
				val_accuracy, _ = get_ensemble_accuracy(task, models, features, y_val, config, learnt_voter=learnt_voter,  fold=fold)

				print('----'*10)

			train_accuracies.append(train_accuracy)
			val_accuracies.append(val_accuracy)
			test_accuracies.append(test_accuracy)

			fold+=1

	print('Train mean: {:.3f}'.format(np.mean(train_accuracies)))
	print('Train std: {:.3f}'.format(np.std(train_accuracies)))
	if len(val_accuracies) > 0:
		print('Val mean: {:.3f}'.format(np.mean(val_accuracies)))
		print('Val std: {:.3f}'.format(np.std(val_accuracies)))
	if len(test_accuracies) > 0:
		print('Test mean: {:.3f}'.format(np.mean(test_accuracies)))
		print('Test std: {:.3f}'.format(np.std(test_accuracies)))

	if config.uncertainty and len(config.model_types) > 1:
		print('Test Average Uncertainties: ', list(np.mean(average_uncertainties, axis=0)))
		print('Test Average Entropies: ', list(np.mean(average_entropies, axis=0)))


def get_individual_accuracy(task, model, feature, y, config, fold=None):

	if task == 'classification':
		preds = model.predict(feature)
		preds = np.argmax(preds, axis=-1)
		accuracy = accuracy_score(np.argmax(y, axis=-1), preds)
		report = precision_recall_fscore_support(np.argmax(y, axis=-1), preds, average='binary')
		print('precision: {:.3f}, recall: {:.3f}, f1_score: {:.3f}, accuracy: {:.3f}'.format(report[0], report[1], report[2], accuracy))
		return accuracy

	elif task == 'regression':
		if config.uncertainty:
			preds = model(feature).mean().numpy()
		else:
			preds = model.predict(feature)

		y = np.array(y)
		score = mean_squared_error(np.expand_dims(y, axis=-1), preds, squared=False)
		print('rmse ', score)
		return score

def get_ensemble_accuracy(task, models, features, y, config, num_classes=2, learnt_voter=None, fold=None, plot=False):
	
	if task == 'classification':
		probs = []
		for model, feature in zip(models, features):
			pred = model.predict(feature)
			probs.append(pred)
		probs = np.stack(probs, axis=1)

		if config.voting_type=='hard_voting':
			model_predictions = np.argmax(probs, axis=-1)
			model_predictions = np.squeeze(model_predictions)
			voted_predictions = [max(set(i), key = list(i).count) for i in model_predictions]
		elif config.voting_type=='soft_voting':
			model_predictions = np.sum(probs, axis=1)
			voted_predictions = np.argmax(model_predictions, axis=-1)
		elif config.voting_type=='learnt_voting':
			model_predictions = np.reshape(probs, (len(y), -1))
			if learnt_voter is None:
				learnt_voter = LogisticRegression(C=0.1).fit(model_predictions, np.argmax(y, axis=-1))
			# print('Voter coef ', voter.coef_)
			voted_predictions = learnt_voter.predict(model_predictions)

		accuracy = accuracy_score(np.argmax(y, axis=-1), voted_predictions)
		report = precision_recall_fscore_support(np.argmax(y, axis=-1), voted_predictions, average='binary')
		print('precision: {:.3f}, recall: {:.3f}, f1_score: {:.3f}, accuracy: {:.3f}'.format(report[0], report[1], report[2], accuracy))

		return accuracy, learnt_voter, None

	elif task == 'regression':

		if config.voting_type == 'hard_voting':
			preds = []
			pred_stds = []
			pred_entropies = []

			for model, feature in zip(models, features):
				if config.uncertainty:
					predictions = model(feature)
					probs = predictions.mean().numpy()
					probs_std = predictions.stddev().numpy()
					probs_entropy = predictions.entropy().numpy()

					pred_stds.append(probs_std)
					pred_entropies.append(probs_entropy)

				else:
					probs = model.predict(feature)
				preds.append(probs)

			preds = np.stack(preds, axis=1) # 86,3,1
			voted_predictions = np.mean(preds, axis=1)

			pred_stds = np.stack(pred_stds, axis=1) # N,3,1
			pred_entropies = np.stack(pred_entropies, axis=1) # N,3,1

		elif config.voting_type == 'uncertainty_voting':
			pred_means = []
			pred_stds = []
			pred_entropies = []

			for model, feature in zip(models, features):
				probs = model(feature)
				probs_mean = probs.mean().numpy()
				probs_std = probs.stddev().numpy()
				probs_entropy = probs.entropy().numpy()

				pred_means.append(probs_mean)
				pred_stds.append(probs_std)
				pred_entropies.append(probs_entropy)

			pred_means = np.stack(pred_means, axis=1) # N,3,1
			pred_stds = np.stack(pred_stds, axis=1) # N,3,1
			pred_entropies = np.stack(pred_entropies, axis=1) # N,3,1

			std_inverse = np.reciprocal(pred_stds)
			std_sums = np.sum(std_inverse, axis=1, keepdims=True)

			voting_weights = std_inverse / std_sums
			voted_predictions = np.sum(pred_means * voting_weights, axis=1)

		if config.uncertainty:
			average_uncertainties = np.squeeze(np.mean(pred_stds, axis=0))
			average_entropies = np.squeeze(np.mean(pred_entropies, axis=0))
			print('Average Uncertainties ', average_uncertainties)
			print('Average Entropies ', average_entropies)

		if plot:
			plot_entropy(pred_entropies, fold, config)

		score = mean_squared_error(np.expand_dims(y, axis=-1), voted_predictions, squared=False)
		print('rmse: {:.3f}'.format(score))

		if config.task == 'regression' and config.uncertainty:
			return score, None, [average_uncertainties, average_entropies]
		return score, None, None

def plot_entropy(entropies, fold, config):

	for i in range(3):
		b = sns.distplot(entropies[:,i,:], hist = False, kde = True,
	                 kde_kws = {'linewidth': 3}, label=config.model_types[i])

	b.set_title('Entropy on Test Set',  fontsize=26)
	b.set_ylabel('Density', fontsize=26)

	os.makedirs(os.path.join(config.model_dir, 'plots'), exist_ok=True)

	b.legend(loc='upper right', fontsize=14)
	plt.tight_layout(pad=0)
	plt.savefig(os.path.join(config.model_dir, 'plots/fold_{}.png'.format(fold)), dpi=300)
	plt.clf()
	plt.close()



