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
import dataset_features

def infer(dataset_dir, model_dir):
	print('Loading data...')

	audio_length_normalization = 10
	spectrogram_size = (480, 640)
	longest_speaker_length = 32
	compare_feature_size = 21

	cc_transcription_files = sorted(glob.glob(os.path.join(dataset_dir, 'transcription/cc/*.cha')))
	cc_audio_files = sorted(glob.glob(os.path.join(dataset_dir, 'Full_wave_enhanced_audio/cc/*.wav')))
	cc_spectogram_files = sorted(glob.glob(os.path.join(dataset_dir, 'spectograms/cc/*.png')))
	cc_compare_files = sorted(glob.glob(os.path.join(dataset_dir, 'compare/cc/*.csv')))

	cc_pause_features, cc_spectogram_features, cc_invervention_features, cc_compare_features = [], [], [], []
	for i in range(len(cc_transcription_files)):
		cc_pause_features.append(dataset_features.get_pause_features(
			transcription_filename=cc_transcription_files[i], 
			audio_filename=cc_audio_files[i], 
			audio_length_normalization=audio_length_normalization))

		cc_spectogram_features.append(dataset_features.get_old_spectogram_features(
			spectogram_filename=cc_spectogram_files[i]))

		cc_invervention_features.append(dataset_features.get_intervention_features(
			transcription_filename=cc_transcription_files[i],
			max_length=longest_speaker_length))

		cc_compare_features.append(dataset_features.get_compare_features(
			compare_filename=cc_compare_files[i]))

	cd_transcription_files = sorted(glob.glob(os.path.join(dataset_dir, 'transcription/cd/*.cha')))
	cd_audio_files = sorted(glob.glob(os.path.join(dataset_dir, 'Full_wave_enhanced_audio/cd/*.wav')))
	cd_spectogram_files = sorted(glob.glob(os.path.join(dataset_dir, 'spectograms/cd/*.png')))
	cd_compare_files = sorted(glob.glob(os.path.join(dataset_dir, 'compare/cd/*.csv')))


	cd_pause_features, cd_spectogram_features, cd_invervention_features, cd_compare_features = [], [], [], []
	for i in range(len(cd_transcription_files)):
		cd_pause_features.append(dataset_features.get_pause_features(
			transcription_filename=cd_transcription_files[i], 
			audio_filename=cd_audio_files[i], 
			audio_length_normalization=audio_length_normalization))

		cd_spectogram_features.append(dataset_features.get_old_spectogram_features(
			spectogram_filename=cd_spectogram_files[i]))

		cd_invervention_features.append(dataset_features.get_intervention_features(
			transcription_filename=cd_transcription_files[i],
			max_length=longest_speaker_length))

		cd_compare_features.append(dataset_features.get_compare_features(
			compare_filename=cd_compare_files[i]))

	pause_features = np.concatenate((cc_pause_features, cd_pause_features), axis=0)
	spectrogram_features = np.concatenate((cc_spectogram_features, cd_spectogram_features), axis=0)
	intervention_features = np.concatenate((cc_invervention_features, cd_invervention_features), axis=0)
	compare_features = np.concatenate((cc_compare_features, cd_compare_features), axis=0)

	filenames = np.concatenate((cc_transcription_files, cd_transcription_files), axis=0)

	y_cc = np.zeros((len(cc_pause_features), 2))
	y_cc[:,0] = 1

	y_cd = np.zeros((len(cd_pause_features), 2))
	y_cd[:,1] = 1

	y = np.concatenate((y_cc, y_cd), axis=0).astype(np.float32)

	p = np.random.permutation(len(pause_features))
	pause_features = pause_features[p]
	spectogram_features = spectrogram_features[p]
	intervention_features = intervention_features[p]
	compare_features = compare_features[p]
	y = y[p]
	filenames = filenames[p]

	print('Loading models from {}'.format(model_dir))
	pause_model_files = sorted(glob.glob(os.path.join(model_dir, 'pause/*.h5')))
	pause_models = list(map(lambda x: tf.keras.models.load_model(x), pause_model_files))

	# spec_model_files = sorted(glob.glob(os.path.join(model_dir,'spectogram/*.h5')))
	# spec_models = list(map(lambda x: tf.keras.models.load_model(x), spec_model_files))

	compare_model_files = sorted(glob.glob(os.path.join(model_dir,'compare/*.h5')))
	compare_models = list(map(lambda x: tf.keras.models.load_model(x), compare_model_files))

	intervention_model_files = sorted(glob.glob(os.path.join(model_dir,'intervention/*.h5')))
	intervention_models = list(map(lambda x: tf.keras.models.load_model(x), intervention_model_files))

	fold = 0
	n_split = 5

	# dataset_split = 'no_split'
	dataset_split = 'k_fold'

	voting_type = 'hard_voting'
	# voting_type = 'soft_voting'
	# voting_type = 'learnt_voting'

	print('Voting type {}'.format(voting_type))

	train_accuracies = []
	val_accuracies = []

	if dataset_split == 'no_split':
		pause_probs = pause_models[0].predict(pause_features)
		# spec_probs = spec_models[0].predict(spectogram_features)
		inv_probs = intervention_models[0].predict(intervention_features)
		compare_probs = compare_models[0].predict(compare_features)

		if voting_type=='hard_voting':
				model_predictions = [[np.argmax(pause_probs[i]), np.argmax(spec_probs[i]), np.argmax(inv_probs[i]),  np.argmax(compare_probs[i])] for i in range(len(y_train))]
				voted_predictions = [max(set(i), key = i.count) for i in model_predictions]

		elif voting_type=='soft_voting':
			model_predictions = pause_probs + spec_probs + inv_probs + compare_probs
			# model_predictions = pause_probs + inv_probs
			voted_predictions = np.argmax(model_predictions, axis=-1)

		elif voting_type=='learnt_voting':
			model_predictions = np.concatenate((pause_probs, spec_probs, inv_probs, compare_probs), axis=-1)
			voter = LogisticRegression(C=0.1).fit(model_predictions, np.argmax(y_train, axis=-1))
			# print('Voter coef ', voter.coef_)
			voted_predictions = voter.predict(model_predictions)

			for i in range(len(model_predictions)):
				if voted_predictions[i]!= np.argmax(y, axis=-1)[i]:
					print(filenames[i])

		train_accuracy = accuracy_score(np.argmax(y, axis=-1), voted_predictions)
		train_accuracies.append(train_accuracy)
		print('Fold {} Training Accuracy {:.3f}'.format(fold, train_accuracy))

	elif dataset_split == 'k_fold':
		
		for train_index, val_index in KFold(n_split).split(pause_features):
			pause_train, pause_val = pause_features[train_index], pause_features[val_index]
			spec_train, spec_val = spectogram_features[train_index], spectogram_features[val_index]
			inv_train, inv_val = intervention_features[train_index], intervention_features[val_index]
			compare_train, compare_val = compare_features[train_index], compare_features[val_index]

			sc = StandardScaler()
			sc.fit(compare_train)

			compare_train = sc.transform(compare_train)
			compare_val = sc.transform(compare_val)

			pca = PCA(n_components=compare_feature_size)
			pca.fit(compare_train)

			compare_train = pca.transform(compare_train)
			compare_val = pca.transform(compare_val)

			y_train, y_val = y[train_index], y[val_index]

			filenames_train, filenames_val = filenames[train_index], filenames[val_index]

			pause_probs = pause_models[fold].predict(pause_train)
			# spec_probs = spec_models[fold].predict(spec_train)
			inv_probs = intervention_models[fold].predict(inv_train)
			compare_probs = compare_models[fold].predict(compare_train)

			if voting_type=='hard_voting':
				model_predictions = [[np.argmax(pause_probs[i]), np.argmax(inv_probs[i]),  np.argmax(compare_probs[i])] for i in range(len(y_train))]
				# model_predictions = [[np.argmax(pause_probs[i]), np.argmax(spec_probs[i]), np.argmax(inv_probs[i])] for i in range(len(y_train))]
				voted_predictions = [max(set(i), key = i.count) for i in model_predictions]

			elif voting_type=='soft_voting':
				model_predictions = pause_probs + spec_probs + inv_probs + compare_probs
				# model_predictions = pause_probs + inv_probs
				voted_predictions = np.argmax(model_predictions, axis=-1)

			elif voting_type=='learnt_voting':
				model_predictions = np.concatenate((pause_probs, spec_probs, inv_probs, compare_probs), axis=-1)
				voter = LogisticRegression(C=0.1).fit(model_predictions, np.argmax(y_train, axis=-1))
				# print('Voter coef ', voter.coef_)
				voted_predictions = voter.predict(model_predictions)

			train_accuracy = accuracy_score(np.argmax(y_train, axis=-1), voted_predictions)
			train_accuracies.append(train_accuracy)
			print('Fold {} Training Accuracy {:.3f}'.format(fold, train_accuracy))

			pause_probs = pause_models[fold].predict(pause_val)
			# spec_probs = spec_models[fold].predict(spec_val)
			inv_probs = intervention_models[fold].predict(inv_val)
			compare_probs = compare_models[fold].predict(compare_val)

			if voting_type=='hard_voting':
				model_predictions = [[np.argmax(pause_probs[i]), np.argmax(inv_probs[i]),  np.argmax(compare_probs[i])] for i in range(len(y_train))]
				# model_predictions = [[np.argmax(pause_probs[i]), np.argmax(spec_probs[i]), np.argmax(inv_probs[i])] for i in range(len(y_train))]
				voted_predictions = [max(set(i), key = i.count) for i in model_predictions]

			elif voting_type=='soft_voting':
				model_predictions = pause_probs + spec_probs + inv_probs + compare_probs
				# model_predictions = pause_probs + inv_probs
				voted_predictions = np.argmax(model_predictions, axis=-1)

			elif voting_type=='learnt_voting':
				model_predictions = np.concatenate((pause_probs, spec_probs, inv_probs, compare_probs), axis=-1)
				voter = LogisticRegression(C=0.1).fit(model_predictions, np.argmax(y_train, axis=-1))
				# print('Voter coef ', voter.coef_)
				voted_predictions = voter.predict(model_predictions)
			# continue
			val_accuracy = accuracy_score(np.argmax(y_val, axis=-1), voted_predictions)
			val_accuracies.append(val_accuracy)
			print('Fold {} Val Accuracy {:.3f}'.format(fold, val_accuracy))
			fold+=1


	print('Train accuracy mean: {:.3f}'.format(np.mean(train_accuracies)))
	print('Train accuracy std: {:.3f}'.format(np.std(train_accuracies)))
	if len(val_accuracies) > 0:
		print('Val accuracy mean: {:.3f}'.format(np.mean(val_accuracies)))
		print('Val accuracy std: {:.3f}'.format(np.std(val_accuracies)))

