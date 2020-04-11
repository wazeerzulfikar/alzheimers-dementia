import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np

# Local imports
import pause_features
import spectrogram_features
import intervention_features

X1, y1 = pause_features.prepare_data

def majority_voting(X1, X2, X3, y, models):
	'''
	X1 is the X of pause_features
	X2 is the X of spectrograms
	X3 is the X of interventions
	y is the labels
	models is a list of strings - names of best models from all 3 kinds of models = [best_model_pause_features_0, best_model_spectrograms_0, best_model_interventions_0, .. , best_model_pause_features_4, best_model_spectrograms_4, best_model_interventions_4]
	
	Returns a list of accuracies over all folds
	'''

	models = list(map(lambda x: tf.keras.models.load_model(x), models))

	fold = 0
	all_accuracies = []
	for _, val_index in KFold(n_split).split(X):
		x_val, y_val = X[val_index], y[val_index]
		probs = list(map(lambda x: x.predict(x_val), models[fold*3, (fold+1)*3]))
		all_predictions = list(map(lambda x: np.argmax(x, axis=-1), probs))
		y_pred = []
		for i in range(all_predictions[0].shape[0]): # iteration over all validation samples
			model_predictions = [all_predictions[0][i], all_predictions[1][i], all_predictions[2][i]]
			predictions, counts = np.unique(np.array(model_predictions), return_counts=True)
			voted_prediction = predictions[np.argmax(counts)]
			y_pred.append(voted_prediction)
		y_pred = np.array(y_pred)
		accuracy = accuracy_score(y_val, y_pred)
		all_accuracies.append(accuracy)
		fold+=1

	return all_accuracies

