import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import *

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import glob, os, math, time
from decimal import *
import numpy as np
np.random.seed(0)
from pathlib import Path
import seaborn as sns

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import dataset

tfd = tfp.distributions

def create_pause_model():
	model = tf.keras.Sequential()
	model.add(Input(shape=(11,)))
	model.add(BatchNormalization())
	model.add(Dense(16, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(16, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Dense(32, activation='relu'))
	model.add(Dense(16, activation='relu'))
	model.add(Dense(8, activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(2, kernel_regularizer=tf.keras.regularizers.l2(0.01), activity_regularizer=tf.keras.regularizers.l1(0.1)))
	model.add(tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :1], scale=tf.math.softplus(t[..., 1:])+1e-6)))
	return model

def create_intervention_model():
	model = tf.keras.Sequential()
	model.add(Input((32,3)))
	model.add(LSTM(12))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))
	model.add(Dense(16, activation='relu'))
	model.add(Dense(16, activation='relu'))
	model.add(Dense(2, kernel_regularizer=tf.keras.regularizers.l2(0.01), activity_regularizer=tf.keras.regularizers.l1(0.1)))
	model.add(tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :1], scale=tf.math.softplus(t[..., 1:])+1e-6)))
	return model

def train_a_fold(model_type, x_train, y_train, x_val, y_val, model_dir, fold=1):

	if model_type == 'pause':
		model = create_pause_model()
		lr = 0.01
		epochs = 5000

	elif model_type == 'intervention':
		model = create_intervention_model()
		lr = 0.001
		epochs = 2000

	negloglik = lambda y, p_y: -p_y.log_prob(y)
	try:
		for i in os.path.join(model_dir, model_type, '*.h5'):		os.remove(i)
	except:
		pass		
	checkpoint_filepath = os.path.join(model_dir, model_type, '{epoch:d}-{val_loss:.3f}.h5')
	checkpointer = tf.keras.callbacks.ModelCheckpoint(
			checkpoint_filepath, monitor='val_loss', verbose=0, save_best_only=True,
			save_weights_only=True, mode='auto', save_freq='epoch')

	model.compile(optimizer=tf.optimizers.Adam(learning_rate=lr),
				  loss=negloglik)

	hist = model.fit(x_train, y_train,
					batch_size=8,
					epochs=epochs,
					verbose=1,
					callbacks=[checkpointer],
					validation_data=(x_val, y_val))
	# print(model.summary())
	epoch_val_losses = hist.history['val_loss']
	best_epoch_val_loss, best_epoch = np.min(epoch_val_losses), np.argmin(epoch_val_losses)+1
	best_epoch_train_loss = hist.history['loss'][best_epoch-1]
	checkpoint_filepath = os.path.join(model_dir, model_type, '{:d}-{:.3f}.h5'.format(best_epoch, best_epoch_val_loss))
	model.load_weights(checkpoint_filepath)

	return model, best_epoch_train_loss, best_epoch_val_loss, best_epoch

data = dataset.prepare_data('../ADReSS-IS2020-data/train')
X_intervention, X_pause, X_spec, X_compare, X_reg_intervention, X_reg_pause, X_reg_compare = data[0:7]
y, y_reg, filenames_intervention, filenames_pause, filenames_spec, filenames_compare = data[7:]

feature_types = {
	'intervention': X_reg_intervention,
	'pause': X_reg_pause,
	'compare': X_reg_compare
}

results = {}


model_types = ['pause'] # from scratch running in tmux 2
model_types = ['intervention']

model_dir = 'uncertainty'
model_dir = Path(model_dir)
model_dir.joinpath(model_types[0]).mkdir(parents=True, exist_ok=True)
model_dir = str(model_dir)

model_type = model_types[0]
x, y = feature_types[model_type], y_reg
n_samples = x.shape[0]


n_models = 10
mus, sigmas = [], []
train_nlls, val_nlls = [], []
start = time.time()
for model in range(n_models):
	print('~ Training model {} ~'.format(model+1))
	# p = np.random.permutation(n_samples) # mus are corresponding to different samples, cant mean it out
	# x, y = x[p], y[p]

	x_train, x_val = x[int(0.2*n_samples):], x[:int(0.2*n_samples)]
	y_train, y_val = y[int(0.2*n_samples):], y[:int(0.2*n_samples)]
	n_val_samples = x_val.shape[0]

	model, best_epoch_train_loss, best_epoch_val_loss, best_epoch = train_a_fold(model_type, x_train, y_train, x_val, y_val, model_dir)
	train_nll, val_nll = best_epoch_train_loss, best_epoch_val_loss

	y_val = y_val.reshape(-1,1)
	pred = model(x_val)
	
	mu = pred.mean()
	sigma = pred.stddev()

	# preds.append(pred)
	mus.append(mu.numpy())
	sigmas.append(sigma.numpy())
	train_nlls.append(train_nll)
	val_nlls.append(val_nll)

	val_rmse = mean_squared_error(y_val, mu, squared=False)

	print()
	print(model_type)
	print('Best Epoch: {:d}'.format(best_epoch))
	print('Train NLL: {:.3f}'.format(best_epoch_train_loss)) 
	print('Val NLL: {:.3f}'.format(best_epoch_val_loss)) # NLL
	print('Val RMSE: {:.3f}'.format(val_rmse))

	pred_probs = pred.prob(mu)
	true_y_probs = pred.prob(y_val)
	print()
	for i in range(n_val_samples):
		tf.print('Pred: {:.3f}'.format(mu[i][0]), '\t\t\tProb: {:.7f}'.format(pred_probs[i][0]), '\t\t\tTrue: {}'.format(y_val[i][0]), '\t\t\tProb: {:.7f}'.format(true_y_probs[i][0]), '\t\t\tStd Dev: {:.7f}'.format(sigma[i][0]), '\t\t\tEntropy: {:.7f}'.format(pred.entropy()[i][0]))
	print()
	print(model_type)

mus, sigmas = np.concatenate(mus, axis=-1), np.concatenate(sigmas, axis=-1)
ensemble_mu = np.mean(mus, axis=-1).reshape(-1,1)
ensemble_sigma = np.sqrt(np.mean(np.square(sigmas) + np.square(mus), axis=-1).reshape(-1,1) - np.square(ensemble_mu))

ensemble_dist = tfd.Normal(loc=ensemble_mu, scale=ensemble_sigma)
ensemble_pred_probs = ensemble_dist.prob(ensemble_mu).numpy()
ensemble_pred_log_probs = ensemble_dist.log_prob(ensemble_mu).numpy()
ensemble_true_log_probs = ensemble_dist.log_prob(y_val).numpy()
ensemble_nll = np.mean(-ensemble_true_log_probs)

print()
print('Deep ensemble results')
for i in range(n_val_samples):
	print('Pred: {:.3f}'.format(ensemble_mu[i][0]), '\t\t\tProb: {:.7f}'.format(ensemble_pred_probs[i][0]), '\t\t\tTrue: {}'.format(y_val[i][0]), '\t\t\tProb: {:.7f}'.format(ensemble_dist.prob(y_val)[i][0]), '\t\t\tStd Dev: {:.7f}'.format(ensemble_sigma[i][0]), '\t\t\tEntropy: {:.7f}'.format(ensemble_dist.entropy()[i][0]))
print()
print('Train NLLs: ', train_nlls)
print('Val NLLs: ',val_nlls)
print('Train NLL: {:.3f} +/- {:.3f}'.format(np.mean(train_nlls), np.std(train_nlls)))
print('Val NLL: {:.3f} +/- {:.3f}'.format(np.mean(val_nlls), np.std(val_nlls)))
print('Ensemble Val NLL calculated using ensemble distribution: {:.3f}'.format(ensemble_nll))
print('Deep ensemble of {} models took {} minutes'.format(n_models, int((time.time()-start)/60)))

min_prob = float(Decimal(str(np.min(ensemble_pred_probs))).quantize(Decimal('.001'), rounding=ROUND_DOWN))
max_prob = float(Decimal(str(np.max(ensemble_pred_probs))).quantize(Decimal('.001'), rounding=ROUND_DOWN))
bins = np.arange(min_prob, max_prob, 1e-3)

binned_rmses = []
for b in bins:
	conditioned_preds = ensemble_mu[ensemble_pred_probs>b]
	conditioned_true = y_val[ensemble_pred_probs>b]
	bin_rmse = mean_squared_error(conditioned_true, conditioned_preds, squared=False)
	binned_rmses.append(bin_rmse)

print(min_prob, max_prob)
print(bins)
print(binned_rmses)

