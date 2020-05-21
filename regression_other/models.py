import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Adam, SGD
# from tensorflow.keras.losses import mean_absolute_error, mean_squared_error
import tensorflow_probability as tfp


def build_model(config):

	if config.build_model=='point':
		loss = tf.keras.losses.mean_squared_error
		model = Sequential()
		model.add(Dense(50, activation='relu'))
		model.add(Dense(1))

	if config.build_model=='gaussian':
		tfd = tfp.distributions
		loss = lambda y, p_y: -p_y.log_prob(y)
		model = Sequential()
		model.add(Dense(50, activation='relu', dtype='float64'))
		model.add(Dense(2, dtype='float64'))
		model.add(tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :1], scale=tf.math.softplus(t[..., 1:])+1e-6), dtype='float64'))
	
	return model, loss