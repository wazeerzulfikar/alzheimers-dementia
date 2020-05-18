import dataset
import ensemble_trainer
import evaluator
import test_writer
import utils

# Config to choose the hyperparameters for everything

config = utils.EasyDict({
	'task': 'classification',
	# 'task': 'regression',

	# 'dataset_dir': '../DementiaBank'
	'dataset_dir': '../ADReSS-IS2020-data/train',

	'model_dir': 'models/bagging',
	'model_types': ['intervention', 'pause', 'compare'],

	'training_type': 'bagging',
	# 'training_type' :'boosting',

	'n_folds': 5,

	# 'dataset_split' :'full_dataset',
	'dataset_split' :'k_fold',

	'voting_type': 'hard_voting',
	# 'voting_type': 'soft_voting',
	# 'voting_type': 'learnt_voting',


	'longest_speaker_length': 32,
	'n_pause_features': 11,
	'compare_features_size': 21,
	'split_reference': 'samples'
})

def main(config):

	# Create save directories
	utils.create_directories(config)

	# Prepare and load the data
	data = dataset.prepare_data(config)

	# Train the ensemble models
	if config.training_type == 'bagging':
		ensemble_trainer.bagging_ensemble_training(data, config)
	elif config.training_type == 'boosting':		
		ensemble_trainer.boosted_ensemble_training(data, config)

	# Evaluate the model
	evaluator.evaluate(data, config)

	# Write out test results (todo)
	# test_writer.test(test_filename, dataset_dir, test_dataset_dir, model_dir, model_types, voting_type=voting_type, select_fold=None)

if __name__ == '__main__':
	main(config)