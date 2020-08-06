import dataset
import ensemble_trainer
import evaluator
import test_writer
import utils

from config import config

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