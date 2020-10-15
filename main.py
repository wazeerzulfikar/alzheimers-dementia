import dataset
import ensemble_trainer
import evaluator
import utils

from config import config

def main(config):

	# Create save directories
	utils.create_directories(config)

	# Prepare and load the data
	if 'silences' in config.model_types: 	
		data = dataset.prepare_data_new(config.dataset_dir, config)
	else:
		data = dataset.prepare_data(config.dataset_dir, config)
	# print(data)
	# return
	# Train the ensemble models
	if config.training_type == 'bagging':
		ensemble_trainer.bagging_ensemble_training(data, config)
	elif config.training_type == 'boosting':		
		ensemble_trainer.boosted_ensemble_training(data, config)

	# Evaluate the model
	if 'silences' not in config.model_types: 	
		test_data = dataset.prepare_test_data(config.test_dataset_dir, config)
		evaluator.evaluate(data, test_data, config)

if __name__ == '__main__':
	main(config)
