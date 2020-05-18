from pathlib import Path

import ensemble_trainer_reg
import evaluator_reg
import test_writer_reg

def main():
	# dataset_dir = '../DementiaBank'
	dataset_dir = '../ADReSS-IS2020-data/train'
	model_dir = 'models/bagging_reg'

	training_type = 'bagging'
	# training_type = 'boosting'

	n_splits = 5

	# dataset_split = 'full_dataset'
	dataset_split = 'k_fold'

	voting_type = 'hard_voting'
	# voting_type = 'soft_voting'
	# voting_type = 'learnt_voting'

	model_types = ['intervention', 'pause', 'compare']

	# Create save directories
	model_dir = Path(model_dir)
	for m in model_types:
		model_dir.joinpath(m).mkdir(parents=True, exist_ok=True)
	model_dir = str(model_dir)


	if training_type == 'bagging':
		ensemble_trainer_reg.bagging_ensemble_training(dataset_dir, model_dir, model_types, n_splits)

	elif training_type == 'boosting':
		# Training order is same as model_types
		ensemble_trainer_reg.boosted_ensemble_training(dataset_dir, model_dir, model_types, n_splits)

	evaluator_reg.evaluate(dataset_dir, model_dir, model_types, voting_type=voting_type, dataset_split=dataset_split, n_split=n_splits)

	# test_dataset_dir = '../ADReSS-IS2020-data/test'
	# test_filename = '../submissions_scratch/35.txt'
	# test_writer_reg.test(test_filename, dataset_dir, test_dataset_dir, model_dir, model_types, voting_type=voting_type, select_fold=None)

if __name__ == '__main__':
	main()