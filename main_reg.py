from pathlib import Path

import ensemble_trainer_reg
import evaluator_reg

def main():
	dataset_dir = '../ADReSS-IS2020-data/train'

	# model_dir = 'results/bagging_i_p_c'
	model_dir = 'results/bagging_val_loss'
	# model_dir = 'results/bagging_i_p_c_val_acc'
	# model_dir = 'results/bagging_i_p_s_val_acc'

	# model_dir = 'results/boosting_i_p_c_val_loss'
	# model_dir = 'results/boosting_i_p_s_val_loss'
	# model_dir = 'results/boosting_i_p_c_val_acc'
	# model_dir = 'results/boosting_i_p_s_val_acc'

	# model_dir = 'temp'

	model_dir = Path(model_dir)
	submodel_dirs = ['intervention', 'pause', 'spectogram', 'compare']
	for m in submodel_dirs:
		model_dir.joinpath(m+'/reg').mkdir(parents=True, exist_ok=True)

	model_dir = str(model_dir)

	model_types = ['intervention', 'pause', 'compare']
	# model_types = ['intervention', 'pause', 'spectogram']

	n_splits = 5

	results = ensemble_trainer_reg.bagging_ensemble_training(dataset_dir, model_dir, model_types, n_splits)

	# Training order is same as model_types
	# results = ensemble_trainer.boosted_ensemble_training(dataset_dir, model_dir, model_types, n_splits)

	# dataset_split = 'full_dataset'
	dataset_split = 'k_fold'

	# voting_type = 'hard_voting'
	voting_type = 'soft_voting'
	# voting_type = 'learnt_voting'

	evaluator_reg.evaluate(dataset_dir, model_dir, model_types, voting_type=voting_type, dataset_split=dataset_split)

if __name__ == '__main__':
	main()