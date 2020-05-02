from pathlib import Path

import ensemble_trainer
import ensemble_infer

def main():
	dataset_dir = '../ADReSS-IS2020-data/train'

	# model_dir = 'results/bagging_i_p_c'
	# model_dir = 'results/bagging_i_p_s'
	# model_dir = 'results/bagging_i_p_c_val_acc'
	model_dir = 'results/bagging_i_p_s_val_acc'

	# model_dir = 'results/boosting_i_p_c'
	# model_dir = 'results/boosting_i_p_s'
	# model_dir = 'results/boosting_i_p_c_val_acc'
	# model_dir = 'results/boosting_i_p_s_val_acc'

	model_dir = Path(model_dir)
	submodel_dirs = ['intervention', 'pause', 'spectogram', 'compare']
	for m in submodel_dirs:
		model_dir.joinpath(m).mkdir(parents=True, exist_ok=True)

	model_dir = str(model_dir)

	n_splits = 5

	results = ensemble_trainer.bagging_ensemble_training(dataset_dir, model_dir, n_splits)
	# results = ensemble_trainer.boosted_ensemble_training(dataset_dir, model_dir, n_splits)
	# print(results)

	ensemble_infer.infer(dataset_dir, model_dir)

if __name__ == '__main__':
	main()