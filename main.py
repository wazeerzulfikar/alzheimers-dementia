import ensemble_trainer
import ensemble_infer

def main():
	dataset_dir = '../ADReSS-IS2020-data/train'
	# model_dir = 'boosted_models_val_acc'
	model_dir = 'bagging_models_val_acc_1'
	model_dir = 'bagging_models_val_acc_1'
	# model_dir = 'boosted_models_val_acc_1'
	model_dir = 'boosted_models_1'
	n_splits = 5

	# results = ensemble_trainer.bagging_ensemble_training(dataset_dir, model_dir, n_splits)
	results = ensemble_trainer.boosted_ensemble_training(dataset_dir, model_dir, n_splits)
	# print(results)

	ensemble_infer.infer(dataset_dir, model_dir)

if __name__ == '__main__':
	main()