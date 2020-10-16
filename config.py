import utils

config = utils.EasyDict({
	'task': 'classification',
	# 'task': 'regression',

	'uncertainty': False,

	# 'dataset_dir': '../DementiaBank'
	'dataset_dir': '../DementiaBank/0extra/ADReSS-IS2020-train/ADReSS-IS2020-data/train/',
	'test_dataset_dir': '../../alzheimers/ADReSS-IS2020-data/test',

	# 'model_dir': 'models/uncertainty_individual/1',
	# 'model_types': ['compare'],

	# 'model_dir': 'models/uncertainty_boosting_rmse/1',
	'model_dir': 'models/testing_silence/3_c',

	# 'model_types': [ 'compare', 'pause', 'intervention'],
	'model_types': [ 'silences'],

	'training_type': 'bagging',
	# 'training_type' :'boosting',

	'boosting_type': 'rmse',
	# 'boosting_type': 'stddev',


	'n_folds': 5,

	'dataset_split' :'full_dataset',
	# 'dataset_split' :'k_fold',
	'split_ratio': 0.8,

	'voting_type': 'hard_voting',
	# 'voting_type': 'soft_voting',
	# 'voting_type': 'learnt_voting',
	# 'voting_type': 'uncertainty_voting',


	'longest_speaker_length': 32,
	'n_pause_features': 11,
	'compare_features_size': 21,
	'split_reference': 'samples',

	'n_epochs': 2000,
	'batch_size': 24,
	'lr': 0.001,
	'verbose': 0
})
