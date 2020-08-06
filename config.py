import utils

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