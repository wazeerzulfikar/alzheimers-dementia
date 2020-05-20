# import dataset
# import ensemble_trainer
# import evaluator
# import test_writer
# from  import utils
from load_dataset import load_dataset
from pathlib import Path
from evaluator import evaluate_n

# Config to choose the hyperparameters for everything
class EasyDict(dict):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name] 

def create_directories(config):
    model_dir = Path(config.model_dir)
    model_dir.joinpath(config.dataset).mkdir(parents=True, exist_ok=True)        

config = EasyDict({
    # 'task': 'classification',
    'task': 'regression',

    'dataset_dir': './datasets/',
    # 'dataset_dir': '../ADReSS-IS2020-data/train',

    'dataset': 'boston',
    'model_dir': 'models/',
    'model_types': ['intervention', 'pause', 'compare'],

    # 'training_type': 'bagging',
    # 'training_type' :'boosting',

    'n_folds': 20,

    # 'dataset_split' :'full_dataset',
    'dataset_split' :'k_fold',
    
    'mod_split' :'none',
#     'mod_split' :'human',
    
    'learning_rate' : 0.1,
    
    'loss' : 'mse',
    
    'optimizer' : 'adams',
})

def main(config):

    if config.dataset=='protein':
        config.n_folds = 5

    # Create save directories
#     create_directories(config)

    data = load_dataset(config)
    
    evaluate_n(config, data)

    # # Train the ensemble models
    # if config.training_type == 'bagging':
    # 	ensemble_trainer.bagging_ensemble_training(data, config)
    # elif config.training_type == 'boosting':		
    # 	ensemble_trainer.boosted_ensemble_training(data, config)

    # # Evaluate the model
    # evaluator.evaluate(data, config)

    # # Write out test results (todo)
    # # test_writer.test(test_filename, dataset_dir, test_dataset_dir, model_dir, model_types, voting_type=voting_type, select_fold=None)

if __name__ == '__main__':
    main(config)