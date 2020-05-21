import os
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

    model_dir = Path(os.path.join(config.model_dir, config.dataset))
    model_dir.joinpath(config.expt_name).mkdir(parents=True, exist_ok=True)        
    # print(os.listdir(os.path.join(config.model_dir, config.dataset)))
config = EasyDict({
    # 'task': 'classification',
    # 'task': 'regression',

    # 'dataset_dir': './datasets/',
    # 'dataset_dir': '../ADReSS-IS2020-data/train',

    'dataset': 'boston',
    'model_dir': 'models/',
    # 'model_types': ['intervention', 'pause', 'compare'],

    # 'training_type': 'bagging',
    # 'training_type' :'boosting',

    'n_folds': 20,
    'n_models': 5, # models in deep ensemble

    # 'dataset_split' :'full_dataset',
    # 'dataset_split' :'k_fold',

    'build_model': 'gaussian',
    # 'build_model': 'point',

    
    # 'mod_split' :'none',
    # 'mod_split' :'human',
    'mod_split' :'computation_split',
    
    'learning_rate' : 0.1,

    'epochs' : 200,
    
    'loss' : 'mse',
    
    'optimizer' : 'adam',

    'batch_size' : 100,
})

config.expt_name = config.build_model + "_" + config.optimizer + str(config.learning_rate) + "_bs" + str(config.batch_size) + "_epochs" + str(config.epochs)

def main(config):

    if config.dataset=='protein':
        config.n_folds = 5

    # Create save directories
    create_directories(config)

    data = load_dataset(config)
    # print(data['X1'].shape)
    # print(data['X2'].shape)
    evaluate_n(config, data)

    # # Train the ensemble models
    # if config.training_type == 'bagging':
    #   ensemble_trainer.bagging_ensemble_training(data, config)
    # elif config.training_type == 'boosting':      
    #   ensemble_trainer.boosted_ensemble_training(data, config)

    # # Evaluate the model
    # evaluator.evaluate(data, config)

    # # Write out test results (todo)
    # # test_writer.test(test_filename, dataset_dir, test_dataset_dir, model_dir, model_types, voting_type=voting_type, select_fold=None)

if __name__ == '__main__':
    main(config)