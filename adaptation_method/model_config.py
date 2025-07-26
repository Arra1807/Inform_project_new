import wandb

run = wandb.init(
    project= 'Adapter', 
    config = {
        'model': 'ResNetAutoEncoder', 
        'batch_size': 64,
        'learning_rate': 1e-3,        
        'epochs': 20,
    }, 
)

