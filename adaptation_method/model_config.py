from Inform_project_new.adaptation_method.Autoencoder import ResNetAutoEncoder, Autoencoder
import wandb
def Configuration(model_name = 'Autoencoder'):    
    run = wandb.init(
        project= 'Adapter',
        config = { 
            'model': model_name,
            'batch_size': 32,
            'learning_rate': 1e-5,        
            'epochs': 50,
            'Weight_decay': 1e-4
        }, 
    )
    return run


