from Inform_project_new.adaptation_method.Autoencoder import ResNetAutoEncoder, Autoencoder
import wandb
def Configuration():    
    run = wandb.init(
        project= 'Adapter',
        config = { 
            'batch_size': 32,
            'learning_rate': 3e-4,        
            'epochs': 50,
            'Weight_decay': 1e-4
        }, 
    )
    return run
