import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm 
from Inform_project_new.adaptation_method.EarlyStopping import EarlyStopping

def train_val_encoder(model, optimizer, Loss_func, num_epochs, train_dataloader, test_dataloader, run):
    avg_loss_train = []
    avg_loss_val = []
    best_val_loss = float('inf')
    best_model_path = 'best_autoencoder.pth'
    earlystopping = EarlyStopping(patience= 10)
    stop_epoch = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    all_train_latents = []
    all_val_latents = []
    
    #---Training---
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_latents = []
        
        #for label, train_data, mask in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        for train_data in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            #label = label.to(device)
            train_data = train_data.to(device)
            #mask = mask.to(device)
            optimizer.zero_grad()
            
            outputs, latent = model(train_data)
            loss = Loss_func(outputs, train_data)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_latents.append(latent.detach().cpu())

        train_avg_loss = epoch_loss / len(train_dataloader)
        avg_loss_train.append(train_avg_loss)
        
        #Saving all epoch latents
        all_train_latents.append(torch.cat(epoch_latents, dim = 0).cpu())

        print(f"Train encodings: min={latent.min():.4f}, max={latent.max():.4f}")
        
        # --- Validation ---
        model.eval()
        val_loss = 0
        val_latents = []
        
        with torch.no_grad():
            #for label, test_data, mask in test_dataloader:
            for test_data in test_dataloader:
                #label = label.to(device)
                test_data = test_data.to(device)
                #mask = mask.to(device)
                
                val_outputs, val_latent = model(test_data)
                loss = Loss_func(val_outputs, test_data)
                val_loss += loss.item()
                val_latents.append(val_latent.detach().cpu())
        
                  
        val_avg_loss = val_loss / len(test_dataloader)
        avg_loss_val.append(val_avg_loss)
        
        #Saving validation latents
        all_val_latents.append(torch.cat(val_latents, dim = 0).cpu())
        
        
        print(f"Val latents: min={val_latents.min():.4f}, max={val_latents.max():.4f}")
        
        print(f" Train Loss = {train_avg_loss:.4f} ,Validation Loss = {val_avg_loss:.4f}")
        
        
        #Early stopping
        earlystopping(val_avg_loss)
        if earlystopping.early_stop and stop_epoch is None:
            stop_epoch = epoch
            print(f'Stopping early at epoch {epoch+1}')
            
        #Saving the best model    
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            torch.save(model.state_dict(), best_model_path)
            
            #print(f'Saved new best model at epoch {epoch+1} with val_loss = {val_avg_loss:.4f}')

            
        #Logging Hyperparameters
        run.log({
            'epoch': epoch+1, 
            'train_loss':  train_avg_loss,
            'val_loss': val_avg_loss,
        })

    run.finish()

    return all_train_latents[-1], all_val_latents[-1], avg_loss_train, avg_loss_val, stop_epoch


def plot_loss(num_epochs, avg_loss_train, avg_loss_val, stop_epoch):
    plt.figure(figsize=(12,8))
    
    plt.plot(range(1, num_epochs+1), avg_loss_train, label = 'Training Loss')
    plt.plot(range(1, num_epochs+1), avg_loss_val, label = 'Validation Loss')
    if stop_epoch is not None:
        plt.axvline(x = stop_epoch, color = 'r', linestyle = '--', label = f'Early stop at Epoch{stop_epoch+1}')
        
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation')
    plt.legend()
    plt.grid(True)
    plt.show()
    













