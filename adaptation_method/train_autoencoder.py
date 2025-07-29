import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm 
import torch.nn.functional as F
from model_config import Configuration
from EarlyStopping import EarlyStopping

def train_val_encoder(model, optimizer, Loss_func, num_epochs, dataloader_train, dataloader_test, run):
    avg_loss_train = []
    avg_loss_val = []
    
    best_val_loss = float('inf')
    best_model_path = 'best_autoencoder.pth'
    earlystopping = EarlyStopping(patience= 3)
    stop_epoch = None
    
    #---Training---
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch in tqdm(dataloader_train, desc=f'Epoch {epoch+1}/{num_epochs}'):
            print(batch[0])
            inputs = batch[0]
            optimizer.zero_grad()
            
            outputs, encoded = model(inputs)
            loss = Loss_func(outputs, inputs)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        train_avg_loss = epoch_loss / len(dataloader_train)
        avg_loss_train.append(train_avg_loss)


        #print(f"Train encodings: min={encoded.min():.4f}, max={encoded.max():.4f}")
        #print(f"Train Loss = {train_avg_loss:.4f}")
        
        # --- Validation ---
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in dataloader_test:
                inputs = batch[0].unsqueeze(0)
                val_outputs, val_encoded = model(inputs)
                loss = Loss_func(val_outputs, inputs)
                val_loss += loss.item()
        
                  
        val_avg_loss = val_loss / len(dataloader_test)
        avg_loss_val.append(val_avg_loss)
        
        #print(f"Val encodings: min={val_encoded.min():.4f}, max={val_encoded.max():.4f}")
        print(f" Train Loss = {train_avg_loss:.4f} ,Validation Loss = {val_avg_loss:.4f}")
        
        #Early stopping
        earlystopping(val_avg_loss)
        if earlystopping.early_stop and stop_epoch is None:
            stop_epoch = epoch
            print(f'Stopping early at epoch {epoch+1}')
            
    #Saving the best model    
    if val_avg_loss > best_val_loss:
        best_val_loss = val_avg_loss
        torch.save(model.state_dict(), best_model_path)
        
        print(f'Saved new best model at epoch {epoch+1} with val_loss = {val_avg_loss:.4f}')
    
    #Logging Hyperparameters
        run.log({
            'epoch': epoch+1, 
            'train_loss':  train_avg_loss,
            'val_loss': val_avg_loss,
        })

    return encoded, avg_loss_train, val_encoded, avg_loss_val, stop_epoch


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
    
    
