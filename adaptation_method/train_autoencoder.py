import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm 
import torch.nn.functional as F
from Inform_project_new.adaptation_method.model_config import Configuration
from Inform_project_new.adaptation_method.EarlyStopping import EarlyStopping

def train_val_encoder(model, optimizer, Loss_func, num_epochs, data_train, data_test, run):
    avg_loss_train = []
    avg_loss_val = []
    best_val_loss = float('inf')
    best_model_path = 'best_autoencoder.pth'
    earlystopping = EarlyStopping(patience= 3)
    stop_epoch = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    #---Training---
    for epoch in range(num_epochs):
        encoded_all = []
        val_encoded_all = []
        
        model.train()
        epoch_loss = 0
        for batch in tqdm(data_train, desc=f'Epoch {epoch+1}/{num_epochs}'):
            print(batch.shape)
            inputs = batch.to(device)
            optimizer.zero_grad()
            
            outputs, encoded = model(inputs)
            loss = Loss_func(outputs, inputs)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            encoded_all.append(encoded.detach().cpu())
            

        train_avg_loss = epoch_loss / len(data_train)
        avg_loss_train.append(train_avg_loss)


        #print(f"Train encodings: min={encoded.min():.4f}, max={encoded.max():.4f}")
        
        # --- Validation ---
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in data_test:
                inputs = batch.to(device)
                val_outputs, val_encoded = model(inputs)
                loss = Loss_func(val_outputs, inputs)
                val_loss += loss.item()
                val_encoded_all.append(val_encoded.detach().cpu())
        
                  
        val_avg_loss = val_loss / len(data_test)
        avg_loss_val.append(val_avg_loss)
        
        #print(f"Val encodings: min={val_encoded.min():.4f}, max={val_encoded.max():.4f}")
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
            
            best_encoded_train = torch.cat(encoded_all, dim = 0)
            best_encoded_val = torch.cat(val_encoded_all, dim = 0)
            
            print(f'Saved new best model at epoch {epoch+1} with val_loss = {val_avg_loss:.4f}')

            
        #Logging Hyperparameters
            run.log({
                'epoch': epoch+1, 
                'train_loss':  train_avg_loss,
                'val_loss': val_avg_loss,
            })

        if best_val_loss == float('inf'):
            best_encoded_train = None
            best_encoded_train = None

    return best_encoded_train, avg_loss_train, best_encoded_val, avg_loss_val, stop_epoch


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
    
    
