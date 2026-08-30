import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ml.models.speednet.speednet import SpeedNet
from ml.data.dataset_loader import IOVNBDProductionDataset
import matplotlib.pyplot as plt

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    config_path = "configs/speednet_training.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    set_seed(config['training']['seed'])
    
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    os.makedirs("ml/evaluation/outputs/speednet", exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training SpeedNet on {device}...")
    
    # Load Datasets
    print("Loading datasets...")
    train_dataset = IOVNBDProductionDataset(split="train", stride_override=config['training']['stride_override_train'])
    val_dataset = IOVNBDProductionDataset(split="val", stride_override=config['training']['stride_override_val'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    
    # Init Model
    m_cfg = config['model']
    model = SpeedNet(
        in_channels=m_cfg['in_channels'],
        cnn_channels=m_cfg['cnn_channels'],
        kernel_size=m_cfg['kernel_size'],
        gru_hidden=m_cfg['gru_hidden'],
        gru_layers=m_cfg['gru_layers']
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Denormalization stats
    target_std = train_dataset.target_std
    target_mean = train_dataset.target_mean
    
    train_losses = []
    val_losses = []
    
    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_loss = 0.0
        
        for batch_x, batch_y, _ in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_x.size(0)
            
        epoch_loss /= len(train_dataset)
        train_losses.append(epoch_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for batch_x, batch_y, _ in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                
                # Calculate MAE in real units (m/s)
                preds_real = preds * target_std + target_mean
                y_real = batch_y * target_std + target_mean
                val_mae += torch.abs(preds_real - y_real).sum().item()
                
        val_loss /= len(val_dataset)
        val_losses.append(val_loss)
        val_mae /= len(val_dataset)
        val_rmse = (val_loss**0.5) * target_std
        
        print(f"Epoch {epoch+1}/{config['training']['epochs']} | Train MSE: {epoch_loss:.4f} | Val MSE: {val_loss:.4f} | Val RMSE: {val_rmse:.2f} m/s | Val MAE: {val_mae:.2f} m/s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), config['paths']['best_model'])
            print("  --> Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= config['training'].get('patience', 10):
                print(f"Early stopping at epoch {epoch+1}")
                break
            
    print("Training complete. Best model saved.")
    
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (Normalized)")
    plt.legend()
    plt.title("SpeedNet Training Curve")
    plt.savefig("ml/evaluation/outputs/speednet/training_curve.png")
    plt.close()

if __name__ == "__main__":
    main()
