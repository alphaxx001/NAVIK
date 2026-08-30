import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ml.models.heading.heading_net import HeadingNet
from ml.data.dataset_loader import IOVNBDProductionDataset
import matplotlib.pyplot as plt

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    config_path = "configs/heading_training.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    set_seed(config['training']['seed'])
    
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    os.makedirs("ml/evaluation/outputs/heading", exist_ok=True)
    
    device = torch.device("cpu")
    print(f"Training HeadingNet on {device}...")
    
    print("Loading datasets...")
    train_dataset = IOVNBDProductionDataset(split="train", stride_override=config['training']['stride_override_train'], target_type="yaw_rate")
    val_dataset = IOVNBDProductionDataset(split="val", stride_override=config['training']['stride_override_val'], target_type="yaw_rate")
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    m_cfg = config['model']
    model = HeadingNet(
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
    
    target_std = train_dataset.target_std
    
    train_losses = []
    val_losses = []
    
    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_loss = 0.0
        
        for batch_x, batch_y, _ in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds.squeeze(), batch_y.squeeze())
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_x.size(0)
            
        epoch_loss /= len(train_dataset)
        train_losses.append(epoch_loss)
        
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for batch_x, batch_y, _ in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x).squeeze()
                by = batch_y.squeeze()
                
                loss = criterion(preds, by)
                val_loss += loss.item() * batch_x.size(0)
                val_mae += torch.abs(preds - by).sum().item()
                
        val_loss /= len(val_dataset)
        val_losses.append(val_loss)
        
        # Convert val MAE back to physical deg/s for interpretation
        val_mae_normalized = val_mae / len(val_dataset)
        val_mae_physical_rad = val_mae_normalized * target_std
        val_mae_physical_deg = val_mae_physical_rad * (180.0 / 3.14159)
        
        print(f"Epoch {epoch+1}/{config['training']['epochs']} | Train MSE: {epoch_loss:.4f} | Val MSE: {val_loss:.4f} | Val MAE: {val_mae_physical_deg:.2f} deg/s")
        
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
                
    print("Training complete.")
    
    plt.figure()
    plt.plot(train_losses, label="Train Loss (MSE)")
    plt.plot(val_losses, label="Val Loss (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Normalized MSE Loss")
    plt.legend()
    plt.title("HeadingNet Training Curve")
    plt.savefig("ml/evaluation/outputs/heading/training_curve.png")
    plt.close()

if __name__ == "__main__":
    main()
