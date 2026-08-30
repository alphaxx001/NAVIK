import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ml.models.motion_state.motion_state_net import MotionStateNet
from ml.data.dataset_loader import IOVNBDProductionDataset
import matplotlib.pyplot as plt

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    config_path = "configs/motion_state_training.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    set_seed(config['training']['seed'])
    
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    os.makedirs("ml/evaluation/outputs/motion_state", exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training MotionStateNet on {device}...")
    
    print("Loading datasets...")
    train_dataset = IOVNBDProductionDataset(split="train", stride_override=config['training']['stride_override_train'])
    val_dataset = IOVNBDProductionDataset(split="val", stride_override=config['training']['stride_override_val'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    m_cfg = config['model']
    model = MotionStateNet(
        in_channels=m_cfg['in_channels'],
        cnn_channels=m_cfg['cnn_channels'],
        kernel_size=m_cfg['kernel_size'],
        gru_hidden=m_cfg['gru_hidden'],
        gru_layers=m_cfg['gru_layers']
    ).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Needs to un-normalize target velocity to check threshold
    target_std = train_dataset.target_std
    target_mean = train_dataset.target_mean
    stat_thresh = config['training']['stationary_threshold']
    
    train_losses = []
    val_losses = []
    
    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_loss = 0.0
        
        for batch_x, batch_y, _ in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Create classification target: 1 if stationary, 0 if moving
            y_real = (batch_y * target_std) + target_mean
            batch_y_class = (y_real < stat_thresh).float()
            
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y_class)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_x.size(0)
            
        epoch_loss /= len(train_dataset)
        train_losses.append(epoch_loss)
        
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for batch_x, batch_y, _ in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                y_real = (batch_y * target_std) + target_mean
                batch_y_class = (y_real < stat_thresh).float()
                
                preds = model(batch_x)
                loss = criterion(preds, batch_y_class)
                val_loss += loss.item() * batch_x.size(0)
                
                preds_class = (preds > 0.5).float()
                correct += (preds_class == batch_y_class).sum().item()
                
        val_loss /= len(val_dataset)
        val_losses.append(val_loss)
        val_acc = correct / len(val_dataset)
        
        print(f"Epoch {epoch+1}/{config['training']['epochs']} | Train BCE: {epoch_loss:.4f} | Val BCE: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
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
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.legend()
    plt.title("MotionStateNet Training Curve")
    plt.savefig("ml/evaluation/outputs/motion_state/training_curve.png")
    plt.close()

if __name__ == "__main__":
    main()
