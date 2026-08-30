import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as torch_nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ml.models.speednet.avnet_legacy import AVNet
from ml.data.dataset_loader import IOVNBDDataset
import os

def train_model(s_data_path, v_data_path, epochs=10, batch_size=64, lr=0.001):
    print("Initializing Training Pipeline...")
    
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Load Data
    # Note: In a real scenario, you'd load multiple CSVs and split into Train/Val
    dataset = IOVNBDDataset(s_data_path, v_data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 3. Initialize Model, Loss, Optimizer
    model = AVNet().to(device)
    criterion_velocity = torch_nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 4. Training Loop
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        
        for batch_idx, (inputs, targets_vel) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets_vel = targets_vel.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            # Note: We are only training the velocity head for now. 
            # Training the attitude head requires Ground Truth quaternions which requires more complex parsing.
            pred_vel, _ = model(inputs)
            
            # Calculate loss (Velocity only)
            loss = criterion_velocity(pred_vel, targets_vel)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch {batch_idx}/{len(dataloader)} Loss: {loss.item():.4f}")
                
        epoch_loss = running_loss / len(dataloader)
        print(f"--- Epoch {epoch+1} Completed | Average Loss: {epoch_loss:.4f} ---")
        
    # 5. Save the Model
    save_path = "avnet_velocity_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    print("To convert to TFLite for the mobile app, you will need to export this to ONNX first.")

if __name__ == "__main__":
    # Provide dummy paths or real paths to test
    s_csv = r"E:\Shraddha\BITS\Hackathons\SIH\IDR-System\data\raw\IO-VNBD\Synchronised V abd S datasets\Uncategorised IOVNB Dataset\S-Dataset\S-S1.csv"
    v_csv = r"E:\Shraddha\BITS\Hackathons\SIH\IDR-System\data\raw\IO-VNBD\Synchronised V abd S datasets\Uncategorised IOVNB Dataset\V-Dataset\V-S1.csv"
    
    if os.path.exists(s_csv) and os.path.exists(v_csv):
        train_model(s_csv, v_csv, epochs=2)
    else:
        print("CSV files not found. Please check the paths in train.py.")
