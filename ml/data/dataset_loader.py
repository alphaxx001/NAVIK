import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class IOVNBDDataset(Dataset):
    def __init__(self, s_data_path, v_data_path, window_size=200):
        """
        Custom Dataset for IO-VNBD
        s_data_path: Path to the smartphone dataset CSV (Features)
        v_data_path: Path to the vehicle dataset CSV (Ground Truth Labels)
        """
        self.window_size = window_size
        
        # 1. Load Data
        print(f"Loading {os.path.basename(s_data_path)} and {os.path.basename(v_data_path)}...")
        self.s_df = pd.read_csv(s_data_path, on_bad_lines='skip', encoding='utf-8', engine='python')
        self.v_df = pd.read_csv(v_data_path, on_bad_lines='skip', encoding='utf-8', engine='python')
        
        # 2. Extract relevant columns (Robust extraction bypassing messy headers)
        cols_s = self.s_df.columns.tolist()
        self.accel_x = self.s_df[[c for c in cols_s if 'ACCELEROMETER X' in c][0]].values
        self.accel_y = self.s_df[[c for c in cols_s if 'ACCELEROMETER Y' in c][0]].values
        self.accel_z = self.s_df[[c for c in cols_s if 'ACCELEROMETER Z' in c][0]].values
        self.gyro_x = self.s_df[[c for c in cols_s if 'GYROSCOPE X' in c][0]].values
        self.gyro_y = self.s_df[[c for c in cols_s if 'GYROSCOPE Y' in c][0]].values
        self.gyro_z = self.s_df[[c for c in cols_s if 'GYROSCOPE Z' in c][0]].values
        
        cols_v = self.v_df.columns.tolist()
        # Using GPS Speed or Wheel Speed as ground truth velocity
        try:
            self.gt_velocity = self.v_df[[c for c in cols_v if 'velocity' in c.lower()][0]].values
        except IndexError:
             # Fallback
            self.gt_velocity = np.zeros_like(self.accel_x) 

        # 3. Synchronize / Align lengths
        min_length = min(len(self.accel_x), len(self.gt_velocity))
        self.features = np.stack([self.accel_x[:min_length], self.accel_y[:min_length], self.accel_z[:min_length],
                                  self.gyro_x[:min_length], self.gyro_y[:min_length], self.gyro_z[:min_length]], axis=1)
        self.labels_vel = self.gt_velocity[:min_length]

    def __len__(self):
        # Number of sliding windows
        return len(self.features) - self.window_size

    def __getitem__(self, idx):
        # Extract a window of features
        x_window = self.features[idx : idx + self.window_size] # Shape: (200, 6)
        
        # The target label is the velocity at the END of the window, or average over the window
        y_vel = self.labels_vel[idx + self.window_size - 1]
        
        # Convert to PyTorch tensors
        # CNN expects (Channels, Sequence_Length), so we transpose
        x_tensor = torch.tensor(x_window.T, dtype=torch.float32) 
        y_vel_tensor = torch.tensor([y_vel], dtype=torch.float32)
        
        return x_tensor, y_vel_tensor

if __name__ == "__main__":
    # Example usage:
    # dataset = IOVNBDDataset("path_to_S_dataset.csv", "path_to_V_dataset.csv")
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print("Dataset loader template ready.")
