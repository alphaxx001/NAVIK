import os
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class IOVNBDProductionDataset(Dataset):
    """
    Production PyTorch Dataset for IO-VNBD.
    Reads manifesting and normalization from 'data/manifests/'.
    """
    def __init__(self, config_path="configs/data_pipeline.json", split="train"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        with open("data/manifests/normalization.json", 'r') as f:
            self.norm = json.load(f)
            
        split_file = f"data/manifests/{split}_sessions.json"
        with open(split_file, 'r') as f:
            self.sessions = json.load(f)
            
        self.inventory = pd.read_csv("data/manifests/session_inventory.csv")
        self.inventory = self.inventory[self.inventory['session_id'].isin(self.sessions)]
        
        self.window_size = self.config['windowing']['sample_count']
        self.stride = self.config['windowing']['stride']
        
        self.features_mean = np.array(self.norm['features']['mean'])
        self.features_std = np.array(self.norm['features']['std'])
        self.target_mean = self.norm['targets']['mean']
        self.target_std = self.norm['targets']['std']
        
        self.windows = []
        self._load_data()
        
    def _load_data(self):
        # Iterate over all assigned sessions and extract windows
        for _, row in self.inventory.iterrows():
            s_path = row['s_file']
            v_path = row['v_file']
            session_id = row['session_id']
            
            s_df = pd.read_csv(s_path, encoding=self.config['processing']['encoding'], on_bad_lines='skip')
            v_df = pd.read_csv(v_path, encoding=self.config['processing']['encoding'], on_bad_lines='skip')
            
            # Clean headers
            s_df.columns = [c.strip() for c in s_df.columns]
            v_df.columns = [c.strip() for c in v_df.columns]
            
            # Find velocity column
            vel_cols = [c for c in v_df.columns if "velocity" in c.lower() or "speed" in c.lower()]
            vel_col = None
            for c in vel_cols:
                if "km/hr" in c.lower() and "velocity" in c.lower():
                    vel_col = c
                    break
            if not vel_col and vel_cols: vel_col = vel_cols[0]
            
            # Get data arrays
            X_raw = s_df[self.config['processing']['features']].interpolate(method='linear').bfill().ffill().values
            Y_raw = (v_df[vel_col] / 3.6).interpolate(method='linear').bfill().ffill().values
            
            # Normalize
            X_norm = (X_raw - self.features_mean) / self.features_std
            Y_norm = (Y_raw - self.target_mean) / self.target_std
            
            # Generate overlapping windows
            for i in range(0, len(X_norm) - self.window_size + 1, self.stride):
                x_window = X_norm[i:i + self.window_size]
                y_target = Y_norm[i + self.window_size - 1] # End of window prediction
                
                self.windows.append({
                    "x": torch.tensor(x_window, dtype=torch.float32).transpose(0, 1), # Shape: [channels, window]
                    "y": torch.tensor([y_target], dtype=torch.float32),
                    "session_id": session_id,
                    "index": i
                })
                
    def __len__(self):
        return len(self.windows)
        
    def __getitem__(self, idx):
        item = self.windows[idx]
        return item['x'], item['y'], item['session_id']
