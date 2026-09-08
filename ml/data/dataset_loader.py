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
    def __init__(self, config_path="configs/data_pipeline.json", split="train", stride_override=None, target_type='velocity'):
        self.target_type = target_type
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
        self.stride = stride_override if stride_override is not None else self.config['windowing']['stride']
        
        self.features_mean = np.array(self.norm['features']['mean'])
        self.features_std = np.array(self.norm['features']['std'])
        
        if self.target_type == 'velocity':
            self.target_mean = self.norm['targets']['mean']
            self.target_std = self.norm['targets']['std']
        elif self.target_type == 'yaw_rate':
            # Yaw rate is naturally zero-centered. We can use std=0.2 (approx 11 deg/s) or just 1.0.
            self.target_mean = 0.0
            self.target_std = 0.2
        else:
            self.target_mean = 0.0
            self.target_std = 1.0
            
        self.windows = []
        self._load_data()
        
    def _load_data(self):
        demo_file = "data/demo/replay_session.csv"
        # If none of the inventory session files exist on disk, fall back to demo session
        has_any_file = any(os.path.exists(row['s_file']) for _, row in self.inventory.iterrows())
        
        if not has_any_file:
            if not os.path.exists(demo_file):
                return
            df = pd.read_csv(demo_file)
            X_raw = df[['ax', 'ay', 'az', 'gx', 'gy', 'gz']].interpolate(method='linear').bfill().ffill().values
            
            # Approximate target from consecutive GNSS or zero
            if self.target_type == 'velocity':
                target_vals = np.zeros(len(df))
                for i in range(1, len(df)):
                    # Dist in meters
                    dlat = np.radians(df['gnss_lat'].iloc[i] - df['gnss_lat'].iloc[i-1]) * 6378137.0
                    dlon = np.radians(df['gnss_lon'].iloc[i] - df['gnss_lon'].iloc[i-1]) * 6378137.0 * np.cos(np.radians(52.5))
                    dt = max(0.01, df['timestamp'].iloc[i] - df['timestamp'].iloc[i-1])
                    target_vals[i] = np.hypot(dlat, dlon) / dt
                target_vals[0] = target_vals[1]
            else:
                target_vals = np.zeros(len(df))
                
            X_norm = (X_raw - self.features_mean) / self.features_std
            Y_norm = (target_vals - self.target_mean) / self.target_std
            
            for i in range(0, len(X_norm) - self.window_size + 1, self.stride):
                x_window = X_norm[i:i + self.window_size]
                y_target = Y_norm[i + self.window_size - 1]
                self.windows.append({
                    "x": torch.tensor(x_window, dtype=torch.float32).transpose(0, 1),
                    "y": torch.tensor([y_target], dtype=torch.float32),
                    "session_id": "Vta1a",
                    "index": i
                })
            return

        # Iterate over all assigned sessions and extract windows
        for _, row in self.inventory.iterrows():
            s_path = row['s_file']
            v_path = row['v_file']
            session_id = row['session_id']
            if not os.path.exists(s_path):
                continue
            
            s_df = pd.read_csv(s_path, encoding=self.config['processing']['encoding'], on_bad_lines='skip')
            v_df = pd.read_csv(v_path, encoding=self.config['processing']['encoding'], on_bad_lines='skip')
            
            # Clean headers
            s_df.columns = [c.strip() for c in s_df.columns]
            v_df.columns = [c.strip() for c in v_df.columns]
            
            # Find target column
            if self.target_type == 'velocity':
                cols = [c for c in v_df.columns if "velocity" in c.lower() or "speed" in c.lower()]
                target_vals = (v_df[cols[0]] / 3.6).interpolate(method='linear').bfill().ffill().values
            else:
                cols = [c for c in v_df.columns if "yaw" in c.lower()]
                target_vals = np.radians(v_df[cols[0]]).interpolate(method='linear').bfill().ffill().values
            
            # Get data arrays
            X_raw = s_df[self.config['processing']['features']].interpolate(method='linear').bfill().ffill().values
            
            # Normalize
            X_norm = (X_raw - self.features_mean) / self.features_std
            Y_norm = (target_vals - self.target_mean) / self.target_std
            
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
