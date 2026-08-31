import torch
import json
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from ml.models.speednet.speednet import SpeedNet
from ml.models.motion_state.motion_state_net import MotionStateNet
from ml.models.heading.heading_net import HeadingNet

class ModelRunner:
    """
    Unified Inference Interface for Edge Engine.
    Enforces shape checks and isolated evaluation mode execution.
    """
    def __init__(self, device_str="cpu"):
        self.device = torch.device(device_str)
        self.speednet = None
        self.motionnet = None
        self.headingnet = None
        self.norm = None
        
        self.load_models()
        self.load_normalization()
        
    def load_models(self):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        s_cfg = json.load(open(os.path.join(base_dir, "configs/speednet_training.json")))
        m_cfg = json.load(open(os.path.join(base_dir, "configs/motion_state_training.json")))
        h_cfg = json.load(open(os.path.join(base_dir, "configs/heading_training.json")))
        
        # Load and lock to eval mode
        self.speednet = SpeedNet(s_cfg['model']['in_channels'], s_cfg['model']['cnn_channels'], s_cfg['model']['kernel_size'], s_cfg['model']['gru_hidden'], s_cfg['model']['gru_layers'])
        self.speednet.load_state_dict(torch.load(os.path.join(base_dir, s_cfg['paths']['best_model']), map_location=self.device, weights_only=True))
        self.speednet.eval()
        
        self.motionnet = MotionStateNet(m_cfg['model']['in_channels'], m_cfg['model']['cnn_channels'], m_cfg['model']['kernel_size'], m_cfg['model']['gru_hidden'], m_cfg['model']['gru_layers'])
        self.motionnet.load_state_dict(torch.load(os.path.join(base_dir, m_cfg['paths']['best_model']), map_location=self.device, weights_only=True))
        self.motionnet.eval()
        
        self.headingnet = HeadingNet(h_cfg['model']['in_channels'], h_cfg['model']['cnn_channels'], h_cfg['model']['kernel_size'], h_cfg['model']['gru_hidden'], h_cfg['model']['gru_layers'])
        self.headingnet.load_state_dict(torch.load(os.path.join(base_dir, h_cfg['paths']['best_model']), map_location=self.device, weights_only=True))
        self.headingnet.eval()
        
    def load_normalization(self):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        norm_path = os.path.join(base_dir, "data/manifests/normalization.json")
        with open(norm_path, 'r') as f:
            self.norm = json.load(f)
            
    def run_inference(self, window):
        """
        Input: (6, 20) normalized numpy array
        Expected Channels: [ax, ay, az, gx, gy, gz]
        Expected Length: 20 (2.0s at 10Hz)
        
        Returns:
            speed_mps (float)
            stationary_probability (float)
            yaw_rate_rad_s (float)
        """
        # Strict dimension enforcement
        if window.shape != (6, 20):
            raise ValueError(f"ModelRunner expected tensor shape (6, 20), got {window.shape}")
            
        with torch.no_grad():
            x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            p_s = self.speednet(x).item()
            p_m = self.motionnet(x).item()
            p_h = self.headingnet(x).item()
            
        # Physical Denormalization based on canonical Phase 2 artifacts
        speed_mu = self.norm['targets']['mean']
        speed_std = self.norm['targets']['std']
        
        speed_mps = (p_s * speed_std) + speed_mu
        stationary_probability = p_m
        yaw_rate_rad_s = p_h * 0.2 # 0.2 was the standard scaler used for Yaw-rate normalization
        
        # Guard against NaN/Inf edge cases from PyTorch
        if np.isnan(speed_mps) or np.isinf(speed_mps):
            speed_mps = 0.0
        if np.isnan(yaw_rate_rad_s) or np.isinf(yaw_rate_rad_s):
            yaw_rate_rad_s = 0.0
            
        # Probability clamping
        stationary_probability = float(np.clip(stationary_probability, 0.0, 1.0))
            
        return speed_mps, stationary_probability, yaw_rate_rad_s
