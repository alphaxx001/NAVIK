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
    def __init__(self, device_str="cpu"):
        self.device = torch.device(device_str)
        self.speednet = None
        self.motionnet = None
        self.headingnet = None
        self.load_models()
        
    def load_models(self):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        s_cfg = json.load(open(os.path.join(base_dir, "configs/speednet_training.json")))
        m_cfg = json.load(open(os.path.join(base_dir, "configs/motion_state_training.json")))
        h_cfg = json.load(open(os.path.join(base_dir, "configs/heading_training.json")))
        
        self.speednet = SpeedNet(s_cfg['model']['in_channels'], s_cfg['model']['cnn_channels'], s_cfg['model']['kernel_size'], s_cfg['model']['gru_hidden'], s_cfg['model']['gru_layers'])
        self.speednet.load_state_dict(torch.load(os.path.join(base_dir, s_cfg['paths']['best_model']), map_location=self.device, weights_only=True))
        self.speednet.eval()
        
        self.motionnet = MotionStateNet(m_cfg['model']['in_channels'], m_cfg['model']['cnn_channels'], m_cfg['model']['kernel_size'], m_cfg['model']['gru_hidden'], m_cfg['model']['gru_layers'])
        self.motionnet.load_state_dict(torch.load(os.path.join(base_dir, m_cfg['paths']['best_model']), map_location=self.device, weights_only=True))
        self.motionnet.eval()
        
        self.headingnet = HeadingNet(h_cfg['model']['in_channels'], h_cfg['model']['cnn_channels'], h_cfg['model']['kernel_size'], h_cfg['model']['gru_hidden'], h_cfg['model']['gru_layers'])
        self.headingnet.load_state_dict(torch.load(os.path.join(base_dir, h_cfg['paths']['best_model']), map_location=self.device, weights_only=True))
        self.headingnet.eval()
        
    def run_inference(self, window):
        """
        window: (6, 200) numpy array
        Returns:
            speed (m/s), stationary_prob (0-1), yaw_rate (rad/s)
        """
        with torch.no_grad():
            x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            p_s = self.speednet(x).item()
            p_m = self.motionnet(x).item()
            p_h = self.headingnet(x).item()
            
        # Denormalize output values based on frozen dataset targets
        speed_mps = (p_s * 7.2721) + 10.4019
        yaw_rate = p_h * 0.2
        
        return speed_mps, p_m, yaw_rate
