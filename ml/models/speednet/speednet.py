import torch
import torch.nn as nn

class SpeedNet(nn.Module):
    """
    SpeedNet: 1D-CNN + GRU model for predicting vehicle velocity from Smartphone IMU.
    Input: [batch, channels, seq_length] -> [B, 6, 20]
    Output: [batch, 1] -> Velocity (m/s)
    """
    def __init__(self, in_channels=6, cnn_channels=32, kernel_size=3, gru_hidden=64, gru_layers=1):
        super(SpeedNet, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, cnn_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU()
        )
        
        self.gru = nn.GRU(
            input_size=cnn_channels,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Predicts velocity
        )
        
    def forward(self, x):
        # x shape: [B, 6, 20]
        c = self.cnn(x) # [B, 32, 20]
        
        # Prepare for GRU: [B, Seq, Features]
        c = c.permute(0, 2, 1) # [B, 20, 32]
        
        out, _ = self.gru(c) # out: [B, 20, 64]
        
        # Take the last hidden state for prediction (end of window)
        last_out = out[:, -1, :] # [B, 64]
        
        vel = self.fc(last_out) # [B, 1]
        return vel
