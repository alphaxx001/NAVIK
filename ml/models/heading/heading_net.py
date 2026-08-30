import torch
import torch.nn as nn

class HeadingNet(nn.Module):
    """
    HeadingNet predicts Vehicle Yaw Rate (rad/s) from 6-DoF smartphone IMU windows.
    Because of corrupted axis mappings in IO-VNBD, it consumes all 6 axes and learns
    the correct orientation transformations and denoising logic.
    Input: [batch, 6, 20]
    Output: [batch, 1] (yaw rate)
    """
    def __init__(self, in_channels=6, cnn_channels=32, kernel_size=3, gru_hidden=64, gru_layers=1):
        super(HeadingNet, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, cnn_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
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
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        c = self.cnn(x)
        c = c.permute(0, 2, 1)
        out, _ = self.gru(c)
        last_out = out[:, -1, :]
        return self.fc(last_out)
