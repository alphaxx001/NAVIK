import torch
import torch.nn as nn

class MotionStateNet(nn.Module):
    """
    MotionStateNet: 1D-CNN + GRU for binary classification of vehicle motion state.
    Input: [batch, 6, 20]
    Output: [batch, 1] (Probability of being Stationary)
    """
    def __init__(self, in_channels=6, cnn_channels=16, kernel_size=3, gru_hidden=32, gru_layers=1):
        super(MotionStateNet, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, cnn_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU()
        )
        
        # Sequence length was 20, maxpool halves it to 10
        self.gru = nn.GRU(
            input_size=cnn_channels,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        c = self.cnn(x)
        c = c.permute(0, 2, 1)
        out, _ = self.gru(c)
        last_out = out[:, -1, :]
        return self.fc(last_out)
