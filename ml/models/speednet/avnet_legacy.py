import torch
import torch.nn as torch_nn
import torch.nn.functional as F

class AVNet(torch_nn.Module):
    def __init__(self, in_channels=6, hidden_dim=128, num_layers=2):
        super(AVNet, self).__init__()
        
        # 1D Convolutional layers to extract features from the 200-sample window
        # Input shape: (Batch, Channels, Sequence_Length) -> e.g., (B, 6, 200)
        self.conv1 = torch_nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=11, padding=5)
        self.conv2 = torch_nn.Conv1d(in_channels=64, out_channels=128, kernel_size=9, padding=4)
        
        # Pooling layer to reduce sequence length
        self.pool = torch_nn.MaxPool1d(kernel_size=2)
        
        # GRU to capture temporal dependencies over the sequence
        # After pooling, sequence length becomes 100
        self.gru = torch_nn.GRU(input_size=128, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.2)
        
        # Fully connected layers for output
        self.fc_velocity = torch_nn.Linear(hidden_dim, 1) # DDODO: predicts longitudinal velocity (1D)
        self.fc_attitude = torch_nn.Linear(hidden_dim, 4) # DDATT: predicts attitude change quaternion (4D)

    def forward(self, x):
        # x shape: (Batch, 6, 200)
        x = F.relu(self.conv1(x))
        x = self.pool(x) # Shape: (Batch, 64, 100)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x) # Shape: (Batch, 128, 50)
        
        # Prepare for GRU: (Batch, Seq_Len, Features)
        x = x.permute(0, 2, 1) 
        
        # Pass through GRU
        gru_out, _ = self.gru(x)
        
        # We only care about the last output of the GRU for the window
        last_out = gru_out[:, -1, :] # Shape: (Batch, hidden_dim)
        
        # Predict velocity and attitude
        velocity = self.fc_velocity(last_out)
        attitude = self.fc_attitude(last_out)
        
        # Normalize the quaternion attitude output to ensure it represents a valid rotation
        attitude = F.normalize(attitude, p=2, dim=1)
        
        return velocity, attitude

if __name__ == "__main__":
    # Test the model with dummy data
    # Batch size 32, 6 channels (accel x,y,z + gyro x,y,z), 200 samples
    dummy_input = torch.randn(32, 6, 200)
    model = AVNet()
    vel, att = model(dummy_input)
    print(f"Model Forward Pass Successful!")
    print(f"Velocity shape: {vel.shape} (Expected: 32, 1)")
    print(f"Attitude shape: {att.shape} (Expected: 32, 4)")
