import numpy as np

class SensorBuffer:
    def __init__(self, window_size=200):
        self.window_size = window_size
        self.buffer = []
        
        # Scaling parameters from Phase 2
        self.mu_a = np.array([-0.0528,  0.4284,  9.7424])
        self.std_a = np.array([1.2335, 1.4878, 1.4429])
        self.mu_g = np.array([-0.0017, -0.0033, -0.0028])
        self.std_g = np.array([0.0577, 0.0827, 0.0825])
        
    def add_sample(self, timestamp, accel, gyro):
        """
        accel: [ax, ay, az] in m/s^2
        gyro: [gx, gy, gz] in rad/s
        """
        self.buffer.append({'t': timestamp, 'a': accel, 'g': gyro})
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
            
    def is_ready(self):
        return len(self.buffer) == self.window_size
        
    def get_window(self):
        if not self.is_ready():
            return None
            
        accels = np.array([x['a'] for x in self.buffer])
        gyros = np.array([x['g'] for x in self.buffer])
        
        a_w = (accels - self.mu_a) / self.std_a
        g_w = (gyros - self.mu_g) / self.std_g
        
        feats = np.concatenate([a_w, g_w], axis=1) # Shape: (200, 6)
        return feats.T # Shape: (6, 200)
        
    def get_latest_timestamp(self):
        if not self.buffer: return None
        return self.buffer[-1]['t']
