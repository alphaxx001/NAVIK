import numpy as np
from scipy.spatial.transform import Rotation as R

def skew(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

class ESKF:
    def __init__(self):
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.q = R.from_quat([0, 0, 0, 1])  # xyzw
        self.b_a = np.zeros(3)
        self.b_g = np.zeros(3)
        self.P = np.eye(15) * 1e-2
        self.g = np.array([0, 0, -9.81])

    def predict(self, accel, gyro, dt):
        # Compensate IMU measurements
        f = accel - self.b_a
        w = gyro - self.b_g

        # Nominal State Kinematics
        C = self.q.as_matrix()
        a_n = C @ f + self.g
        
        self.p = self.p + self.v * dt + 0.5 * a_n * (dt ** 2)
        self.v = self.v + a_n * dt
        
        w_norm = np.linalg.norm(w)
        if w_norm > 1e-8:
            dq = R.from_rotvec(w * dt)
            # Local perturbation for nominal state propagation: q = q * dq
            self.q = self.q * dq

        # Error State Jacobians (Continuous-time approximated to discrete)
        F = np.eye(15)
        F[0:3, 3:6] = np.eye(3) * dt
        
        # -C * [f]_x
        F[3:6, 6:9] = -C @ skew(f) * dt
        F[3:6, 9:12] = -C * dt
        
        F[6:9, 6:9] = np.eye(3) - skew(w) * dt
        F[6:9, 12:15] = -C * dt # If global error state is used, normally -C*dt. 
        # Actually, standard error state uses F[6:9, 12:15] = -np.eye(3)*dt for LOCAL error. 
        # For global error state: delta_theta_dot = -C * delta_bg. Let's stick to global.
        
        # Process Noise Covariance Q
        Q = np.zeros((15, 15))
        Q[3:6, 3:6] = np.eye(3) * (0.01 * dt)**2
        Q[6:9, 6:9] = np.eye(3) * (0.001 * dt)**2
        Q[9:12, 9:12] = np.eye(3) * (1e-5 * dt)**2
        Q[12:15, 12:15] = np.eye(3) * (1e-6 * dt)**2
        
        self.P = F @ self.P @ F.T + Q

    def update(self, dz, H, R_cov):
        """Generic Kalman Update"""
        S = H @ self.P @ H.T + R_cov
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ dz
        
        # Joseph form covariance update
        I_KH = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_cov @ K.T
        
        # Inject error state
        self.p += dx[0:3]
        self.v += dx[3:6]
        
        # Global attitude perturbation: C = (I + [dtheta]x) * C
        dtheta = dx[6:9]
        if np.linalg.norm(dtheta) > 1e-8:
            dq = R.from_rotvec(dtheta)
            self.q = dq * self.q
            
        self.b_a += dx[9:12]
        self.b_g += dx[12:15]

    def update_oracle_speed(self, speed_ms, nhc=False):
        """
        Updates filter using Vehicle CAN Speed.
        CAN speed is assumed to be the forward velocity in v-frame (Y axis).
        """
        C = self.q.as_matrix()
        
        # Predicted velocity in v-frame
        v_v = C.T @ self.v
        
        if not nhc:
            # Oracle Speed ONLY (Y-axis)
            dz = np.array([speed_ms - v_v[1]])
            H = np.zeros((1, 15))
            
            # H_v = C.T, we take the 2nd row (Y axis)
            H[0, 3:6] = C.T[1, :]
            
            # H_theta = C.T * [v_n]x
            H_theta = C.T @ skew(self.v)
            H[0, 6:9] = H_theta[1, :]
            
            R_cov = np.array([[1.0]]) # 1 m/s uncertainty
            self.update(dz, H, R_cov)
            
        else:
            # ORACLE + NHC (X=0, Y=speed, Z=0)
            dz = np.array([
                0.0 - v_v[0],
                speed_ms - v_v[1],
                0.0 - v_v[2]
            ])
            H = np.zeros((3, 15))
            H[0:3, 3:6] = C.T
            H[0:3, 6:9] = C.T @ skew(self.v)
            
            R_cov = np.diag([0.1, 1.0, 0.1])
            self.update(dz, H, R_cov)

    def update_zupt(self):
        """Zero Velocity Update"""
        dz = np.zeros(3) - self.v
        H = np.zeros((3, 15))
        H[0:3, 3:6] = np.eye(3)
        R_cov = np.eye(3) * 0.01
        self.update(dz, H, R_cov)
