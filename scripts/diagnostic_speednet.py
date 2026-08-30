import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ml.models.speednet.speednet import SpeedNet
from libnavik.python_reference.eskf import ESKF
from libnavik.python_reference.evaluate_baseline import latlon_to_enu
from scipy.spatial.transform import Rotation as R

def run_eskf_trajectory(session_id, vel_predictions, use_nhc=True, r_cov_y=1.0):
    s_path = f"data/raw/IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/S-Dataset/S-{session_id}.csv"
    v_path = f"data/raw/IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/V-Dataset/V-{session_id}.csv"
    if not os.path.exists(s_path):
        s_path = s_path.replace(f"S-{session_id}", f"S-V{session_id.lower()}")
        v_path = v_path.replace(f"V-{session_id}", f"V-v{session_id.lower()}")
        
    s_df = pd.read_csv(s_path, encoding='latin1', on_bad_lines='skip')
    v_df = pd.read_csv(v_path, encoding='latin1', on_bad_lines='skip')
    s_df.columns = [c.strip() for c in s_df.columns]
    v_df.columns = [c.strip() for c in v_df.columns]
    
    ax_c = [c for c in s_df.columns if 'ACCELEROMETER X' in c.upper()][0]
    ay_c = [c for c in s_df.columns if 'ACCELEROMETER Y' in c.upper()][0]
    az_c = [c for c in s_df.columns if 'ACCELEROMETER Z' in c.upper()][0]
    gx_c = [c for c in s_df.columns if 'GYROSCOPE X' in c.upper()][0]
    gy_c = [c for c in s_df.columns if 'GYROSCOPE Y' in c.upper()][0]
    gz_c = [c for c in s_df.columns if 'GYROSCOPE Z' in c.upper()][0]
    t_c = [c for c in s_df.columns if 'TIME' in c.upper()][0]
    
    accel = s_df[[ax_c, ay_c, az_c]].values
    gyro = s_df[[gx_c, gy_c, gz_c]].values
    heading = v_df['Heading (degrees)'].values
    lat = v_df['Latitude (degrees)'].values
    lon = v_df['Longitude (degrees)'].values
    t_ms = s_df[t_c].values
    
    lat0, lon0 = lat[0], lon[0]
    enu_gt = np.array([latlon_to_enu(lat[i], lon[i], lat0, lon0) for i in range(len(lat))])
    
    eskf = ESKF()
    yaw_0 = (90.0 - heading[0]) * np.pi / 180.0
    eskf.q = R.from_euler('z', yaw_0)
    eskf.v = eskf.q.as_matrix() @ np.array([0, vel_predictions[0], 0])
    
    pos_est = []
    
    for i in range(1, len(vel_predictions)):
        idx = i + 19
        dt = (t_ms[idx] - t_ms[idx-1]) / 1000.0
        if dt <= 0: dt = 0.1
        
        eskf.predict(accel[idx], gyro[idx], dt)
        
        v_pred = vel_predictions[i]
        
        # Manually apply NHC and Speed update with specific covariance
        C = eskf.q.as_matrix()
        v_v = C.T @ eskf.v
        
        if use_nhc:
            dz = np.array([0.0 - v_v[0], v_pred - v_v[1], 0.0 - v_v[2]])
            H = np.zeros((3, 15))
            H[0:3, 3:6] = C.T
            H[0:3, 6:9] = C.T @ np.array([[0, -eskf.v[2], eskf.v[1]], [eskf.v[2], 0, -eskf.v[0]], [-eskf.v[1], eskf.v[0], 0]])
            R_cov = np.diag([0.1, r_cov_y, 0.1])
            eskf.update(dz, H, R_cov)
            
        pos_est.append(eskf.p.copy())
        
    pos_est = np.array(pos_est)
    gt_traj = enu_gt[20:]
    err = np.linalg.norm(pos_est[:, 0:2] - gt_traj, axis=1)
    rmse = np.sqrt(np.mean(err**2))
    drift_pct = (err[-1] / (np.sum(vel_predictions * 0.1) + 1e-6)) * 100.0
    return rmse, drift_pct

def run_diagnostics():
    device = torch.device("cpu")
    config_path = "configs/speednet_training.json"
    with open(config_path, 'r') as f: config = json.load(f)
    
    from ml.data.dataset_loader import IOVNBDProductionDataset
    val_dataset = IOVNBDProductionDataset(split="val", stride_override=1)
    test_dataset = IOVNBDProductionDataset(split="test", stride_override=1)
    
    m_cfg = config['model']
    model = SpeedNet(m_cfg['in_channels'], m_cfg['cnn_channels'], m_cfg['kernel_size'], m_cfg['gru_hidden'], m_cfg['gru_layers'])
    model.load_state_dict(torch.load(config['paths']['best_model'], map_location=device, weights_only=True))
    model.eval()
    
    def get_predictions(dataset):
        loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False)
        preds, ys, sess = [], [], []
        with torch.no_grad():
            for bx, by, s in loader:
                preds.append(model(bx.to(device)).cpu().numpy())
                ys.append(by.numpy())
                sess.extend(s)
        t_std, t_mean = dataset.target_std, dataset.target_mean
        p_real = (np.vstack(preds) * t_std) + t_mean
        y_real = (np.vstack(ys) * t_std) + t_mean
        return p_real.flatten(), y_real.flatten(), np.array(sess)

    print("Evaluating Validation Set for Covariance estimation...")
    val_p, val_y, _ = get_predictions(val_dataset)
    val_variance = np.var(val_p - val_y)
    print(f"Validation Residual Variance (R_cov_y): {val_variance:.2f}")
    
    print("Evaluating Test Set...")
    test_p, test_y, test_sess = get_predictions(test_dataset)
    
    # 1. Per-session performance
    unique_sess = np.unique(test_sess)
    print("\nPer-Session Test Performance (MAE m/s):")
    for s in unique_sess:
        idx = (test_sess == s)
        mae = np.mean(np.abs(test_p[idx] - test_y[idx]))
        print(f"  Session {s}: {mae:.2f}")
        
    # 2. Low-speed / High-speed regime
    print("\nSpeed Regime Performance (MAE m/s):")
    stationary = (test_y < 0.5)
    low_speed = (test_y >= 0.5) & (test_y < 5.0)
    high_speed = (test_y >= 5.0)
    
    if np.sum(stationary)>0: print(f"  Stationary (<0.5m/s): {np.mean(np.abs(test_p[stationary] - test_y[stationary])):.2f}")
    if np.sum(low_speed)>0: print(f"  Low Speed (0.5-5m/s): {np.mean(np.abs(test_p[low_speed] - test_y[low_speed])):.2f}")
    if np.sum(high_speed)>0: print(f"  High Speed (>5m/s): {np.mean(np.abs(test_p[high_speed] - test_y[high_speed])):.2f}")
    
    # 3. ZUPT threshold check
    zeros_gt = np.sum(test_y < 0.1)
    zeros_pred = np.sum(test_p < 0.1)
    print(f"\nZUPT Check (<0.1 m/s): Ground Truth Zeros: {zeros_gt}, Predicted Zeros: {zeros_pred}")
    
    # Plot Error Dist
    plt.figure()
    plt.hist(test_p - test_y, bins=50)
    plt.title("SpeedNet Residual Distribution")
    plt.savefig("ml/evaluation/outputs/speednet/error_dist.png")
    
    # 4. Critical Navigation Sanity Test
    sess_id = unique_sess[0]
    idx = (test_sess == sess_id)
    sess_gt = test_y[idx]
    sess_pr = test_p[idx]
    
    print(f"\nNavigation Isolation Test on Session {sess_id}:")
    rmse1, d1 = run_eskf_trajectory(sess_id, sess_gt, use_nhc=True, r_cov_y=1.0)
    rmse2, d2 = run_eskf_trajectory(sess_id, sess_pr, use_nhc=True, r_cov_y=1.0)
    rmse3, d3 = run_eskf_trajectory(sess_id, sess_pr, use_nhc=True, r_cov_y=val_variance)
    
    print(f"  Oracle Speed (R=1.0)              : RMSE {rmse1:.2f}m | Drift {d1:.2f}%")
    print(f"  SpeedNet Prediction (R=1.0)       : RMSE {rmse2:.2f}m | Drift {d2:.2f}%")
    print(f"  SpeedNet + Val Covariance (R={val_variance:.2f}): RMSE {rmse3:.2f}m | Drift {d3:.2f}%")

if __name__ == "__main__":
    run_diagnostics()
