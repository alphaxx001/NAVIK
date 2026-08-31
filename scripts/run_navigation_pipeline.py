import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.spatial.transform import Rotation as R
import sys

from ml.models.speednet.speednet import SpeedNet
from ml.models.motion_state.motion_state_net import MotionStateNet
from ml.models.heading.heading_net import HeadingNet
from libnavik.python_reference.eskf import ESKF
from libnavik.python_reference.evaluate_baseline import latlon_to_enu
from ml.data.dataset_loader import IOVNBDProductionDataset
from ml.data.io_vnbd_schema import IOVNBDSchemaResolver

def enu_to_latlon(e, n, lat0, lon0):
    R = 6378137.0
    lat0_rad = np.radians(lat0)
    lat = lat0 + np.degrees(n / R)
    lon = lon0 + np.degrees(e / (R * np.cos(lat0_rad)))
    return lat, lon

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.test_kdtree_equivalence import KDTreeHMMMapMatcher

def load_models(device):
    s_cfg = json.load(open("configs/speednet_training.json"))
    m_cfg = json.load(open("configs/motion_state_training.json"))
    h_cfg = json.load(open("configs/heading_training.json"))
    
    speednet = SpeedNet(s_cfg['model']['in_channels'], s_cfg['model']['cnn_channels'], s_cfg['model']['kernel_size'], s_cfg['model']['gru_hidden'], s_cfg['model']['gru_layers'])
    speednet.load_state_dict(torch.load(s_cfg['paths']['best_model'], map_location=device, weights_only=True)); speednet.eval()
    
    motionnet = MotionStateNet(m_cfg['model']['in_channels'], m_cfg['model']['cnn_channels'], m_cfg['model']['kernel_size'], m_cfg['model']['gru_hidden'], m_cfg['model']['gru_layers'])
    motionnet.load_state_dict(torch.load(m_cfg['paths']['best_model'], map_location=device, weights_only=True)); motionnet.eval()
    
    headingnet = HeadingNet(h_cfg['model']['in_channels'], h_cfg['model']['cnn_channels'], h_cfg['model']['kernel_size'], h_cfg['model']['gru_hidden'], h_cfg['model']['gru_layers'])
    headingnet.load_state_dict(torch.load(h_cfg['paths']['best_model'], map_location=device, weights_only=True)); headingnet.eval()
    
    return speednet, motionnet, headingnet

def extract_windows(df_imu, imu_schema, window_size=200):
    gx = df_imu[imu_schema['gyro_x']].values; gy = df_imu[imu_schema['gyro_y']].values; gz = df_imu[imu_schema['gyro_z']].values
    ax = df_imu[imu_schema['accel_x']].values; ay = df_imu[imu_schema['accel_y']].values; az = df_imu[imu_schema['accel_z']].values
    
    g_m = np.pi/180.0 if 'deg' in imu_schema['gyro_x'].lower() else 1.0
    a_m = 9.81 if '(g)' in imu_schema['accel_x'].lower() else 1.0
    
    gyro = np.stack([gx, gy, gz], axis=1) * g_m
    accel = np.stack([ax, ay, az], axis=1) * a_m
    t_ms = df_imu[imu_schema['time']].values
    
    # Standard scaling constants from Phase 2
    mu_a = np.array([-0.0528,  0.4284,  9.7424]); std_a = np.array([1.2335, 1.4878, 1.4429])
    mu_g = np.array([-0.0017, -0.0033, -0.0028]); std_g = np.array([0.0577, 0.0827, 0.0825])
    
    inputs = []
    times = []
    
    for i in range(len(df_imu) - window_size + 1):
        a_w = (accel[i:i+window_size] - mu_a) / std_a
        g_w = (gyro[i:i+window_size] - mu_g) / std_g
        feats = np.concatenate([a_w, g_w], axis=1)
        inputs.append(feats.T)
        times.append(t_ms[i + window_size - 1])
        
    return np.array(inputs, dtype=np.float32), np.array(times), gyro, accel, t_ms

def infer_networks(inputs, speednet, motionnet, headingnet, device):
    bs = 1024
    p_speed = []; p_motion = []; p_heading = []
    
    with torch.no_grad():
        for i in range(0, len(inputs), bs):
            batch = torch.tensor(inputs[i:i+bs]).to(device)
            s = speednet(batch).cpu().numpy()
            m = motionnet(batch).cpu().numpy()
            h = headingnet(batch).cpu().numpy()
            p_speed.append(s); p_motion.append(m); p_heading.append(h)
            
    p_speed = np.concatenate(p_speed).squeeze()
    p_motion = np.concatenate(p_motion).squeeze()
    p_heading = np.concatenate(p_heading).squeeze()
    
    p_speed = (p_speed * 7.2721) + 10.4019
    p_heading = p_heading * 0.2
    
    p_class = (p_motion > 0.569).astype(int)
    p_class_hyst = np.zeros_like(p_class)
    for i in range(2, len(p_class)):
        if p_class[i] == 1 and p_class[i-1] == 1 and p_class[i-2] == 1:
            p_class_hyst[i] = 1; p_class_hyst[i-1] = 1; p_class_hyst[i-2] = 1
            
    return p_speed, p_class_hyst, p_heading

def run_pipeline(session_id="Vta1a", output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")
    
    # 1. Load Data
    inventory = pd.read_csv("data/manifests/session_inventory.csv")
    row = inventory[inventory['session_id'] == session_id].iloc[0]
    df_imu = pd.read_csv(row['s_file'], encoding='latin1', on_bad_lines='skip')
    df_v = pd.read_csv(row['v_file'], encoding='latin1', on_bad_lines='skip')
    
    imu_schema = IOVNBDSchemaResolver.resolve_imu_columns(df_imu)
    v_schema = IOVNBDSchemaResolver.resolve_v_columns(df_v)
    
    lat = df_v[v_schema['lat']].values
    lon = df_v[v_schema['lon']].values
    head = df_v[v_schema['heading']].values
    t_v = df_v[v_schema['time']].values
    
    # 2. Neural Inference
    speednet, motionnet, headingnet = load_models(device)
    inputs, t_pred_ms, gyro, accel, t_ms = extract_windows(df_imu, imu_schema)
    p_speed, p_class, p_heading = infer_networks(inputs, speednet, motionnet, headingnet, device)
    
    # 3. ESKF Initialization (Using GNSS for init only)
    eskf = ESKF()
    lat0, lon0 = lat[0], lon[0]
    yaw_0 = (90.0 - head[0]) * np.pi / 180.0
    eskf.q = R.from_euler('z', yaw_0)
    
    t_imu = t_ms / 1000.0
    pos_dr = np.zeros((len(p_speed), 3))
    head_dr = np.zeros(len(p_speed))
    timestamps = np.zeros(len(p_speed))
    
    # 4. ESKF Propagation (Absolute Zero GNSS consumption here)
    for i in range(1, len(p_speed)):
        idx = i + 199 # align with window end
        if idx >= len(t_ms): break
        dt = t_imu[idx] - t_imu[idx-1]
        if dt <= 0: dt = 0.01
        
        p_gyro = gyro[idx].copy(); p_gyro[2] = p_heading[i]
        eskf.predict(accel[idx], p_gyro, dt)
        
        if p_class[i] == 1:
            eskf.update_zupt()
        else:
            eskf.update_oracle_speed(p_speed[i], nhc=True)
            
        pos_dr[i] = eskf.p.copy()
        head_dr[i] = eskf.q.as_euler('xyz')[2]
        timestamps[i] = t_ms[idx]
        
    # 5. Open-Loop KD-Tree HMM Map Matching
    print("Running KD-Tree Map Matcher...")
    graph_file = f"map/runtime/{session_id}_graph.json"
    matcher = KDTreeHMMMapMatcher(graph_file, search_radius=100.0)
    
    # Downsample trajectory for HMM
    trajectory = []
    valid_idxs = []
    for i in range(0, len(p_speed), 10):
        if timestamps[i] == 0: continue
        trajectory.append({'x': pos_dr[i, 0], 'y': pos_dr[i, 1], 'h': head_dr[i]})
        valid_idxs.append(i)
        
    matched_pts_down, states_down = matcher.viterbi_match(trajectory)
    
    # Upsample
    pos_map = np.zeros_like(pos_dr[:, 0:2])
    states_full = ['DR'] * len(p_speed)
    for i in range(len(p_speed)):
        if timestamps[i] == 0: continue
        low_idx = i // 10
        high_idx = min(low_idx + 1, len(matched_pts_down) - 1)
        if low_idx >= len(matched_pts_down): low_idx = len(matched_pts_down) - 1
        alpha = (i % 10) / 10.0
        
        p1 = matched_pts_down[low_idx]; p2 = matched_pts_down[high_idx]
        pos_map[i, 0] = p1[0] * (1 - alpha) + p2[0] * alpha
        pos_map[i, 1] = p1[1] * (1 - alpha) + p2[1] * alpha
        states_full[i] = 'DR_FALLBACK' if states_down[low_idx] == 'DR_FALLBACK' else 'MAP_CONTEXT'
        
    # 6. Format Output
    output_records = []
    for i in range(len(p_speed)):
        if timestamps[i] == 0: continue
        lat_map, lon_map = enu_to_latlon(pos_map[i,0], pos_map[i,1], lat0, lon0)
        
        mode = states_full[i]
        gnss_avail = False
        map_match_avail = (mode == 'MAP_CONTEXT')
        
        output_records.append({
            'timestamp': timestamps[i],
            'latitude': lat_map,
            'longitude': lon_map,
            'velocity': p_speed[i],
            'heading': head_dr[i] * 180.0 / np.pi,
            'navigation_mode': mode,
            'gnss_available': gnss_avail,
            'map_match_available': map_match_avail,
            'map_match_confidence': 0.9 if map_match_avail else 0.0,
            'road_id': 'UNKNOWN' # We can track road ID in HMM if needed, simplified here
        })
        
    df_out = pd.DataFrame(output_records)
    csv_path = os.path.join(output_dir, f"{session_id}_navigation_output.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"Output saved to {csv_path}")
    
    # 7. Visualization
    plt.figure(figsize=(12, 12))
    
    # GT
    enu_gt = np.zeros((len(lat), 2))
    for i in range(len(lat)):
        x, y = latlon_to_enu(lat[i], lon[i], lat0, lon0)
        enu_gt[i, 0] = x; enu_gt[i, 1] = y
    plt.plot(enu_gt[:, 0], enu_gt[:, 1], 'k--', label="Ground Truth (GNSS)", linewidth=2)
    
    # Map
    segs = 0
    for s in matcher.segments:
        if segs > 1000: break
        plt.plot([s['x1'], s['x2']], [s['y1'], s['y2']], 'gray', alpha=0.3)
        segs += 1
        
    plt.plot(pos_dr[:, 0], pos_dr[:, 1], 'r-', label="Raw DR (ESKF Authoritative)", linewidth=1)
    
    # Fallback vs Context
    map_context_x = [pos_map[i, 0] for i in range(len(pos_map)) if states_full[i] == 'MAP_CONTEXT']
    map_context_y = [pos_map[i, 1] for i in range(len(pos_map)) if states_full[i] == 'MAP_CONTEXT']
    fallback_x = [pos_map[i, 0] for i in range(len(pos_map)) if states_full[i] == 'DR_FALLBACK']
    fallback_y = [pos_map[i, 1] for i in range(len(pos_map)) if states_full[i] == 'DR_FALLBACK']
    
    plt.scatter(map_context_x, map_context_y, c='blue', s=1, label="MAP_CONTEXT")
    plt.scatter(fallback_x, fallback_y, c='orange', s=1, label="DR_FALLBACK")
    
    plt.legend()
    plt.title(f"Final Architecture End-to-End: {session_id}")
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    fig_path = os.path.join(output_dir, f"{session_id}_visualization.png")
    plt.savefig(fig_path)
    print(f"Visualization saved to {fig_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_pipeline(sys.argv[1])
    else:
        run_pipeline("Vta1a")
