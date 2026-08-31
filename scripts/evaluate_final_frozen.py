import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.spatial.transform import Rotation as R
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.run_navigation_pipeline import load_models, extract_windows, infer_networks, enu_to_latlon
from libnavik.python_reference.eskf import ESKF
from libnavik.python_reference.evaluate_baseline import latlon_to_enu
from ml.data.io_vnbd_schema import IOVNBDSchemaResolver
from scripts.test_kdtree_equivalence import KDTreeHMMMapMatcher

def evaluate_session(session_id, speednet, motionnet, headingnet, device, out_dir):
    print(f"\n--- Evaluating Test Session: {session_id} ---")
    
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
    vel_gt = df_v[v_schema['speed']].values if 'speed' in v_schema else df_v[v_schema['velocity']].values
    t_v = df_v[v_schema['time']].values
    
    # 2. Neural Inference
    inputs, t_pred_ms, gyro, accel, t_ms = extract_windows(df_imu, imu_schema)
    p_speed, p_class, p_heading = infer_networks(inputs, speednet, motionnet, headingnet, device)
    
    # 3. ESKF Initialization
    t0_eskf = time.time()
    eskf = ESKF()
    lat0, lon0 = lat[0], lon[0]
    yaw_0 = (90.0 - head[0]) * np.pi / 180.0
    eskf.q = R.from_euler('z', yaw_0)
    
    t_imu = t_ms / 1000.0
    pos_dr = np.zeros((len(p_speed), 3))
    head_dr = np.zeros(len(p_speed))
    vel_dr = np.zeros((len(p_speed), 3))
    timestamps = np.zeros(len(p_speed))
    
    # 4. ESKF Propagation
    for i in range(1, len(p_speed)):
        idx = i + 19 # align with window end
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
        vel_dr[i] = eskf.v.copy()
        timestamps[i] = t_ms[idx]
        
    t_eskf_total = time.time() - t0_eskf
    
    # Ground truth alignment
    # Align clocks to 0 to fix relative epoch differences between S and V datasets
    t_v_rel = t_v - t_v[0]
    timestamps_rel = timestamps.copy()
    valid_ts = timestamps > 0
    if np.any(valid_ts):
        timestamps_rel[valid_ts] = timestamps[valid_ts] - timestamps[valid_ts][0]
        
    lat_interp = np.interp(timestamps_rel, t_v_rel, lat)
    lon_interp = np.interp(timestamps_rel, t_v_rel, lon)
    v_gt_interp = np.interp(timestamps_rel, t_v_rel, vel_gt)
    enu_gt = np.zeros((len(lat_interp), 2))
    for i in range(len(lat_interp)):
        if timestamps[i] == 0: continue
        x, y = latlon_to_enu(lat_interp[i], lon_interp[i], lat0, lon0)
        enu_gt[i, 0] = x; enu_gt[i, 1] = y
        
    # 5. Open-Loop KD-Tree HMM Map Matching
    t0_map = time.time()
    graph_file = f"map/runtime/{session_id}_graph.json"
    matcher = KDTreeHMMMapMatcher(graph_file, search_radius=100.0)
    
    trajectory = []
    for i in range(0, len(p_speed), 10):
        if timestamps[i] == 0: continue
        trajectory.append({'x': pos_dr[i, 0], 'y': pos_dr[i, 1], 'h': head_dr[i]})
        
    matched_pts_down, states_down = matcher.viterbi_match(trajectory)
    
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
        
    t_map_total = time.time() - t0_map
    
    # 6. Metrics Calculation
    valid = timestamps > 0
    errs_dr = np.linalg.norm(pos_dr[valid, 0:2] - enu_gt[valid], axis=1)
    errs_map = np.linalg.norm(pos_map[valid] - enu_gt[valid], axis=1)
    
    # Distance checkpoints
    dist_cum = np.cumsum(np.linalg.norm(np.diff(enu_gt[valid], axis=0), axis=1))
    dist_cum = np.insert(dist_cum, 0, 0)
    
    total_dist = dist_cum[-1]
    
    drift_dr = (np.mean(errs_dr) / total_dist) * 100
    drift_map = (np.mean(errs_map) / total_dist) * 100
    rmse_dr = np.sqrt(np.mean(errs_dr**2))
    rmse_map = np.sqrt(np.mean(errs_map**2))
    fpe_dr = errs_dr[-1]
    fpe_map = errs_map[-1]
    
    mae_vel = np.mean(np.abs(np.linalg.norm(vel_dr[valid, 0:2], axis=1) - v_gt_interp[valid]))
    
    fb_pct = (sum(1 for s in states_full if s == 'DR_FALLBACK') / len(states_full)) * 100
    cov_pct = 100.0 - fb_pct
    
    metrics = {
        'session_id': session_id,
        'total_distance': total_dist,
        'dr_drift': drift_dr,
        'map_drift': drift_map,
        'dr_rmse': rmse_dr,
        'map_rmse': rmse_map,
        'dr_fpe': fpe_dr,
        'map_fpe': fpe_map,
        'vel_mae': mae_vel,
        'fallback_pct': fb_pct,
        'coverage_pct': cov_pct,
        'runtime_eskf': t_eskf_total,
        'runtime_map': t_map_total
    }
    
    # Save specific distance errors (Ablation)
    checkpoints = [50, 100, 200, 500, 1000, 2000]
    for cp in checkpoints:
        if total_dist >= cp:
            idx = np.argmax(dist_cum >= cp)
            metrics[f'dr_err_{cp}m'] = errs_dr[idx]
            metrics[f'map_err_{cp}m'] = errs_map[idx]
        else:
            metrics[f'dr_err_{cp}m'] = None
            metrics[f'map_err_{cp}m'] = None
            
    # 7. Visualization
    plt.figure(figsize=(12, 12))
    plt.plot(enu_gt[valid, 0], enu_gt[valid, 1], 'k--', label="Ground Truth (GNSS)", linewidth=2)
    plt.plot(pos_dr[valid, 0], pos_dr[valid, 1], 'r-', label=f"Raw DR (Drift: {drift_dr:.1f}%)", linewidth=1.5)
    plt.plot(pos_map[valid, 0], pos_map[valid, 1], 'g-', label=f"Map Context (Drift: {drift_map:.1f}%)", linewidth=1.5)
    
    segs = 0
    for s in matcher.segments:
        if segs > 1000: break
        plt.plot([s['x1'], s['x2']], [s['y1'], s['y2']], 'gray', alpha=0.3)
        segs += 1
        
    plt.legend()
    plt.title(f"Final Test Evaluation: {session_id}")
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    plt.savefig(os.path.join(out_dir, f"{session_id}_final_trajectory.png"))
    plt.close()
    
    return metrics

def main():
    test_sessions = [
        "Vta10", "Vta11", "Vta2", "Vta20", "Vta29", "Vta3",
        "Vtb11", "Vtb12", "Vw10", "Vw11", "Vw3", "Vw5"
    ]
    
    out_dir = "ml/evaluation/outputs/final"
    os.makedirs(out_dir, exist_ok=True)
    
    device = torch.device("cpu")
    speednet, motionnet, headingnet = load_models(device)
    
    results = []
    
    for sess in test_sessions:
        metrics = evaluate_session(sess, speednet, motionnet, headingnet, device, out_dir)
        results.append(metrics)
        print(f"  -> DR Drift: {metrics['dr_drift']:.1f}% | Map Drift: {metrics['map_drift']:.1f}% | Coverage: {metrics['coverage_pct']:.1f}%")
        
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(out_dir, "aggregate_final_results.csv"), index=False)
    
    print("\n--- FINAL TEST SET SUMMARY ---")
    print(f"Median DR Drift: {df_res['dr_drift'].median():.2f}%")
    print(f"Mean DR Drift: {df_res['dr_drift'].mean():.2f}%")
    print(f"Median MAP Drift: {df_res['map_drift'].median():.2f}%")
    print(f"Mean MAP Drift: {df_res['map_drift'].mean():.2f}%")
    
    worst_dr_sess = df_res.loc[df_res['dr_drift'].idxmax()]
    print(f"Worst Session (DR): {worst_dr_sess['session_id']} ({worst_dr_sess['dr_drift']:.2f}%)")
    
    worst_map_sess = df_res.loc[df_res['map_drift'].idxmax()]
    print(f"Worst Session (MAP): {worst_map_sess['session_id']} ({worst_map_sess['map_drift']:.2f}%)")
    
    # Save metadata
    meta = {
        "git_commit": "f561adb",
        "architecture": "Open-Loop KD-Tree HMM",
        "map_feedback": False,
        "gnss_used_after_init": False,
        "test_sessions": test_sessions
    }
    with open(os.path.join(out_dir, "evaluation_metadata.json"), "w") as f:
        json.dump(meta, f, indent=4)

if __name__ == "__main__":
    main()
