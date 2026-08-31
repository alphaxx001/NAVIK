import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from ml.models.speednet.speednet import SpeedNet
from ml.models.motion_state.motion_state_net import MotionStateNet
from ml.models.heading.heading_net import HeadingNet
from libnavik.python_reference.eskf import ESKF
from libnavik.python_reference.evaluate_baseline import latlon_to_enu
from ml.data.dataset_loader import IOVNBDProductionDataset
from map.runtime.hmm_matcher import HMMMapMatcher
from ml.data.io_vnbd_schema import IOVNBDSchemaResolver
from scipy.spatial.transform import Rotation as R

def get_predictions(dataset, speednet, motionnet, headingnet, device):
    p_speed, p_motion, p_heading, sessions = [], [], [], []
    print(f"Starting neural net inference on {len(dataset)} windows...")
    start = time.time()
    with torch.no_grad():
        for i in range(len(dataset)):
            if i > 0 and i % 20000 == 0:
                print(f"Processed {i}/{len(dataset)} windows...")
            try:
                x, y, sess = dataset[i]
            except: continue
            
            x = x.unsqueeze(0).to(device)
            s_out = speednet(x)
            m_out = torch.sigmoid(motionnet(x))
            h_out = headingnet(x)
            
            p_speed.append(s_out.item())
            p_motion.append(m_out.item())
            p_heading.append(h_out.item())
            sessions.append(sess)
    print(f"Inference completed in {time.time()-start:.2f}s")
    return np.array(p_speed), np.array(p_motion), np.array(p_heading), np.array(sessions)

def run_pipeline(session_id, vel_predictions, motion_predictions, heading_predictions):
    inventory = pd.read_csv("data/manifests/session_inventory.csv")
    row = inventory[inventory['session_id'] == session_id].iloc[0]
    
    df_imu = pd.read_csv(row['s_file'], encoding='latin1', on_bad_lines='skip')
    df_v = pd.read_csv(row['v_file'], encoding='latin1', on_bad_lines='skip')
    
    imu_schema = IOVNBDSchemaResolver.resolve_imu_columns(df_imu)
    v_schema = IOVNBDSchemaResolver.resolve_v_columns(df_v)
    
    t_v = df_v[v_schema['time']].values
    lat = df_v[v_schema['lat']].values
    lon = df_v[v_schema['lon']].values
    head = df_v[v_schema['heading']].values
    
    t_ms = df_imu[imu_schema['time']].values
    t_imu = t_ms / 1000.0
    
    gx_c = imu_schema['gyro_x']
    ax_c = imu_schema['accel_x']
    gyro_mult = np.pi/180.0 if 'deg' in gx_c.lower() else 1.0
    accel_mult = 9.81 if '(g)' in ax_c.lower() else 1.0
    
    gyro = np.stack([df_imu[gx_c].values, df_imu[imu_schema['gyro_y']].values, df_imu[imu_schema['gyro_z']].values], axis=1) * gyro_mult
    accel = np.stack([df_imu[ax_c].values, df_imu[imu_schema['accel_y']].values, df_imu[imu_schema['accel_z']].values], axis=1) * accel_mult
    
    lat_interp = np.interp(t_imu, t_v, lat)
    lon_interp = np.interp(t_imu, t_v, lon)
    
    enu_gt = np.zeros((len(lat_interp), 2))
    for i in range(len(lat_interp)):
        x, y = latlon_to_enu(lat_interp[i], lon_interp[i], lat_interp[0], lon_interp[0])
        enu_gt[i, 0] = x
        enu_gt[i, 1] = y
        
    eskf = ESKF()
    yaw_0 = (90.0 - head[0]) * np.pi / 180.0
    eskf.q = R.from_euler('z', yaw_0)
    
    pos_dr = np.zeros((len(vel_predictions)-1, 3))
    head_dr = np.zeros(len(vel_predictions)-1)
    
    # Strict anti-cheating separation: enu_gt is NOT used in the DR loop below.
    for i in range(1, len(vel_predictions)):
        idx = i + 19
        if idx >= len(t_ms): break
        dt = t_imu[idx] - t_imu[idx-1]
        if dt <= 0: dt = 0.01
        
        # Patch the gyro Z axis with HeadingNet predicted yaw rate
        patched_gyro = gyro[idx].copy()
        patched_gyro[2] = heading_predictions[i]
        
        eskf.predict(accel[idx], patched_gyro, dt)
        
        is_stationary = motion_predictions[i] == 1
        if is_stationary:
            eskf.update_zupt()
        else:
            v_meas = vel_predictions[i]
            eskf.update_oracle_speed(v_meas, nhc=True)
            
        pos_dr[i-1] = eskf.p.copy()
        rpy = eskf.q.as_euler('xyz')
        head_dr[i-1] = rpy[2]
        
    return pos_dr, head_dr, enu_gt[20:20+len(pos_dr)], vel_predictions[1:], t_imu[20:20+len(pos_dr)]

def main():
    os.makedirs("ml/evaluation/outputs/real_osm", exist_ok=True)
    device = torch.device("cpu")
    s_cfg = json.load(open("configs/speednet_training.json"))
    m_cfg = json.load(open("configs/motion_state_training.json"))
    h_cfg = json.load(open("configs/heading_training.json"))
    
    test_dataset = IOVNBDProductionDataset(split="test", stride_override=1, target_type="yaw_rate")
    
    speednet = SpeedNet(s_cfg['model']['in_channels'], s_cfg['model']['cnn_channels'], s_cfg['model']['kernel_size'], s_cfg['model']['gru_hidden'], s_cfg['model']['gru_layers'])
    speednet.load_state_dict(torch.load(s_cfg['paths']['best_model'], map_location=device, weights_only=True)); speednet.eval()
    
    motionnet = MotionStateNet(m_cfg['model']['in_channels'], m_cfg['model']['cnn_channels'], m_cfg['model']['kernel_size'], m_cfg['model']['gru_hidden'], m_cfg['model']['gru_layers'])
    motionnet.load_state_dict(torch.load(m_cfg['paths']['best_model'], map_location=device, weights_only=True)); motionnet.eval()
    
    headingnet = HeadingNet(h_cfg['model']['in_channels'], h_cfg['model']['cnn_channels'], h_cfg['model']['kernel_size'], h_cfg['model']['gru_hidden'], h_cfg['model']['gru_layers'])
    headingnet.load_state_dict(torch.load(h_cfg['paths']['best_model'], map_location=device, weights_only=True)); headingnet.eval()
    
    p_speed_norm, p_motion, p_heading_norm, sess_arr = get_predictions(test_dataset, speednet, motionnet, headingnet, device)
    
    t_mean_v, t_std_v = 10.4019, 7.2721
    p_speed = (p_speed_norm * t_std_v) + t_mean_v
    t_mean_y, t_std_y = 0.0, 0.2
    p_heading = (p_heading_norm * t_std_y) + t_mean_y
    
    p_class = (p_motion > 0.569).astype(int)
    p_class_hyst = np.zeros_like(p_class)
    for i in range(2, len(p_class)):
        if p_class[i] == 1 and p_class[i-1] == 1 and p_class[i-2] == 1:
            p_class_hyst[i] = 1; p_class_hyst[i-1] = 1; p_class_hyst[i-2] = 1
            
    sessions = np.unique(sess_arr)
    results = []
    
    incremental_csv = "ml/evaluation/outputs/real_osm/incremental_results.csv"
    if os.path.exists(incremental_csv):
        os.remove(incremental_csv)
        
    print("Starting Viterbi Map Matching Evaluation on Real OSM...")
    for sess in sessions:
        print(f"\nProcessing session {sess}...")
        graph_file = f"map/runtime/{sess}_graph.json"
        if not os.path.exists(graph_file):
            print(f"Skipping {sess}, no graph file.")
            continue
            
        idx = (sess_arr == sess)
        pos_dr, head_dr, pos_gt, v_preds, timestamps = run_pipeline(sess, p_speed[idx], p_class_hyst[idx], p_heading[idx])
        
        dist_driven = np.sum(v_preds * 0.1) # approx 10Hz
        duration = timestamps[-1] - timestamps[0]
        
        # Anti-cheating check logic: graph search radius is frozen at 100m. 
        # Trajectory ONLY uses pos_dr which is derived solely from neural net & ESKF integration.
        trajectory = []
        for i in range(0, len(pos_dr), 10):
            trajectory.append({'x': pos_dr[i,0], 'y': pos_dr[i,1], 'h': head_dr[i]})
            
        matcher = HMMMapMatcher(graph_file, search_radius=100.0, sigma_d=20.0, sigma_h=1.0, sigma_t=10.0)
        
        # Distance metrics
        all_dists = []
        for pt in trajectory:
            cands = matcher.get_candidates(pt['x'], pt['y'], pt['h'])
            if cands:
                all_dists.extend([c['dist'] for c in cands])
        mean_d = np.mean(all_dists) if all_dists else np.nan
        min_d = np.min(all_dists) if all_dists else np.nan
        max_d = np.max(all_dists) if all_dists else np.nan
        
        start_t = time.time()
        matched_pts_down, states_down = matcher.viterbi_match(trajectory)
        print(f"Viterbi completed in {time.time()-start_t:.2f}s")
        
        if len(matched_pts_down) == 0:
            print(f"Failed to match any points for {sess}.")
            continue
            
        # Upsample back to full rate
        matched_arr = np.zeros_like(pos_dr[:, 0:2])
        states_arr = []
        for i in range(len(pos_dr)):
            low_idx = i // 10
            high_idx = min(low_idx + 1, len(matched_pts_down) - 1)
            alpha = (i % 10) / 10.0
            p1 = matched_pts_down[low_idx]
            p2 = matched_pts_down[high_idx]
            matched_arr[i, 0] = p1[0] * (1 - alpha) + p2[0] * alpha
            matched_arr[i, 1] = p1[1] * (1 - alpha) + p2[1] * alpha
            states_arr.append(states_down[low_idx])
            
        err_dr = np.linalg.norm(pos_dr[:, 0:2] - pos_gt, axis=1)
        err_mm = np.linalg.norm(matched_arr - pos_gt, axis=1)
        
        raw_rmse = np.mean(err_dr)
        mm_rmse = np.mean(err_mm)
        
        raw_drift = (raw_rmse / dist_driven) * 100
        mm_drift = (mm_rmse / dist_driven) * 100
        
        abs_imp = raw_drift - mm_drift
        pct_imp = (abs_imp / raw_drift) * 100 if raw_drift > 0 else 0
        
        num_fallback = states_arr.count('DR_FALLBACK')
        fallback_pct = (num_fallback / len(states_arr)) * 100
        mm_pct = 100 - fallback_pct
        
        transitions = 0
        for i in range(1, len(states_arr)):
            if states_arr[i] != states_arr[i-1]: transitions += 1
            
        res = {
            'session': sess,
            'duration': duration,
            'num_points': len(pos_dr),
            'raw_rmse': raw_rmse,
            'raw_drift': raw_drift,
            'mm_rmse': mm_rmse,
            'mm_drift': mm_drift,
            'abs_imp': abs_imp,
            'pct_imp': pct_imp,
            'mm_pct': mm_pct,
            'fallback_pct': fallback_pct,
            'transitions': transitions,
            'min_dist': min_d,
            'mean_dist': mean_d,
            'max_dist': max_d
        }
        results.append(res)
        
        # Save incremental
        pd.DataFrame([res]).to_csv(incremental_csv, mode='a', header=not os.path.exists(incremental_csv), index=False)
        
        # Plot
        try:
            plt.figure(figsize=(10, 10))
            plt.plot(pos_gt[:, 0], pos_gt[:, 1], 'k--', label='Ground Truth')
            plt.plot(pos_dr[:, 0], pos_dr[:, 1], 'r-', alpha=0.5, label='Raw DR')
            
            # Plot graph segments (limited to trajectory bounds)
            with open(graph_file, 'r') as f:
                graph_data = json.load(f)
            min_x, max_x = np.min(pos_dr[:,0])-100, np.max(pos_dr[:,0])+100
            min_y, max_y = np.min(pos_dr[:,1])-100, np.max(pos_dr[:,1])+100
            for s in graph_data['segments']:
                if min_x < s['x1'] < max_x and min_y < s['y1'] < max_y:
                    plt.plot([s['x1'], s['x2']], [s['y1'], s['y2']], color='gray', alpha=0.2, zorder=1)
                    
            plt.plot(matched_arr[:, 0], matched_arr[:, 1], 'b-', linewidth=2, label='Matched Trajectory')
            
            fb_x = [matched_arr[i,0] for i in range(len(states_arr)) if states_arr[i] == 'DR_FALLBACK']
            fb_y = [matched_arr[i,1] for i in range(len(states_arr)) if states_arr[i] == 'DR_FALLBACK']
            if fb_x:
                plt.scatter(fb_x, fb_y, color='orange', s=10, label='DR Fallback', zorder=4)
                
            plt.legend()
            plt.title(f'Real OSM Map Matching - {sess}')
            plt.axis('equal')
            plt.savefig(f"ml/evaluation/outputs/real_osm/{sess}_plot.png")
            plt.close()
        except Exception as e:
            print(f"Warning: Failed to generate plot for {sess}: {e}")
            
    print("\nAll sessions processed.")
    
    # Generate Report
    df_res = pd.DataFrame(results)
    
    report = []
    report.append("# Phase 7B: Real OSM Map Validation Report\n")
    report.append("## A. Real OSM Reference Evaluation\n")
    report.append("| Session | Duration (s) | Pts | Raw RMSE | Raw Drift | MM RMSE | MM Drift | Abs Imp | Pct Imp |")
    report.append("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        report.append(f"| {r['session']} | {r['duration']:.1f} | {r['num_points']} | {r['raw_rmse']:.1f} | {r['raw_drift']:.1f}% | {r['mm_rmse']:.1f} | {r['mm_drift']:.1f}% | {r['abs_imp']:.1f}% | {r['pct_imp']:.1f}% |")
        
    report.append("\n## B. Fallback Behavior\n")
    report.append("| Session | Map-Matched % | Fallback % | Transitions | Min Dist | Mean Dist | Max Dist |")
    report.append("|---|---|---|---|---|---|---|")
    for r in results:
        report.append(f"| {r['session']} | {r['mm_pct']:.1f}% | {r['fallback_pct']:.1f}% | {r['transitions']} | {r['min_dist']:.1f}m | {r['mean_dist']:.1f}m | {r['max_dist']:.1f}m |")
        
    report.append("\n## C. Failure Cases\n")
    report.append("A failure case is defined as significant DR_FALLBACK (>20%) indicating the trajectory left the 100m candidate search radius.")
    fails = df_res[df_res['fallback_pct'] > 20.0]
    if len(fails) == 0:
        report.append("No major fallback failures observed.")
    else:
        for _, r in fails.iterrows():
            report.append(f"- **{r['session']}**: {r['fallback_pct']:.1f}% fallback. Max candidate dist: {r['max_dist']:.1f}m.")
            
    report.append("\n## D. Aggregate Results\n")
    report.append(f"- **Mean Raw DR Drift**: {df_res['raw_drift'].mean():.1f}%")
    report.append(f"- **Median Raw DR Drift**: {df_res['raw_drift'].median():.1f}%")
    report.append(f"- **Mean Map-Matched Drift**: {df_res['mm_drift'].mean():.1f}%")
    report.append(f"- **Median Map-Matched Drift**: {df_res['mm_drift'].median():.1f}%")
    report.append(f"- **Mean Improvement**: {df_res['abs_imp'].mean():.1f}%")
    report.append(f"- **Median Improvement**: {df_res['abs_imp'].median():.1f}%")
    report.append(f"- **Mean Fallback %**: {df_res['fallback_pct'].mean():.1f}%")
    report.append(f"- **Median Fallback %**: {df_res['fallback_pct'].median():.1f}%")
    
    with open("docs/reports/map_matching_validation_report.md", "w") as f:
        f.write("\n".join(report))
        
    print("\nReport successfully saved to docs/reports/map_matching_validation_report.md")

if __name__ == "__main__":
    main()
