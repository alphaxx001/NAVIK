import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.run_navigation_pipeline import load_models, extract_windows, infer_networks
from libnavik.python_reference.eskf import ESKF
from libnavik.python_reference.evaluate_baseline import latlon_to_enu
from ml.data.dataset_loader import IOVNBDProductionDataset
from ml.data.io_vnbd_schema import IOVNBDSchemaResolver
from scripts.test_kdtree_equivalence import KDTreeHMMMapMatcher

def generate_demo_outputs(session_id="Vta1a"):
    """
    Runs the frozen IDR-System pipeline but simulates realistic GNSS availability
    to demonstrate navigation mode transitions for presentation purposes.
    
    Mode Timeline (Example Vta1a):
    0 -> 500s: GNSS Mode (GNSS Available, high confidence)
    500s -> 2000s: Blackout (DR -> MAP_CONTEXT -> DR_FALLBACK)
    2000s -> end: GNSS Restored
    """
    print(f"Initializing Presentation Demo Pipeline on {session_id}...")
    device = torch.device("cpu")
    
    # 1. Load Data
    demo_file = "data/demo/replay_session.csv"
    use_demo = False
    
    if os.path.exists("data/manifests/session_inventory.csv"):
        inventory = pd.read_csv("data/manifests/session_inventory.csv")
        rows = inventory[inventory['session_id'] == session_id]
        if len(rows) > 0 and os.path.exists(rows.iloc[0]['s_file']):
            row = rows.iloc[0]
            df_imu = pd.read_csv(row['s_file'], encoding='latin1', on_bad_lines='skip')
            df_v = pd.read_csv(row['v_file'], encoding='latin1', on_bad_lines='skip')
        else:
            use_demo = True
    else:
        use_demo = True

    if use_demo:
        if not os.path.exists(demo_file):
            raise FileNotFoundError(f"Neither raw session '{session_id}' nor demo file '{demo_file}' found.")
        print(f"Using demo replay session: {demo_file}")
        df_imu = pd.read_csv(demo_file)
        df_v = df_imu
    
    imu_schema = IOVNBDSchemaResolver.resolve_imu_columns(df_imu)
    v_schema = IOVNBDSchemaResolver.resolve_v_columns(df_v)
    
    lat = df_v[v_schema['lat']].values.astype(float)
    lon = df_v[v_schema['lon']].values.astype(float)
    head = df_v[v_schema['heading']].values.astype(float)
    vel_gt = df_v[v_schema['velocity']].values.astype(float) if v_schema['velocity'] in df_v else np.zeros(len(df_v))
    t_v = df_v[v_schema['time']].values.astype(float)
    
    # Align clocks
    t_v_rel = t_v - t_v[0]
    
    # 2. Neural Inference
    speednet, motionnet, headingnet = load_models(device)
    inputs, t_pred_ms, gyro, accel, t_ms = extract_windows(df_imu, imu_schema)
    p_speed, p_class, p_heading = infer_networks(inputs, speednet, motionnet, headingnet, device)
    
    # 3. Simulate ESKF continuously 
    eskf = ESKF()
    lat0, lon0 = lat[0], lon[0]
    yaw_0 = (90.0 - head[0]) * np.pi / 180.0
    eskf.q = R.from_euler('z', yaw_0)
    
    t_imu = t_ms / 1000.0
    pos_dr = np.zeros((len(p_speed), 3))
    head_dr = np.zeros(len(p_speed))
    vel_dr = np.zeros((len(p_speed), 3))
    timestamps = np.zeros(len(p_speed))
    
    # Define GNSS Outage Bounds
    total_dur = t_v_rel[-1] if len(t_v_rel) > 0 else 300.0
    if total_dur > 1000:
        blackout_start = 500.0
        blackout_end = 2000.0
    else:
        blackout_start = 30.0
        blackout_end = min(240.0, total_dur * 0.8)
    
    print(f"Propagating Authoritative Inertial State (Blackout: {blackout_start}s to {blackout_end}s)...")
    for i in range(1, len(p_speed)):
        idx = i + 19
        if idx >= len(t_ms): break
        dt = t_imu[idx] - t_imu[idx-1]
        if dt <= 0: dt = 0.01
        
        # Simulate GNSS reset at restoration point
        current_t = t_imu[idx] - t_imu[0]
        if i > 1 and current_t >= blackout_end and (t_imu[idx-1] - t_imu[0]) < blackout_end:
            # Re-initialize to GT (GNSS restored)
            gt_idx = np.argmin(np.abs(t_v_rel - current_t))
            e_g, n_g = latlon_to_enu(lat[gt_idx], lon[gt_idx], lat0, lon0)
            eskf.p = np.array([e_g, n_g, 0.0])
            yaw_gt = (90.0 - head[gt_idx]) * np.pi / 180.0
            eskf.q = R.from_euler('z', yaw_gt)
            
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
        
    # Relative time array for plotting
    t_rel = np.zeros_like(timestamps)
    valid = timestamps > 0
    t_rel[valid] = (timestamps[valid] - timestamps[valid][0]) / 1000.0
    
    # 4. Open-Loop Map Context (Only executed during blackout theoretically, but we run over the whole set)
    print("Applying Map Context layer...")
    graph_file = f"map/runtime/{session_id}_graph.json"
    matcher = KDTreeHMMMapMatcher(graph_file, search_radius=100.0)
    
    trajectory = []
    for i in range(0, len(p_speed), 10):
        if not valid[i]: continue
        trajectory.append({'x': pos_dr[i, 0], 'y': pos_dr[i, 1], 'h': head_dr[i]})
        
    matched_pts_down, states_down = matcher.viterbi_match(trajectory)
    
    # Upsample
    pos_map = np.zeros_like(pos_dr[:, 0:2])
    states_full = ['DR'] * len(p_speed)
    for i in range(len(p_speed)):
        if not valid[i]: continue
        low_idx = i // 10
        high_idx = min(low_idx + 1, len(matched_pts_down) - 1)
        if low_idx >= len(matched_pts_down): low_idx = len(matched_pts_down) - 1
        alpha = (i % 10) / 10.0
        
        p1 = matched_pts_down[low_idx]; p2 = matched_pts_down[high_idx]
        pos_map[i, 0] = p1[0] * (1 - alpha) + p2[0] * alpha
        pos_map[i, 1] = p1[1] * (1 - alpha) + p2[1] * alpha
        states_full[i] = 'DR_FALLBACK' if states_down[low_idx] == 'DR_FALLBACK' else 'MAP_CONTEXT'
        
    # 5. Assemble final demo presentation data
    final_x = np.zeros(len(p_speed))
    final_y = np.zeros(len(p_speed))
    modes = []
    
    for i in range(len(p_speed)):
        if not valid[i]:
            modes.append("OFF")
            continue
            
        cur_t = t_rel[i]
        if cur_t < blackout_start or cur_t > blackout_end:
            # GNSS Mode
            gt_idx = np.argmin(np.abs(t_v_rel - cur_t))
            e_g, n_g = latlon_to_enu(lat[gt_idx], lon[gt_idx], lat0, lon0)
            final_x[i] = e_g
            final_y[i] = n_g
            modes.append("GNSS")
        else:
            # Blackout Mode
            final_x[i] = pos_map[i, 0]
            final_y[i] = pos_map[i, 1]
            modes.append(states_full[i])
            
    # 6. Presentation Plotting
    os.makedirs("docs/reports", exist_ok=True)
    print("Generating Presentation Graphs...")
    
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, height_ratios=[3, 1, 1])
    
    # --- Trajectory Map ---
    ax_map = fig.add_subplot(gs[0, :])
    enu_gt = np.zeros((len(lat), 2))
    for i in range(len(lat)):
        x, y = latlon_to_enu(lat[i], lon[i], lat0, lon0)
        enu_gt[i, 0] = x; enu_gt[i, 1] = y
        
    ax_map.plot(enu_gt[:, 0], enu_gt[:, 1], 'k--', label="True Path", alpha=0.5)
    
    # Plot segments by mode
    gnss_mask = np.array(modes) == "GNSS"
    map_mask = np.array(modes) == "MAP_CONTEXT"
    fb_mask = np.array(modes) == "DR_FALLBACK"
    
    ax_map.scatter(final_x[gnss_mask], final_y[gnss_mask], c='green', s=1, label="GNSS (Active)")
    ax_map.scatter(final_x[map_mask], final_y[map_mask], c='blue', s=1, label="Map Context (Blackout)")
    ax_map.scatter(final_x[fb_mask], final_y[fb_mask], c='orange', s=1, label="DR Fallback (Blackout)")
    
    ax_map.set_title("End-to-End Navigation System Demo", fontsize=16)
    ax_map.legend()
    ax_map.axis('equal')
    
    # --- Mode Timeline ---
    ax_time = fig.add_subplot(gs[1, :])
    mode_vals = []
    for m in modes:
        if m == "GNSS": mode_vals.append(3)
        elif m == "MAP_CONTEXT": mode_vals.append(2)
        elif m == "DR_FALLBACK": mode_vals.append(1)
        else: mode_vals.append(0)
        
    ax_time.plot(t_rel[valid], np.array(mode_vals)[valid], 'k-', lw=2)
    ax_time.fill_between(t_rel[valid], 0, np.array(mode_vals)[valid], alpha=0.3, color='gray')
    ax_time.set_yticks([1, 2, 3])
    ax_time.set_yticklabels(['DR Fallback', 'Map Context', 'GNSS Acquired'])
    ax_time.set_title("System State Timeline", fontsize=12)
    
    # --- Kinematics ---
    ax_vel = fig.add_subplot(gs[2, 0])
    ax_vel.plot(t_v_rel, vel_gt, 'k--', label="True Speed")
    ax_vel.plot(t_rel[valid], np.linalg.norm(vel_dr[valid, 0:2], axis=1), 'b-', label="SpeedNet/ESKF Speed")
    ax_vel.set_title("Longitudinal Velocity")
    ax_vel.legend()
    
    ax_head = fig.add_subplot(gs[2, 1])
    ax_head.plot(t_v_rel, head, 'k--', label="True Heading")
    ax_head.plot(t_rel[valid], (90.0 - head_dr[valid] * 180.0 / np.pi) % 360.0, 'r-', label="HeadingNet/ESKF Yaw")
    ax_head.set_title("Vehicle Heading")
    ax_head.legend()
    
    plt.tight_layout()
    plt.savefig("docs/reports/final_demo_dashboard.png", dpi=150)
    print("Demo execution complete. Dashboard saved to docs/reports/final_demo_dashboard.png")

if __name__ == "__main__":
    generate_demo_outputs()
