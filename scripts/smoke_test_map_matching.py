import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from ml.data.io_vnbd_schema import IOVNBDSchemaResolver
from libnavik.python_reference.eskf import ESKF
from libnavik.python_reference.evaluate_baseline import latlon_to_enu
from scipy.spatial.transform import Rotation as R
from map.runtime.hmm_matcher import HMMMapMatcher

def run_smoke_test():
    print("--- STARTING SMOKE TEST ---")
    session_id = 'Vta10'
    max_pts = 50
    
    # 1. Verification of OSM Data Loading
    print("Checking REAL OSM Graph...")
    graph_file = f"map/runtime/{session_id}_graph.json"
    pbf_file = "data/raw/OSM/uk_midlands.osm.pbf"
    
    if not os.path.exists(pbf_file):
        print("FAIL: Original PBF missing.")
        return
    print(f"PBF exists: {pbf_file}")
    
    if not os.path.exists(graph_file):
        print(f"FAIL: Graph file missing ({graph_file})")
        return
        
    with open(graph_file, 'r') as f:
        graph_data = json.load(f)
        
    segments = graph_data.get('segments', [])
    print(f"Graph loaded successfully. Contains {len(segments)} road segments.")
    if len(segments) == 0:
        print("FAIL: Graph has no segments.")
        return
        
    # 2. Robust CSV Loading
    print("\nLoading CSV Data...")
    inventory = pd.read_csv("data/manifests/session_inventory.csv")
    row = inventory[inventory['session_id'] == session_id].iloc[0]
    
    s_path = row['s_file']
    v_path = row['v_file']
    
    df_imu = pd.read_csv(s_path, encoding='latin1', on_bad_lines='skip')
    df_v = pd.read_csv(v_path, encoding='latin1', on_bad_lines='skip')
    
    # Resolve schemas
    try:
        imu_schema = IOVNBDSchemaResolver.resolve_imu_columns(df_imu)
        v_schema = IOVNBDSchemaResolver.resolve_v_columns(df_v)
        print("Schema successfully resolved.")
    except Exception as e:
        print(f"FAIL: Schema resolution error: {e}")
        return
        
    # 3. Coordinate Conversion & DR integration
    t_v = df_v[v_schema['time']].values
    lat = df_v[v_schema['lat']].values
    lon = df_v[v_schema['lon']].values
    vel = df_v[v_schema['velocity']].values / 3.6
    heading = df_v[v_schema['heading']].values
    
    t_ms = df_imu[imu_schema['time']].values
    t_imu = t_ms / 1000.0
    
    # Unit conversions
    gx_c = imu_schema['gyro_x']
    ax_c = imu_schema['accel_x']
    gyro_mult = np.pi/180.0 if 'deg' in gx_c.lower() else 1.0
    accel_mult = 9.81 if '(g)' in ax_c.lower() else 1.0
    
    gyro = np.stack([df_imu[gx_c].values, df_imu[imu_schema['gyro_y']].values, df_imu[imu_schema['gyro_z']].values], axis=1) * gyro_mult
    accel = np.stack([df_imu[ax_c].values, df_imu[imu_schema['accel_y']].values, df_imu[imu_schema['accel_z']].values], axis=1) * accel_mult
    
    print("Integrating DR trajectory (Oracle Speed/Yaw for smoke test)...")
    eskf = ESKF()
    yaw_0 = (90.0 - heading[0]) * np.pi / 180.0
    eskf.q = R.from_euler('z', yaw_0)
    eskf.v = eskf.q.as_matrix() @ np.array([0, vel[0], 0])
    
    pos_est = []
    head_est = []
    
    for i in range(1, min(max_pts+1, len(t_ms))):
        dt = t_imu[i] - t_imu[i-1]
        if dt <= 0: dt = 0.1
        
        eskf.predict(accel[i], gyro[i], dt)
        
        # Oracle update
        v_idx = min(int(i / 10), len(vel)-1) # Approx 10Hz to 100Hz
        v_val = vel[v_idx]
        if v_val > 0.1:
            eskf.update_oracle_speed(v_val, nhc=True)
        else:
            eskf.update_zupt()
            
        pos_est.append(eskf.p.copy())
        rpy = eskf.q.as_euler('xyz')
        head_est.append(rpy[2])
        
    print(f"Generated {len(pos_est)} trajectory points.")
    
    # Format for matcher
    trajectory = []
    for i in range(len(pos_est)):
        trajectory.append({'x': pos_est[i][0], 'y': pos_est[i][1], 'h': head_est[i]})
        
    # 4. Map Matching
    print("\nRunning HMM/Viterbi Map Matcher...")
    matcher = HMMMapMatcher(graph_file, search_radius=100.0)
    
    cand_counts = []
    min_dists = []
    max_dists = []
    
    # Inspect candidate generation
    for pt in trajectory:
        cands = matcher.get_candidates(pt['x'], pt['y'], pt['h'])
        cand_counts.append(len(cands))
        if cands:
            dists = [c['dist'] for c in cands]
            min_dists.append(min(dists))
            max_dists.append(max(dists))
            
    print(f"Candidate counts per point: {cand_counts}")
    if min_dists:
        print(f"Minimum candidate distance across all points: {min(min_dists):.2f}m")
        print(f"Maximum candidate distance across all points: {max(max_dists):.2f}m")
    
    if all(c == 0 for c in cand_counts):
        print("FAIL: Zero candidates found across all trajectory points!")
        print(f"Trajectory bounds: X:[{min(pt['x'] for pt in trajectory):.1f}, {max(pt['x'] for pt in trajectory):.1f}], Y:[{min(pt['y'] for pt in trajectory):.1f}, {max(pt['y'] for pt in trajectory):.1f}]")
        print(f"Graph bounds: X:[{min(min(s['x1'], s['x2']) for s in segments):.1f}, {max(max(s['x1'], s['x2']) for s in segments):.1f}]")
        return
        
    start_t = time.time()
    matched_pts, states = matcher.viterbi_match(trajectory)
    rt = time.time() - start_t
    
    print(f"Viterbi completed in {rt:.4f} seconds.")
    print(f"Output matched points length: {len(matched_pts)}")
    print(f"Fallback states: {states.count('DR_FALLBACK')}/{len(states)}")
    
    # 5. Output Generation
    try:
        plt.figure(figsize=(8,8))
        pos_arr = np.array(pos_est)
        m_arr = np.array(matched_pts)
        
        for s in segments:
            # only plot segments near trajectory to keep plot light
            if min(pos_arr[:,0])-100 < s['x1'] < max(pos_arr[:,0])+100:
                plt.plot([s['x1'], s['x2']], [s['y1'], s['y2']], color='gray', alpha=0.3, zorder=1)
                
        plt.plot(pos_arr[:,0], pos_arr[:,1], 'r-', label='DR Trajectory', zorder=2)
        
        if len(m_arr) > 0:
            plt.plot(m_arr[:,0], m_arr[:,1], 'b-', linewidth=2, label='Matched Trajectory', zorder=3)
            
        plt.legend()
        plt.title(f"Smoke Test: {session_id}")
        plt.axis('equal')
        plt.savefig(f"scripts/smoke_test_{session_id}.png")
        print("Saved visualization to scripts/smoke_test_Vta10.png")
    except Exception as e:
        print(f"Warning: plotting failed: {e}")
        
    print("\n--- SMOKE TEST SUCCESSFUL ---")

if __name__ == "__main__":
    run_smoke_test()
