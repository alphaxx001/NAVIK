import os
import sys
import pandas as pd
import numpy as np
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Edge_Engine.runtime.engine import EdgeEngine
from ml.data.io_vnbd_schema import IOVNBDSchemaResolver

def run_replay(session_id="Vta1a"):
    print(f"--- EDGE ENGINE REPLAY: {session_id} ---")
    
    inventory = pd.read_csv("data/manifests/session_inventory.csv")
    row = inventory[inventory['session_id'] == session_id].iloc[0]
    df_imu = pd.read_csv(row['s_file'], encoding='latin1', on_bad_lines='skip')
    df_v = pd.read_csv(row['v_file'], encoding='latin1', on_bad_lines='skip')
    
    imu_schema = IOVNBDSchemaResolver.resolve_imu_columns(df_imu)
    v_schema = IOVNBDSchemaResolver.resolve_v_columns(df_v)
    
    lat = df_v[v_schema['lat']].values
    lon = df_v[v_schema['lon']].values
    head = df_v[v_schema['heading']].values
    
    t_ms = df_imu[imu_schema['time']].values
    gx = df_imu[imu_schema['gyro_x']].values; gy = df_imu[imu_schema['gyro_y']].values; gz = df_imu[imu_schema['gyro_z']].values
    ax = df_imu[imu_schema['accel_x']].values; ay = df_imu[imu_schema['accel_y']].values; az = df_imu[imu_schema['accel_z']].values
    
    g_m = np.pi/180.0 if 'deg' in imu_schema['gyro_x'].lower() else 1.0
    a_m = 9.81 if '(g)' in imu_schema['accel_x'].lower() else 1.0
    
    gyro = np.stack([gx, gy, gz], axis=1) * g_m
    accel = np.stack([ax, ay, az], axis=1) * a_m
    
    t_imu = t_ms / 1000.0
    
    engine = EdgeEngine(f"map/runtime/{session_id}_graph.json")
    
    # Initialize GNSS
    engine.initialize_gnss(lat[0], lon[0], head[0], t_imu[0])
    
    outputs = []
    
    t0 = time.time()
    t_inference_accum = 0
    inference_count = 0
    
    # Simulate GNSS Blackout
    blackout_start = 500.0
    blackout_end = 2000.0
    
    # Pre-warm buffer
    for i in range(19):
        engine.sensor_buffer.add_sample(t_imu[i], accel[i], gyro[i])
        engine.sample_count += 1
        
    for i in range(19, len(t_imu)):
        cur_t_rel = t_imu[i] - t_imu[0]
        gnss_avail = not (blackout_start <= cur_t_rel <= blackout_end)
        
        # Simulate edge stepping
        ti_start = time.time()
        engine.step_imu(t_imu[i], accel[i], gyro[i])
        
        # If it was an inference step
        if engine.sample_count % engine.inference_freq == 0:
            out = engine.generate_output(gnss_available=gnss_avail)
            outputs.append(out)
            inference_count += 1
            t_inference_accum += (time.time() - ti_start)
            
        # Optional: Print progress
        if i % 10000 == 0:
            print(f"Processed {i}/{len(t_imu)} samples...")
            
    t_total = time.time() - t0
    print("\n--- REPLAY COMPLETE ---")
    print(f"Total Replay Time: {t_total:.2f}s")
    if inference_count > 0:
        print(f"Avg Inference Step Time: {(t_inference_accum / inference_count) * 1000:.2f}ms")
    
    # Output to CSV
    records = []
    for o in outputs:
        records.append({
            'timestamp': o.timestamp,
            'mode': o.mode,
            'latitude': o.latitude,
            'longitude': o.longitude,
            'speed_mps': o.speed_mps,
            'map_candidate_count': o.map_candidate_count
        })
        
    df_out = pd.DataFrame(records)
    os.makedirs("Edge_Engine/tests/outputs", exist_ok=True)
    df_out.to_csv(f"Edge_Engine/tests/outputs/{session_id}_edge_replay.csv", index=False)
    print("Replay log saved.")

if __name__ == "__main__":
    run_replay()
