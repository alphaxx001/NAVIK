import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml.data.io_vnbd_schema import IOVNBDSchemaResolver

def generate_replay():
    session_id = "Vta1a"
    inventory = pd.read_csv("data/manifests/session_inventory.csv")
    row = inventory[inventory['session_id'] == session_id].iloc[0]
    
    df_imu = pd.read_csv(row['s_file'], encoding='latin1', on_bad_lines='skip')
    df_v = pd.read_csv(row['v_file'], encoding='latin1', on_bad_lines='skip')
    
    imu_schema = IOVNBDSchemaResolver.resolve_imu_columns(df_imu)
    v_schema = IOVNBDSchemaResolver.resolve_v_columns(df_v)
    
    t_ms = df_imu[imu_schema['time']].values
    gx = df_imu[imu_schema['gyro_x']].values; gy = df_imu[imu_schema['gyro_y']].values; gz = df_imu[imu_schema['gyro_z']].values
    ax = df_imu[imu_schema['accel_x']].values; ay = df_imu[imu_schema['accel_y']].values; az = df_imu[imu_schema['accel_z']].values
    
    g_m = np.pi/180.0 if 'deg' in imu_schema['gyro_x'].lower() else 1.0
    a_m = 9.81 if '(g)' in imu_schema['accel_x'].lower() else 1.0
    
    gyro = np.stack([gx, gy, gz], axis=1) * g_m
    accel = np.stack([ax, ay, az], axis=1) * a_m
    t_imu = t_ms / 1000.0
    
    # Ground truth
    lat = df_v[v_schema['lat']].values
    lon = df_v[v_schema['lon']].values
    head = df_v[v_schema['heading']].values
    t_v = df_v[v_schema['time']].values
    t_v_rel = t_v - t_v[0]
    
    # We will interpolate GNSS to match IMU timestamps to create a single synchronized offline file
    t_rel = t_imu - t_imu[0]
    lat_interp = np.interp(t_rel, t_v_rel, lat)
    lon_interp = np.interp(t_rel, t_v_rel, lon)
    head_interp = np.interp(t_rel, t_v_rel, head)
    
    # We only need the first 3000 samples (~5 minutes at 10Hz) for the mobile app demo prototype to keep APK size small
    limit = min(3000, len(t_imu))
    
    out_df = pd.DataFrame({
        'timestamp': t_imu[:limit],
        'ax': accel[:limit, 0],
        'ay': accel[:limit, 1],
        'az': accel[:limit, 2],
        'gx': gyro[:limit, 0],
        'gy': gyro[:limit, 1],
        'gz': gyro[:limit, 2],
        'gnss_lat': lat_interp[:limit],
        'gnss_lon': lon_interp[:limit],
        'gnss_head': head_interp[:limit]
    })
    
    out_dir = "Mobile_App/app/src/main/assets/demo"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "replay_session.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Generated mobile replay artifact at {out_path} ({limit} samples)")

if __name__ == "__main__":
    generate_replay()
