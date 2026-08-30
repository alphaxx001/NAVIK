import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from libnavik.python_reference.evaluate_baseline import latlon_to_enu

def audit_session(session_id):
    s_path = f"data/raw/IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/S-Dataset/S-{session_id}.csv"
    v_path = f"data/raw/IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/V-Dataset/V-{session_id}.csv"
    if not os.path.exists(s_path):
        s_path = s_path.replace(f"S-{session_id}", f"S-V{session_id.lower()}")
        v_path = v_path.replace(f"V-{session_id}", f"V-v{session_id.lower()}")
        
    s_df = pd.read_csv(s_path, encoding='latin1', on_bad_lines='skip')
    v_df = pd.read_csv(v_path, encoding='latin1', on_bad_lines='skip')
    s_df.columns = [c.strip() for c in s_df.columns]
    v_df.columns = [c.strip() for c in v_df.columns]
    
    gyro_z_c = [c for c in s_df.columns if 'GYROSCOPE Z' in c.upper()][0]
    t_c = [c for c in s_df.columns if 'TIME' in c.upper()][0]
    
    gyro_z = s_df[gyro_z_c].values
    heading = v_df['Heading (degrees)'].values
    yaw_rate = v_df['Yaw Rate (deg/sec)'].values
    vel = v_df['Velocity (km/hr)'].values / 3.6
    t_s = s_df[t_c].values / 1000.0
    lat = v_df['Latitude (degrees)'].values
    lon = v_df['Longitude (degrees)'].values
    
    # Analyze Heading Wrapping
    diff_heading = np.diff(heading)
    wraps = np.sum(np.abs(diff_heading) > 180)
    
    # Calculate GNSS Heading from Lat/Lon
    lat0, lon0 = lat[0], lon[0]
    enu = np.array([latlon_to_enu(lat[i], lon[i], lat0, lon0) for i in range(len(lat))])
    dx = np.diff(enu[:, 0])
    dy = np.diff(enu[:, 1])
    gnss_heading = np.degrees(np.arctan2(dx, dy))
    gnss_heading = (gnss_heading + 360) % 360
    
    # Compare Gyro Z and Yaw Rate
    # Note: Yaw Rate is in deg/s. Gyro Z is rad/s.
    gyro_z_deg = np.degrees(gyro_z)
    
    # Compute correlation
    min_len = min(len(gyro_z_deg), len(yaw_rate))
    corr = np.corrcoef(gyro_z_deg[:min_len], yaw_rate[:min_len])[0, 1]
    
    print(f"Session {session_id} Audit:")
    print(f"  Heading Wraps (>180 deg jump): {wraps}")
    print(f"  Heading Range: [{np.min(heading):.1f}, {np.max(heading):.1f}] degrees")
    print(f"  Yaw Rate Range: [{np.min(yaw_rate):.1f}, {np.max(yaw_rate):.1f}] deg/s")
    print(f"  Gyro Z (deg/s) vs Yaw Rate Correlation: {corr:.4f}")
    
    # Return data for plotting if needed
    return t_s, heading, yaw_rate, gyro_z_deg, gnss_heading, vel

def main():
    os.makedirs("docs/reports", exist_ok=True)
    os.makedirs("ml/evaluation/outputs/heading", exist_ok=True)
    
    sessions_to_audit = ["Vta10", "Vw11", "Vta2"]
    corrs = []
    
    plt.figure(figsize=(15, 10))
    for i, sess in enumerate(sessions_to_audit):
        t, h, yr, gz, gh, v = audit_session(sess)
        
        plt.subplot(3, 1, i+1)
        plt.plot(t[:2000], yr[:2000], label='Vehicle Yaw Rate (deg/s)', alpha=0.7)
        plt.plot(t[:2000], gz[:2000], label='Smartphone Gyro Z (deg/s)', alpha=0.7)
        plt.title(f"Yaw Rate vs Gyro Z - Session {sess}")
        plt.legend()
        plt.grid()
    
    plt.tight_layout()
    plt.savefig("ml/evaluation/outputs/heading/supervision_audit.png")
    
if __name__ == "__main__":
    main()
