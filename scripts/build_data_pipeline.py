import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    root_dir = r"data/raw/IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset"
    s_dir = os.path.join(root_dir, "S-Dataset")
    v_dir = os.path.join(root_dir, "V-Dataset")
    
    s_files = glob.glob(os.path.join(s_dir, "*.csv"))
    v_files = glob.glob(os.path.join(v_dir, "*.csv"))
    
    # Match files by lowercase session id
    v_dict = {os.path.basename(f).lower().replace('v-', '').replace('.csv', ''): f for f in v_files}
    matched_pairs = []
    for s_path in s_files:
        session_id_lower = os.path.basename(s_path).lower().replace('s-', '').replace('.csv', '')
        if session_id_lower in v_dict:
            matched_pairs.append((s_path, v_dict[session_id_lower]))

    
    inventory = []
    data_quality = {
        "anomalies": [],
        "sampling_rates": [],
        "sessions": {}
    }
    
    train_sessions = []
    val_sessions = []
    test_sessions = []
    
    train_features = []
    train_targets = []
    
    output_dir = "data/manifests"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("ml/data/outputs", exist_ok=True)
    
    print(f"Discovered {len(s_files)} S-files and {len(v_files)} V-files.")
    print(f"Matched {len(matched_pairs)} pairs.")
    
    # Process each pair
    valid_sessions = []
    
    for s_path, v_path in matched_pairs:
        # Re-derive standard session ID from s_path (keeping original casing)
        session_id = os.path.basename(s_path).replace("S-", "").replace("s-", "").replace(".csv", "")
        
        try:
            # latin1 avoids UnicodeDecodeError for superscript/degree symbols
            s_df = pd.read_csv(s_path, encoding='latin1', on_bad_lines='skip')
            v_df = pd.read_csv(v_path, encoding='latin1', on_bad_lines='skip')
            
            # Clean headers (strip whitespace)
            s_df.columns = [c.strip() for c in s_df.columns]
            v_df.columns = [c.strip() for c in v_df.columns]
            
            s_rows, v_rows = len(s_df), len(v_df)
            synced = (s_rows == v_rows)
            
            if not synced:
                data_quality["anomalies"].append({
                    "session": session_id,
                    "issue": "Row count mismatch",
                    "s_rows": s_rows,
                    "v_rows": v_rows
                })
                continue # Skip unmatched sessions for now to maintain pure synchronization
                
            # Time column analysis
            # IO-VNBD S-dataset typically uses "TIME SINCE START (ms)"
            time_cols = [c for c in s_df.columns if "TIME" in c.upper() or "ms" in c.lower()]
            s_time_col = time_cols[0] if time_cols else None
            
            dt_mean, dt_std, hz = 0.0, 0.0, 0.0
            if s_time_col and s_time_col in s_df.columns:
                t_ms = s_df[s_time_col].values
                dt_array = np.diff(t_ms)
                # filter out negative or zero dt (if any)
                dt_array = dt_array[dt_array > 0]
                if len(dt_array) > 0:
                    dt_mean = np.mean(dt_array)
                    dt_std = np.std(dt_array)
                    hz = 1000.0 / dt_mean if dt_mean > 0 else 0
                    data_quality["sampling_rates"].append(hz)
            
            # Velocity column from V-dataset
            vel_cols = [c for c in v_df.columns if "velocity" in c.lower() or "speed" in c.lower()]
            vel_col = None
            for c in vel_cols:
                if "km/hr" in c.lower() and "velocity" in c.lower():
                    vel_col = c
                    break
            if not vel_col and vel_cols: vel_col = vel_cols[0]
            
            # Convert velocity km/h -> m/s
            if vel_col:
                v_df['velocity_m_s'] = v_df[vel_col] / 3.6
            
            # Identify IMU columns
            accel_cols = [c for c in s_df.columns if "ACCELEROMETER" in c.upper()]
            gyro_cols = [c for c in s_df.columns if "GYROSCOPE" in c.upper()]
            
            has_nan = s_df[accel_cols + gyro_cols].isna().any().any() or v_df['velocity_m_s'].isna().any()
            if has_nan:
                # Interpolate or drop? We will interpolate linearly for small gaps
                s_df[accel_cols + gyro_cols] = s_df[accel_cols + gyro_cols].interpolate(method='linear').bfill().ffill()
                v_df['velocity_m_s'] = v_df['velocity_m_s'].interpolate(method='linear').bfill().ffill()
            
            valid_sessions.append(session_id)
            
            inventory.append({
                "session_id": session_id,
                "s_file": s_path,
                "v_file": v_path,
                "rows": s_rows,
                "duration_s": (s_rows * dt_mean) / 1000.0 if dt_mean > 0 else 0,
                "dt_mean_ms": dt_mean,
                "dt_std_ms": dt_std,
                "hz": hz,
                "status": "VALID"
            })
            
            data_quality["sessions"][session_id] = {
                "dt_mean": dt_mean,
                "dt_std": dt_std,
                "hz": hz,
                "accel_cols": accel_cols,
                "gyro_cols": gyro_cols,
                "vel_col_raw": vel_col,
                "has_nan_imputed": bool(has_nan)
            }
            
            # Train / Val / Test split based on stable hashing or simple deterministic split
            # 70% train, 15% val, 15% test
            hash_val = int(session_id.replace('S','')) if session_id.replace('S','').isdigit() else len(valid_sessions)
            if hash_val % 10 < 7:
                train_sessions.append(session_id)
                # Accumulate for normalization
                train_features.append(s_df[accel_cols + gyro_cols].values)
                train_targets.append(v_df['velocity_m_s'].values)
            elif hash_val % 10 < 8:
                val_sessions.append(session_id)
            else:
                test_sessions.append(session_id)
                
            # Plot one session for sanity check (e.g. S1)
            if session_id == "S1" or len(valid_sessions) == 1:
                plt.figure(figsize=(15, 10))
                plt.subplot(3, 1, 1)
                plt.plot(s_df[accel_cols])
                plt.title(f"Session {session_id} Accelerometer (m/s²)")
                plt.legend(accel_cols, loc='upper right')
                
                plt.subplot(3, 1, 2)
                plt.plot(s_df[gyro_cols])
                plt.title(f"Session {session_id} Gyroscope (rad/s)")
                plt.legend(gyro_cols, loc='upper right')
                
                plt.subplot(3, 1, 3)
                plt.plot(v_df['velocity_m_s'], label="Velocity (m/s)", color='red')
                plt.title(f"Session {session_id} Ground Truth Velocity")
                plt.legend()
                
                plt.tight_layout()
                plt.savefig(f"ml/data/outputs/sanity_check_{session_id}.png")
                plt.close()
                
        except Exception as e:
            inventory.append({
                "session_id": session_id,
                "status": f"ERROR: {str(e)}"
            })
            data_quality["anomalies"].append({"session": session_id, "error": str(e)})

    # Calculate Normalization Stats
    if train_features:
        all_train_features = np.vstack(train_features)
        all_train_targets = np.concatenate(train_targets)
        
        feature_mean = np.mean(all_train_features, axis=0).tolist()
        feature_std = np.std(all_train_features, axis=0).tolist()
        # Avoid division by zero
        feature_std = [s if s > 1e-6 else 1.0 for s in feature_std]
        
        target_mean = np.mean(all_train_targets)
        target_std = np.std(all_train_targets)
        target_std = target_std if target_std > 1e-6 else 1.0
        
        normalization = {
            "version": "1.0",
            "features": {
                "names": accel_cols + gyro_cols,
                "mean": feature_mean,
                "std": feature_std,
                "units": ["m/s2"]*3 + ["rad/s"]*3
            },
            "targets": {
                "name": "velocity_m_s",
                "mean": float(target_mean),
                "std": float(target_std),
                "unit": "m/s"
            }
        }
        with open("data/manifests/normalization.json", "w") as f:
            json.dump(normalization, f, indent=4)
            
    # Save manifests
    pd.DataFrame(inventory).to_csv("data/manifests/session_inventory.csv", index=False)
    with open("data/manifests/data_quality.json", "w") as f:
        json.dump(data_quality, f, indent=4)
        
    with open("data/manifests/train_sessions.json", "w") as f: json.dump(train_sessions, f)
    with open("data/manifests/val_sessions.json", "w") as f: json.dump(val_sessions, f)
    with open("data/manifests/test_sessions.json", "w") as f: json.dump(test_sessions, f)
    
    print(f"Total processed: {len(valid_sessions)}")
    print(f"Train: {len(train_sessions)} | Val: {len(val_sessions)} | Test: {len(test_sessions)}")
    if len(data_quality['sampling_rates']) > 0:
        print(f"Mean sampling rate: {np.mean(data_quality['sampling_rates']):.2f} Hz")

if __name__ == "__main__":
    main()
