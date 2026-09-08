import argparse
import glob
import hashlib
import json
import os
import re
import matplotlib
matplotlib.use('Agg')  # Ensure headless rendering without X11/display requirement
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build data pipeline manifests and normalization statistics for IO-VNBD dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Root raw dataset directory containing S-Dataset and V-Dataset (auto-discovered if omitted)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data_pipeline.json",
        help="Path to pipeline configuration JSON (default: configs/data_pipeline.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/manifests",
        help="Output manifests directory (default: data/manifests)",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="ml/data/outputs",
        help="Output directory for sanity check plots (default: ml/data/outputs)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite manifests even if 0 valid sessions are found",
    )
    return parser.parse_args()


def find_dataset_root(cli_data_dir, config_path):
    """
    Search for dataset root directory using CLI arg, config file, and known candidate paths.
    Resilient to typographical variations ('abd' vs 'and') and nested directories.
    """
    candidates = []
    if cli_data_dir:
        candidates.append(cli_data_dir)

    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                cfg_root = cfg.get("dataset", {}).get("root_dir")
                if cfg_root:
                    candidates.append(cfg_root)
        except Exception:
            pass

    # Standard candidate paths in repo
    candidates.extend([
        r"data/raw/IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset",
        r"data/raw/IO-VNBD/Synchronised V and S datasets/Uncategorised IOVNB Dataset",
        r"data/raw/IO-VNBD/Uncategorised IOVNB Dataset",
        r"data/raw/IO-VNBD",
        r"data/raw",
    ])

    for cand in candidates:
        if not cand or not os.path.exists(cand):
            continue

        s_cand = os.path.join(cand, "S-Dataset")
        v_cand = os.path.join(cand, "V-Dataset")
        if os.path.isdir(s_cand) and os.path.isdir(v_cand):
            return cand

        # Case-insensitive check for child directories
        try:
            subdirs = {name.lower(): os.path.join(cand, name) for name in os.listdir(cand) if os.path.isdir(os.path.join(cand, name))}
            if "s-dataset" in subdirs and "v-dataset" in subdirs:
                return cand
        except Exception:
            pass

    # Recursive check under data/raw if available
    if os.path.isdir("data/raw"):
        for root, dirs, _ in os.walk("data/raw"):
            lower_dirs = {d.lower(): d for d in dirs}
            if "s-dataset" in lower_dirs and "v-dataset" in lower_dirs:
                return root

    return None


def get_dataset_subdirs(root_dir):
    """Locate S-Dataset and V-Dataset directories inside root_dir case-insensitively."""
    s_dir = os.path.join(root_dir, "S-Dataset")
    v_dir = os.path.join(root_dir, "V-Dataset")

    if not os.path.isdir(s_dir) or not os.path.isdir(v_dir):
        for name in os.listdir(root_dir):
            full = os.path.join(root_dir, name)
            if os.path.isdir(full):
                if name.lower() == "s-dataset":
                    s_dir = full
                elif name.lower() == "v-dataset":
                    v_dir = full

    return s_dir, v_dir


def main():
    args = parse_args()

    root_dir = find_dataset_root(args.data_dir, args.config)
    
    inventory = []
    data_quality = {
        "anomalies": [],
        "sampling_rates": [],
        "sessions": {}
    }
    
    train_sessions = []
    val_sessions = []
    test_sessions = []

    # Streaming statistics accumulator to avoid high-RAM vstack
    train_samples_count = 0
    train_feat_sum = None
    train_feat_sq_sum = None
    train_tgt_sum = 0.0
    train_tgt_sq_sum = 0.0
    canonical_features = None

    valid_sessions = []

    if not root_dir or not os.path.exists(root_dir):
        print(f"[WARNING] Raw IO-VNBD dataset directory not found.")
        print(f"Searched candidate locations including 'data/raw/IO-VNBD'.")
        if not args.force:
            print(f"[GUARD] Existing manifests in '{args.output_dir}' preserved (no overwrite).")
            print(f"Run with --data-dir <path> when raw data is available, or use --force to overwrite.")
            return 0

    if root_dir and os.path.exists(root_dir):
        s_dir, v_dir = get_dataset_subdirs(root_dir)
        s_files = glob.glob(os.path.join(s_dir, "*.csv"))
        v_files = glob.glob(os.path.join(v_dir, "*.csv"))
        
        # Build dictionary of V files keyed by normalized session id
        # Regex removes leading 'V-' or 'v_' prefix only, preserving internal hyphens
        v_dict = {}
        for f in v_files:
            base = os.path.basename(f)
            clean = re.sub(r'^[vV][-_]', '', base)
            clean = re.sub(r'\.csv$', '', clean, flags=re.IGNORECASE).strip().lower()
            v_dict[clean] = f

        matched_pairs = []
        for s_path in s_files:
            base = os.path.basename(s_path)
            # Regex removes leading 'S-' or 's_' prefix only, preserving casing for session id
            clean = re.sub(r'^[sS][-_]', '', base)
            clean = re.sub(r'\.csv$', '', clean, flags=re.IGNORECASE).strip()
            clean_lower = clean.lower()
            
            if clean_lower in v_dict:
                matched_pairs.append((clean, s_path, v_dict[clean_lower]))
            else:
                data_quality["anomalies"].append({
                    "session": clean,
                    "issue": f"No matching V-dataset file found for {base}"
                })

        print(f"Discovered {len(s_files)} S-files and {len(v_files)} V-files.")
        print(f"Matched {len(matched_pairs)} session pairs.")

        # Process each pair
        for session_id, s_path, v_path in matched_pairs:
            try:
                # latin1 avoids UnicodeDecodeError for superscript/degree symbols in headers
                s_df = pd.read_csv(s_path, encoding='latin1', on_bad_lines='skip')
                v_df = pd.read_csv(v_path, encoding='latin1', on_bad_lines='skip')
                
                # Clean headers (strip surrounding whitespace)
                s_df.columns = [c.strip() for c in s_df.columns]
                v_df.columns = [c.strip() for c in v_df.columns]
                
                s_rows, v_rows = len(s_df), len(v_df)
                if s_rows == 0 or v_rows == 0:
                    data_quality["anomalies"].append({
                        "session": session_id,
                        "issue": "Empty file encountered",
                        "s_rows": s_rows,
                        "v_rows": v_rows
                    })
                    continue

                if s_rows != v_rows:
                    data_quality["anomalies"].append({
                        "session": session_id,
                        "issue": "Row count mismatch",
                        "s_rows": s_rows,
                        "v_rows": v_rows
                    })
                    continue # Skip unmatched sessions to maintain strict time synchronization
                    
                # Time column analysis
                time_cols = [c for c in s_df.columns if "TIME" in c.upper() or "ms" in c.lower() or "timestamp" in c.lower()]
                s_time_col = time_cols[0] if time_cols else None
                
                dt_mean, dt_std, hz = 0.0, 0.0, 0.0
                if s_time_col and s_time_col in s_df.columns:
                    t_ms = pd.to_numeric(s_df[s_time_col], errors='coerce').values
                    dt_array = np.diff(t_ms)
                    dt_array = dt_array[~np.isnan(dt_array) & (dt_array > 0)]
                    if len(dt_array) > 0:
                        dt_mean = float(np.mean(dt_array))
                        dt_std = float(np.std(dt_array))
                        if dt_mean > 0:
                            hz = float(1000.0 / dt_mean)
                            data_quality["sampling_rates"].append(hz)
                
                # Velocity column detection from V-dataset
                vel_cols = [c for c in v_df.columns if "velocity" in c.lower() or "speed" in c.lower()]
                vel_col = None
                for c in vel_cols:
                    if "km/hr" in c.lower() and "velocity" in c.lower():
                        vel_col = c
                        break
                if not vel_col and vel_cols:
                    vel_col = vel_cols[0]
                
                if not vel_col or vel_col not in v_df.columns:
                    data_quality["anomalies"].append({
                        "session": session_id,
                        "issue": "Missing velocity column in V-dataset"
                    })
                    continue
                
                # Convert velocity to m/s
                raw_vel = pd.to_numeric(v_df[vel_col], errors='coerce')
                if "m/s" in vel_col.lower():
                    v_df['velocity_m_s'] = raw_vel
                else:
                    v_df['velocity_m_s'] = raw_vel / 3.6
                
                # Identify IMU columns
                accel_cols = [c for c in s_df.columns if "ACCELEROMETER" in c.upper()]
                gyro_cols = [c for c in s_df.columns if "GYROSCOPE" in c.upper()]
                
                if len(accel_cols) == 0 or len(gyro_cols) == 0:
                    data_quality["anomalies"].append({
                        "session": session_id,
                        "issue": "Missing IMU accelerometer or gyroscope columns in S-dataset"
                    })
                    continue

                feature_cols = accel_cols + gyro_cols
                if canonical_features is None:
                    canonical_features = list(feature_cols)

                # Ensure numeric types
                for c in feature_cols:
                    s_df[c] = pd.to_numeric(s_df[c], errors='coerce')

                # Handle NaNs via interpolation + fill
                has_nan = s_df[feature_cols].isna().any().any() or v_df['velocity_m_s'].isna().any()
                if has_nan:
                    s_df[feature_cols] = s_df[feature_cols].interpolate(method='linear').bfill().ffill()
                    v_df['velocity_m_s'] = v_df['velocity_m_s'].interpolate(method='linear').bfill().ffill()
                
                # Verify no remaining NaNs
                if s_df[feature_cols].isna().any().any() or v_df['velocity_m_s'].isna().any():
                    data_quality["anomalies"].append({
                        "session": session_id,
                        "issue": "Unresolvable NaNs after interpolation"
                    })
                    continue

                valid_sessions.append(session_id)
                duration_s = (s_rows * dt_mean) / 1000.0 if dt_mean > 0 else 0.0
                
                inventory.append({
                    "session_id": session_id,
                    "s_file": s_path,
                    "v_file": v_path,
                    "rows": s_rows,
                    "duration_s": duration_s,
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
                
                # Deterministic Train (70%) / Val (15%) / Test (15%) split using MD5 hash
                h = int(hashlib.md5(str(session_id).encode('utf-8')).hexdigest()[:8], 16) % 100
                if h < 70:
                    train_sessions.append(session_id)
                    is_train = True
                elif h < 85:
                    val_sessions.append(session_id)
                    is_train = False
                else:
                    test_sessions.append(session_id)
                    is_train = False
                    
                # Streaming accumulation for train split normalization (O(1) memory)
                if is_train:
                    feat_vals = s_df[feature_cols].values.astype(np.float64)
                    tgt_vals = v_df['velocity_m_s'].values.astype(np.float64)

                    n_pts = len(feat_vals)
                    train_samples_count += n_pts

                    if train_feat_sum is None:
                        train_feat_sum = np.sum(feat_vals, axis=0)
                        train_feat_sq_sum = np.sum(feat_vals ** 2, axis=0)
                    else:
                        train_feat_sum += np.sum(feat_vals, axis=0)
                        train_feat_sq_sum += np.sum(feat_vals ** 2, axis=0)

                    train_tgt_sum += float(np.sum(tgt_vals))
                    train_tgt_sq_sum += float(np.sum(tgt_vals ** 2))

                # Sanity check plot (session S1 or first processed session)
                if args.plots_dir and (session_id == "S1" or len(valid_sessions) == 1):
                    os.makedirs(args.plots_dir, exist_ok=True)
                    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
                    
                    axes[0].plot(s_df[accel_cols].values)
                    axes[0].set_title(f"Session {session_id} Accelerometer (m/s²)")
                    axes[0].legend(accel_cols, loc='upper right')
                    
                    axes[1].plot(s_df[gyro_cols].values)
                    axes[1].set_title(f"Session {session_id} Gyroscope (rad/s)")
                    axes[1].legend(gyro_cols, loc='upper right')
                    
                    axes[2].plot(v_df['velocity_m_s'].values, label="Velocity (m/s)", color='red')
                    axes[2].set_title(f"Session {session_id} Ground Truth Velocity")
                    axes[2].legend()
                    
                    plt.tight_layout()
                    plot_path = os.path.join(args.plots_dir, f"sanity_check_{session_id}.png")
                    fig.savefig(plot_path)
                    plt.close(fig)
                    
            except Exception as e:
                inventory.append({
                    "session_id": session_id,
                    "status": f"ERROR: {str(e)}"
                })
                data_quality["anomalies"].append({"session": session_id, "error": str(e)})

    # Calculate Normalization Stats from streaming sums
    normalization = None
    if train_samples_count > 0 and train_feat_sum is not None and canonical_features is not None:
        feat_mean = (train_feat_sum / train_samples_count).tolist()
        feat_var = (train_feat_sq_sum / train_samples_count) - ((train_feat_sum / train_samples_count) ** 2)
        feat_std = np.sqrt(np.maximum(feat_var, 1e-12)).tolist()
        # Avoid division by zero
        feat_std = [float(s) if s > 1e-6 else 1.0 for s in feat_std]
        feat_mean = [float(m) for m in feat_mean]
        
        target_mean = float(train_tgt_sum / train_samples_count)
        target_var = float((train_tgt_sq_sum / train_samples_count) - (target_mean ** 2))
        target_std = float(np.sqrt(max(target_var, 1e-12)))
        target_std = target_std if target_std > 1e-6 else 1.0
        
        # Dynamic units matching canonical feature names
        units = ["m/s2" if "accel" in c.lower() else "rad/s" for c in canonical_features]

        normalization = {
            "version": "1.0",
            "features": {
                "names": canonical_features,
                "mean": feat_mean,
                "std": feat_std,
                "units": units
            },
            "targets": {
                "name": "velocity_m_s",
                "mean": target_mean,
                "std": target_std,
                "unit": "m/s"
            }
        }

    # Guard against destructive manifest overwrite when no valid data is processed
    if len(valid_sessions) == 0 and not args.force:
        print(f"\n[GUARD] No valid session pairs were processed.")
        print(f"[GUARD] Manifests in '{args.output_dir}' preserved without overwriting.")
        print(f"[GUARD] Use --force if you intentionally wish to overwrite manifests.")
        return 0

    # Save manifests safely
    os.makedirs(args.output_dir, exist_ok=True)
    pd.DataFrame(inventory).to_csv(os.path.join(args.output_dir, "session_inventory.csv"), index=False)
    with open(os.path.join(args.output_dir, "data_quality.json"), "w", encoding="utf-8") as f:
        json.dump(data_quality, f, indent=4)
        
    with open(os.path.join(args.output_dir, "train_sessions.json"), "w", encoding="utf-8") as f:
        json.dump(train_sessions, f)
    with open(os.path.join(args.output_dir, "val_sessions.json"), "w", encoding="utf-8") as f:
        json.dump(val_sessions, f)
    with open(os.path.join(args.output_dir, "test_sessions.json"), "w", encoding="utf-8") as f:
        json.dump(test_sessions, f)
        
    if normalization is not None:
        with open(os.path.join(args.output_dir, "normalization.json"), "w", encoding="utf-8") as f:
            json.dump(normalization, f, indent=4)

    print(f"\nSuccessfully processed {len(valid_sessions)} valid sessions.")
    print(f"Manifests written to '{args.output_dir}':")
    print(f"  - Train: {len(train_sessions)} sessions")
    print(f"  - Val:   {len(val_sessions)} sessions")
    print(f"  - Test:  {len(test_sessions)} sessions")
    if len(data_quality['sampling_rates']) > 0:
        print(f"Mean sampling rate: {np.mean(data_quality['sampling_rates']):.2f} Hz")

    return 0


if __name__ == "__main__":
    main()
