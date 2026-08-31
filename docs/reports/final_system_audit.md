# Final System Audit Report

## 1. Reproducibility Baseline
- **Git Commit (Frozen Architecture)**: `f561adb`
- **Environment**: Python 3.13 (Windows), PyTorch (CPU mode).
- **Model Checkpoints**: 
  - `models/speednet/checkpoints/best_speednet.pth`
  - `models/motion_state/checkpoints/best_motion_state.pth`
  - `models/heading/checkpoints/best_heading.pth`
- **Configuration Files**: `configs/speednet_training.json`, `configs/motion_state_training.json`, `configs/heading_training.json`.
- **Dataset Manifests**: `data/manifests/session_inventory.csv` defining strict Train/Val/Test boundaries.
- **OSM Requirement**: `data/raw/OSM/uk_midlands.osm.pbf` (or functionally equivalent local map chunk).

## 2. Architecture & Data Integrity Audit
- **Pipeline Integrity (PASS)**: `scripts/run_navigation_pipeline.py` definitively feeds raw IMU into the neural networks, pipes the predicted kinematics into the authoritative ESKF, and then passes the purely dead-reckoned trajectory to the KD-Tree HMM. The architecture enforces one-way map data flow; map context is strictly a decoupled display layer.
- **Data Isolation (PASS)**: Normalization constants (e.g., IMU mean/std) were permanently hardcoded during Phase 2 training and originate entirely from the training split. Evaluation scripts explicitly enforce execution exclusively on their designated splits. 
- **GNSS Blackout Strictness (PASS)**: The runtime pipeline reads GNSS exclusively at index `0` for ENU local-frame anchor and initial yaw alignment. For all indices $>0$, the GNSS arrays are never indexed or accessed by the propagation loop.

## 3. Model & ESKF Audit
- **SpeedNet (PASS)**: Receives normalized IMU windows, outputs localized velocity magnitude ($m/s$).
- **MotionStateNet (PASS)**: Outputs a stationary probability. The system applies a frozen $0.569$ threshold coupled with a 3-step hysteresis filter to trigger strict Zero Velocity Updates (ZUPT).
- **HeadingNet (PASS)**: Outputs yaw-rate variation, directly replacing the raw Z-gyroscope measurements during ESKF integration to correct scaling and dynamic bias errors.
- **ESKF (PASS)**: Canonical 15-DOF inertial state. Correctly models Non-Holonomic Constraints (NHC) alongside the SpeedNet updates to penalize impossible lateral/vertical velocity drift.

## 4. Map Matching Audit
- **HMM/Viterbi (PASS)**: Fully implemented utilizing OSM-derived graph topologies.
- **KD-Tree Acceleration (PASS)**: Effectively queries the map manifold within the frozen 100m dynamic search radius.
- **Fallback (PASS)**: Unmapped regions or trajectories that severely physically drift $>100m$ correctly trigger `DR_FALLBACK`, gracefully terminating map-snapping to prevent geometric hallucinations.
