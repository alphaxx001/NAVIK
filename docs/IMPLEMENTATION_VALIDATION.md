# Implementation Validation Report

## A. DATASET (IO-VNBD)
- **Actual smartphone IMU sampling rate:** 10 Hz (Confirmed by paper and `TIME SINCE START (ms)` deltas of 100ms in CSV).
- **Exact accelerometer fields:** `ACCELEROMETER X (m/s²)`, `ACCELEROMETER Y (m/s²)`, `ACCELEROMETER Z (m/s²)`.
- **Exact gyroscope fields:** `GYROSCOPE X (rad/s)`, `GYROSCOPE Y (rad/s)`, `GYROSCOPE Z (rad/s)`. Note: The paper text ambiguously refers to Yaw/Pitch/Roll, but the CSV header clearly contains X/Y/Z.
- **Exact GNSS fields:** `GPS LATITUDE (degrees)`, `GPS LONGITUDE (degrees)`, `GPS SPEED (Kmh)`.
- **Vehicle-side/reference signals available:** `Velocity (km/hr)`, `Heading (degrees)`, `Yaw Rate (deg/sec)`, `Wheel Speed (rad/sec)`, `Steering Angle`.
- **Exact velocity/odometry fields:** `Velocity (km/hr)` from Vehicle CAN-Bus is the primary ground truth.
- **Attitude ground truth availability:** **NOT AVAILABLE**. The dataset does not provide Pitch and Roll ground truth for the vehicle, only Heading and Yaw Rate.
- **Synchronization:** The `Synchronised V and S datasets` directory contains perfectly row-aligned CSV files (e.g., both S-S1.csv and V-S1.csv have exactly 51,746 rows).
- **Coordinate frames and units:** Accelerometer is in m/s², Gyroscope in rad/s. Vehicle velocity is in km/hr.

## B. CURRENT CODE (`dataset_loader.py`)
- **What it actually loads:** Parses `s_df` and `v_df`.
- **Input tensor produced:** Tensors of shape `(Channels, Sequence_Length)` i.e., `(6, 200)`.
- **Labels produced:** A single scalar `y_vel` (Velocity at the end of the window).
- **Alignment:** It assumes row-by-row alignment, which is correct for the `Synchronised` folders. 
- **Assumptions hard-coded:** 
  - `window_size = 200`. At 10 Hz, this represents 20 seconds of data, which is excessively large and smoothing for instantaneous velocity prediction.
  - No attitude labels are extracted because they don't exist.

## C. AVNET (`avnet.py`)
- **Input shape expected:** `(Batch, 6, 200)`.
- **Why 200 samples was chosen:** The AVNet research paper used an IMU sampling at 100 Hz, making 200 samples exactly 2 seconds.
- **Is 200 samples supported by the actual dataset?:** No. At 10 Hz, 200 samples is 20 seconds. We should adapt the architecture to accept `window_size = 20` to represent the same 2-second temporal window.
- **Velocity output represents:** Forward longitudinal velocity.
- **Attitude output represents:** A 4D Quaternion for attitude change.
- **Is every output supervised?:** **NO.** The attitude output (`fc_attitude`) has no ground truth in the IO-VNBD dataset to supervise it against. 

## D. TRAINING (`train.py`)
- **Is it safe to run?:** Technically yes, it won't crash because the loss is only calculated on `pred_vel`, ignoring the unsupervised `attitude` output. However, it is not physically meaningful right now.
- **Train/validation/test split:** Currently non-existent. It trains on a single file.
- **Sequence leakage:** If we slide the window by 1 sample, adjacent windows share 99% of the data. Random shuffling of these windows in `DataLoader` is mathematically valid for CNNs, but careful train/val splitting *by trajectory* (e.g., train on S1, validate on S2) is strictly required to prevent data leakage.
- **Normalization:** The features (Accel/Gyro) and targets (km/h) are not normalized (e.g., Standard Scaling), which will severely hinder neural network convergence.

## E. PAPERS
Comparison of our implementation against the supplied papers:
- **AVNet Architecture (1D-CNN + GRU):** [DIRECTLY SUPPORTED]
- **Data-Driven Velocity (DDODO):** [DIRECTLY SUPPORTED] by IO-VNBD.
- **Data-Driven Attitude (DDATT):** [NOT SUPPORTED]. IO-VNBD lacks Pitch/Roll ground truth. The AVNet paper used a custom dataset with a Novatel SPAN for this.
- **Invariant EKF (InEKF):** [ADAPTED]. We must adapt it to run without DDATT, or only use Heading/Yaw rate constraints instead of full 3D attitude.

## F. PREVIOUS WINNER (Drishti App)
- **Useful:** Their Android UI layout, Sensor Dashboard, preference storage, and system architecture for running models efficiently on-device.
- **Do NOT copy:** Their core algorithm. Drishti relied on P2P communication and standard EKF for positioning. Our problem statement explicitly requires **AI-ML models to predict speed from IMU**. Copying their non-AI dead reckoning would fail the core requirement of SIH 26168.

## G. FINAL DECISION

1. **What is correct and can remain:** The AVNet CNN+GRU hybrid approach, the row-by-row dataset loader concept.
2. **What must be changed:** 
   - `avnet.py`: Make `window_size` dynamic (default to 20 for 10Hz).
   - `dataset_loader.py`: Add Min-Max or Standard scaling for IMU inputs and Velocity targets.
   - `train.py`: Implement proper trajectory-based Train/Val splitting.
3. **What must be removed:** The `fc_attitude` head from AVNet, as we cannot train it.
4. **What is missing:** A validation loop in `train.py`.
5. **What can be implemented later:** The InEKF filter engine and the Mobile App UI.

---

### SAFE TO TRAIN?
**NO.**
If you train right now, the model will ingest 20-second windows (200 samples at 10Hz) without feature normalization, attempting to map raw `m/s²` and `rad/s` to unscaled `km/hr`. The network weights will explode or fail to converge. We must fix the window size and add normalization before starting the training loop.
