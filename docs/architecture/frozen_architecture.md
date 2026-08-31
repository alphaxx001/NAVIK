# IDR-System: Frozen Architecture

**Git Baseline Checkpoint:** `30d2988`

This document defines the strictly frozen architecture for the IDR-System navigation core, serving as the blueprint for Edge Engine productization (Phase 12). 

## 1. Frozen Modules
- **Data Pipeline:** `IOVNBDSchemaResolver` dynamically standardizes varying sensor column schemas for smartphone datasets.
- **Deep Learning Modules:** 
  - `SpeedNet` (CNN-GRU pseudo-velocity estimator)
  - `MotionStateNet` (CNN-GRU stationary classifier for ZUPT triggering)
  - `HeadingNet` (CNN-GRU yaw-rate corrector)
- **Mathematical Filter:** 15-state Error-State Kalman Filter (ESKF) implementing ZUPT (Zero Velocity Updates) and NHC (Non-Holonomic Constraints).
- **Map Context Layer:** Spatial KD-tree lookup paired with an Open-Loop Hidden Markov Model (HMM / Viterbi).

## 2. Data Flow
1. **Input:** Raw IMU (Accelerometer + Gyroscope). GNSS is utilized *exclusively* for timestamp alignment and initial ENU frame anchoring.
2. **Preprocessing:** IMU scaled and Z-score normalized using fixed dataset parameters.
3. **Inference:** $2.0s$ windows are batched and inferred by SpeedNet, MotionStateNet, and HeadingNet at $10Hz$.
4. **Integration (Authoritative):** ESKF integrates the raw IMU at full sensor frequency ($~100Hz$) and periodically consumes the neural pseudo-measurements via Kalman updates ($10Hz$).
5. **Map Post-Processing (Display):** The dead-reckoned ESKF state is overlayed against real OSM geometries via the KD-tree HMM to produce the contextual display layer.

## 3. Interfaces & Model I/O
- **Input Tensors:** `[B, 6, 200]` sequences (6 channels: $Accel_{x,y,z}$, $Gyro_{x,y,z}$; 200 steps representing $2.0s$ of data at $100Hz$).
- **SpeedNet Output:** `[B, 1]` tensor representing longitudinal speed ($m/s$).
- **MotionStateNet Output:** `[B, 1]` tensor representing stationary probability $P \in [0, 1]$. Thresholded at $0.569$ with a 3-step hysteresis filter.
- **HeadingNet Output:** `[B, 1]` tensor representing corrected yaw-rate ($rad/s$).
- **Map Output:** State sequence mapping either to a specific OSM `road_id` or `DR_FALLBACK`.

## 4. Coordinate Frames & Units
- **Sensor Frame:** Accelerometer ($m/s^2$), Gyroscope ($rad/s$).
- **Global Anchor:** WGS84 Latitude/Longitude establishes the anchor point at $t_{GNSS}=0$.
- **Navigation Frame:** Local ENU (East-North-Up) in meters.
- **Heading:** Yaw is tracked in radians from East (counter-clockwise).

## 5. ESKF / Map-Matching Separation (Strict Rule)
The ESKF remains the absolute authoritative physical state. 
**Rule:** Map-matching results (`MAP_CONTEXT`) are mathematically decoupled and MUST NOT be fed back into the ESKF as position/heading updates. Closed-loop map feedback on smartphone-grade inertial hardware leads to catastrophic topological lock-in (as proven in Phase 11 `Vtb1`), and is permanently disabled.

## 6. GNSS Blackout Behavior
1. **Acquisition:** Valid GNSS provides absolute WGS84 fixes.
2. **Loss/Blackout:** The system mathematically seals the GNSS inputs. The ESKF transitions to pure Inertial/Neural integration mode.
3. **Map Overlay:** The HMM continuously attempts to snap the DR state to the OSM graph. If the trajectory physically diverges beyond a $100m$ radius of any valid road, the HMM yields `DR_FALLBACK`, trusting pure kinematics over geometric hallucination.

## 7. Known Limitations
- **Unbounded Kinematic Drift:** Pure inertial integration possesses no absolute geographical anchors. Across massive blackouts ($>20$ minutes), uncorrected heading biases compound heavily. 
- **Off-Road Environments:** Complex parking lots or unmapped rural zones naturally lack OSM structures, perpetually locking the context layer into `DR_FALLBACK`.
