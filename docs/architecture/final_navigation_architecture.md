# Final Navigation Architecture (Frozen)

## 1. System Overview & Data Flow
The IDR-System navigation architecture strictly implements a **Decoupled Open-Loop Post-Processing** design. It prioritizes the uncorrupted physical integrity of the inertial state over greedy topological snapping.

**Data Flow**:
`IMU Stream` $\rightarrow$ `SpeedNet` + `MotionStateNet` + `HeadingNet` $\rightarrow$ `ESKF` $\rightarrow$ `Raw DR Trajectory` $\rightarrow$ `Open-Loop KD-Tree HMM` $\rightarrow$ `Final Display State`

## 2. Component Roles
- **SpeedNet**: Regresses 2.0s windowed IMU data into longitudinal vehicle speed constraints.
- **MotionStateNet**: Classifies IMU windows to detect stationary states, enabling Zero Velocity Updates (ZUPT).
- **HeadingNet**: Regresses yaw-rate variations to correct gyroscope Z-axis bias and scaling.
- **ESKF (Error-State Kalman Filter)**: The *Authoritative Navigation State*. Integrates raw IMU physics loosely coupled with neural pseudo-measurements.
- **GNSS**: Used strictly for initialization and ground-truth validation. Absolutely no GNSS data is consumed during the blackout runtime.
- **KD-Tree**: Discretizes the true OSM road graph into a heavily optimized spatial index, achieving 1000x speedups in candidate generation.
- **Open-Loop HMM**: Uses Viterbi dynamic programming to find the optimal topological sequence of road segments that physically aligns with the ESKF trajectory history.

## 3. Map Feedback Decoupling Rationale
Extensive research (Phases 7-11) conclusively proved that feeding map geometry back into the ESKF (whether greedily, via Particle Filters, or via Factor Graphs) is fundamentally brittle in topologically ambiguous environments. Continuous physical filters cannot natively handle discrete integer choices (e.g., Road A vs Road B). When closed-loop architectures lock onto a false road, they actively pull the ESKF into a physical divergence from reality (e.g., inducing 3048% drift in `Vtb1`).

By decoupling, the map matcher acts as a contextual *display layer*. It provides topological confidence and road-snapped visualizations without risking the physical destruction of the core dead-reckoning state.

## 4. Authoritative vs Contextual State
- **AUTHORITATIVE NAVIGATION STATE**: The pure ESKF (`pos_dr`). Physically continuous, geometrically unbounded.
- **MAP-MATCHED DISPLAY/CONTEXT STATE**: The HMM output (`pos_map`). Geometrically constrained to known OSM roads, but can enter `DR_FALLBACK` if the authoritative state drifts beyond the 100m candidate radius.

## 5. Known Limitations
- **Long-Duration Blackout Drift**: Without GNSS, purely inertial ESKFs eventually drift out of the 100m map candidate radius (entering permanent fallback).
- **Smartphone IMU Noise**: Consumer-grade MEMS sensors contain immense stochastic noise that restricts blackout survivability.
- **Topological Ambiguity**: Unmapped roads or complex parking lots will force the system into fallback.
- **Validation Limits**: Currently achieves ~44% median drift on 2500-second (40 minute) blackouts. 
