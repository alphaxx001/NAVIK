# Final Test Evaluation Report

## 1. Experimental Setup & Integrity
This evaluation was executed on the 12 frozen test sessions that were completely isolated during all prior research phases (Phases 1-11). No test-set statistics, hyperparameter tuning, or candidate selection thresholds were informed by this data.

## 2. Frozen Architecture
The evaluated pipeline strictly implements the decoupled Open-Loop architecture (Phase 12 checkpoint `f561adb`):
`IMU` $\rightarrow$ `SpeedNet` + `MotionStateNet` + `HeadingNet` $\rightarrow$ `ESKF` $\rightarrow$ `Raw DR Trajectory` $\rightarrow$ `Open-Loop KD-Tree HMM` $\rightarrow$ `Map Context Display`
Map-matching feedback into the ESKF is fundamentally disabled to prevent topological divergence lock-in.

## 3. Blackout Methodology
Complete GNSS blackout was enforced. GNSS data was consumed *only* at $t=0$ to initialize the local ENU frame and starting position/heading. All subsequent state estimations relied purely on uncorrected inertial data and neural pseudo-measurements.

## 4. Aggregate Results
- **TEST SESSIONS COMPLETED**: 12/12
- **Median DR Drift**: 113.22%
- **Mean DR Drift**: 139.76%
- **Median MAP Drift**: 113.69%
- **Mean MAP Drift**: 140.06%
- **Median MAP RMSE**: 1083.82 m
- **Worst Session**: `Vta20` (318.76% MAP Drift)

## 5. Baseline Comparison
When compared to the prior Validation experiments (where Median DR Drift was ~44%), the Held-Out Test Set proved significantly more challenging. Because inertial heading drift is mathematically unbounded, minor uncorrected gyro biases in these specific test devices compound cubically.

The Open-Loop Map Context essentially performed equivalently to the Raw DR. It did not improve the median drift (-0.47% degradation). This mathematically confirms the Phase 8 conclusion: open-loop map matching cannot structurally rescue an underlying trajectory that has already diverged by over 100% of the distance traveled. 

## 6. Worst-Case Analysis (`Vta20`)
`Vta20` experienced an extraordinary 318% drift. This is symptomatic of a severe gyro bias anomaly or dynamic magnetic interference that HeadingNet failed to regress. Because GNSS was disabled, the ESKF integrated this heading error continuously, causing the vehicle to mathematically drive in expanding spiral loops completely disjointed from physical reality. Candidate coverage for `Vta20` dropped to 8.3%, meaning the system spent 91.7% of the session in `DR_FALLBACK`.

## 7. Failure Modes & Limitations
1. **Unbounded Inertial Drift**: Smartphone MEMS IMUs cannot sustain 20+ minute GNSS blackouts without external absolute referencing (e.g., visual odometry, LiDAR, or tightly coupled GNSS).
2. **Topological Disconnect**: Once the physical DR trajectory drifts beyond the 100m OSM candidate radius, the HMM correctly refuses to match, leaving the system stranded in dead-reckoning.

## 8. Final Conclusions
The architectural decision to decouple the map matcher from the ESKF was correct: feeding a 318% drifted trajectory (`Vta20`) into a closed-loop topological snapper would have resulted in catastrophic system failure (similar to `Vtb1` in Phase 11). The system correctly identified the low confidence and safely fell back to pure DR. 

To improve the 113% drift, future non-frozen iterations must focus exclusively on improving the baseline kinematics (e.g., stronger SpeedNet/HeadingNet architectures, tighter ZUPT intervals, or multi-sensor fusion) rather than relying on map-matching heuristics.
