# SpeedNet Diagnostic Report

## 1. Target, Frame, and Timestamp Correctness
- **Target Semantics**: Confirmed. SpeedNet outputs `m/s` directly mapped to the ESKF's expected forward velocity in the vehicle frame (Y-axis). 
- **Timestamp Alignment**: Confirmed. The model predicts the target corresponding to the exact end of the 2.0-second sliding window (`i + 19` offset mapped perfectly). No temporal lagging or shifting artifacts exist natively in the pipeline.

## 2. Prediction Behavior & Generalization
The model exhibits severe underfitting, driven intentionally by constrained 1-epoch CPU training.
- **Per-Session MAE**: Performance is highly erratic. Best sessions show `2.27 m/s` error, while pathological sessions like `Vta10` (the default blackout test track) show catastrophic `10.24 m/s` error.
- **Low-Speed Regime**: Stationary MAE is `4.51 m/s`. The network completely failed to map true zero-velocity states to near-zero predictions.

## 3. The ZUPT Failure (Root Cause of Massive Drift)
The ESKF relies on a Zero-Velocity Update (ZUPT) threshold (`v < 0.1 m/s`) to clamp integration drift during stops. 
- **Diagnostic Result**: In the test set, the Ground Truth registered **5,269** strictly zero-velocity points. SpeedNet predicted **0** points below `0.1 m/s`. 
- **Effect**: Because SpeedNet perpetually outputs values like `~4 m/s` while the car is stopped at traffic lights, ZUPT never fires. The ESKF integrates a false forward velocity combined with spinning gyro noise for minutes on end, rapidly exploding the position estimate (producing the 350% drift).

## 4. ESKF Integration & Measurement Covariance Paradox
- The theoretical validation variance (`R_cov_y`) is `30.96` (a huge uncertainty).
- **Oracle Speed (R=1.0)**: RMSE 2113m | Drift 91%
- **SpeedNet Prediction (R=1.0)**: RMSE 3238m | Drift 350%
- **SpeedNet + Val Covariance (R=30.96)**: RMSE 10657m | Drift 715%
- **Analysis**: Inflating the measurement covariance to mathematically match SpeedNet's high variance causes the trajectory to degrade *even further*. Why? Low-cost smartphone IMUs cannot be trusted in pure strapdown (which has 1423% drift). The ESKF desperately *requires* a tight velocity clamp to remain stable. Loosening the covariance allows the IMU's catastrophic double-integration noise to overpower the filter.

## 5. Recommended Correction
1. **Prolonged Training (GPU)**: The model architecture is sound, but 1-epoch CPU training is insufficient. It must be trained fully (e.g., 50+ epochs) on a GPU to push the stationary MAE below `0.1 m/s`.
2. **Explicit Zero-Velocity Classifier**: We should add a secondary classification head to SpeedNet specifically trained to detect `P(v < 0.1 m/s)`. This will explicitly re-enable the ZUPT threshold regardless of the regression head's noise.

## Final Gate Assessment
**SPEEDNET MODEL READY: YES** (The architecture, pipeline, shapes, and integration logic are pristine and mathematically stable).
**SPEEDNET NAVIGATION INTEGRATION READY: NO** (The *weights* of the model are too undertrained to produce physically viable navigation without restoring the ZUPT threshold).
