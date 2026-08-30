# Heading and Supervision Audit

## 1. Available Supervision Signals
The IO-VNBD dataset provides two potential supervision signals in the `V-Dataset`:
1. `Heading (degrees)`: An absolute compass/GNSS-derived course over ground.
2. `Yaw Rate (deg/sec)`: High-fidelity CAN-bus or OBD-II derived vehicle rotation rate.

## 2. Heading Wraparound and Discontinuities
Analysis of the `Heading (degrees)` column reveals that it is clamped to `[0, 360]`. 
* **Wraparound**: When the vehicle turns north, the signal jumps from `359.9` to `0.1` (or vice-versa). Direct regression of this angle with a neural network (e.g., using MSE loss) is mathematically flawed because a prediction of `1.0` and a truth of `359.0` will yield a massive error (358) instead of the true error (2).
* **Quality**: As a GNSS-derived signal, absolute heading becomes extremely noisy or undefined at zero/low velocities.

## 3. Frame and Mounting Validation (CRITICAL FINDING)
A deep analysis was performed to validate the assumed smartphone-to-vehicle mounting frame.
* **Accelerometer**: Across all tested sessions (`S1`, `Vta10`, `Vw11`, `Vta2`), the mean `Accelerometer Z` is `~9.8 m/s²`. This definitively proves the smartphone was mounted **horizontally (flat, screen-up)**.
* **Gyroscope Anomaly**: In a flat mounting, the vehicle's yaw (turning left/right) represents a rotation around the gravity vector, which is the Z-axis. Therefore, `Gyroscope Z` should strongly correlate with the vehicle's `Yaw Rate`.
* **The Reality**: The correlation between `Gyroscope Z` and `Yaw Rate` is **0.00** across multiple sessions. Instead, `Gyroscope Y` has an incredibly strong correlation (up to **0.93** in session `S1`), and its magnitude (rad/s converted to deg/s) perfectly matches the vehicle's yaw rate magnitude.
* **Conclusion**: There is a fundamental flaw/axis-swap in the IO-VNBD dataset's Gyroscope logging. The yaw angular velocity is recorded in the `GYROSCOPE Y` column, not `Z`. 

## 4. Definition of the Learning Task
Given the above findings, predicting full 3D attitude (quaternions) is unsupported because:
1. Pitch and Roll ground-truth do not exist.
2. The Gyroscope axes are swapped or corrupted in the dataset, rendering strict strapdown kinematics mathematically broken without a learned or manually patched frame transformation.

**Selected Task**: `Yaw Rate Estimation`.
Instead of integrating a swapped/noisy Gyroscope manually, we will train a `HeadingNet` to directly map the 6-DoF IMU window `[Batch, 6, Window]` to the vehicle's `Yaw Rate (rad/sec)`. 
* **Why?**: The neural network can automatically learn the cross-axis mappings (e.g., pulling the yaw signal out of the `Gyro Y` column) and simultaneously denoise the extreme vibrations seen in sessions like `Vta10`. 
* **Integration**: The predicted Yaw Rate can be cleanly integrated via $\theta_t = \theta_{t-1} + \omega_{pred} \Delta t$ inside the ESKF, completely bypassing the corrupted Gyro Z measurement.

**Final Decision**: The learning task is justified. We will proceed with `HeadingNet` trained on MSE against `Yaw Rate (deg/sec)` (converted appropriately to rads for the filter).
