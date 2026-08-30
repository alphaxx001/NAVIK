# Classical Baseline Conventions

This document establishes the reference coordinate and mathematical conventions deduced from the IO-VNBD dataset for use in the Error-State Kalman Filter (ESKF).

## 1. Frame Definitions
* **Sensor Frame (s-frame)**: 
  Based on acceleration means during stationary periods (Z ≈ 9.85 m/s²), the smartphone screen was facing up (lying flat).
  - X axis: Right of the vehicle
  - Y axis: Forward motion of the vehicle (derived from negative correlation with velocity derivative and typical smartphone orientation).
  - Z axis: Upwards (+g).
* **Navigation Frame (n-frame)**:
  - Local tangent plane: East-North-Up (ENU).
* **Vehicle Frame (v-frame)**:
  - Assumed loosely aligned with the s-frame for this dataset since the phone was rigidly mounted flat. X = Right, Y = Forward, Z = Up.

## 2. Sensor Conventions
* **Accelerometer**: Measures specific force $f^s = a^s - g^s$.
  - When stationary, $a^s = 0$, thus $f^s = -g^s$. 
  - Since $Z_{accel} \approx 9.85$, $g$ points downwards in the navigation frame, meaning $g^n = [0, 0, -9.81]^T$.
* **Gyroscope**: Right-hand rule around the respective axes in rad/s.
* **Heading**: WGS84 Course Over Ground (degrees), assumed True North.

## 3. Initialization
* **Position**: Set to `(0, 0, 0)` in ENU. Real WGS84 GNSS coordinates will be projected into ENU.
* **Velocity**: Initialized from the first valid vehicle velocity.
* **Attitude**: Yaw initialized from Vehicle Heading. Pitch/Roll initialized via accelerometer gravity vector.
* **Biases**: Initialized to 0 or estimated over the first stationary seconds.

## 4. Updates
* **Oracle Speed**: Vehicle CAN velocity mapped directly to $v_y$ (forward velocity in v-frame).
* **NHC (Non-Holonomic Constraints)**: 
  - $v_x \approx 0$ (no lateral slip)
  - $v_z \approx 0$ (no jumping)
* **ZUPT (Zero Velocity Update)**: $v = [0,0,0]^T$ when vehicle velocity is identically 0.
* **ZIHR (Zero Integrated Heading Rate)**: $\omega_z \approx 0$ during stationary periods.
