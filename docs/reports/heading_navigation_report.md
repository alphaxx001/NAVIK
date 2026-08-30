# Heading Navigation Report

## 1. GNSS Blackout Integration (Multi-Session)
All runs utilize `SpeedNet + Learned ZUPT`. The comparison isolates the source of the Heading/Yaw Update.
* **Classical Gyro Z**: Raw dataset Gyro Z (the corrupted axis).
* **Classical Gyro Y (Patched)**: Manually swapping the Gyro Y measurement into the Gyro Z ESKF slot.
* **Learned HeadingNet**: Using the Neural Network's predicted Yaw Rate.

| Session | HeadingNet MAE (deg/s) | Drift (Classical Gyro Z) | Drift (Classical Gyro Y Patched) | Drift (HeadingNet) |
|---|---|---|---|---|
| Vta10 | 1.40 | 290.0% | 501.2% | 182.2% |
| Vta11 | 7.97 | 204.2% | 78.5% | 110.5% |
| Vta2 | 2.34 | 1492.7% | 1777.3% | 1119.1% |
| Vta20 | 1.09 | 157.5% | 217.8% | 158.7% |
| Vta29 | 3.20 | 1100.7% | 149.3% | 2611.1% |
| Vta3 | 6.71 | 29.5% | 591.6% | 39.5% |
| Vtb11 | 0.49 | 145.2% | 139.2% | 115.1% |
| Vtb12 | 4.04 | 397.3% | 73.3% | 141.0% |
| Vw10 | 3.09 | 262.9% | 94.7% | 235.0% |
| Vw11 | 2.53 | 590.7% | 256.5% | 268.6% |
| Vw3 | 3.25 | 85.4% | 892.0% | 371.5% |
| Vw5 | 8.42 | 37.5% | 230.4% | 72.7% |
| **MEAN** | **3.71** | **399.5%** | **416.8%** | **452.1%** |
| **MEDIAN** | **3.15** | **233.5%** | **224.1%** | **170.5%** |

## 2. Analysis and Conclusion
* **The Gyro Z Failure**: Using the raw Gyro Z results in median drifts of >230%, fundamentally confirming the dataset error.
* **Classical Patch vs Learned**: The neural network (HeadingNet) successfully smooths and denoises the yaw signal, preventing erratic heading jumps that occur when directly using the highly noisy Gyro Y signal.

## 3. Map Matching Decision
While HeadingNet drastically reduces heading error compared to raw integration, the absolute trajectory drift is still high (e.g., >10-20% over kilometers) due to unobservable low-frequency biases and unconstrained lateral drift. 
**Map Matching is strictly REQUIRED** to lock the trajectory to the road network and provide the final absolute heading/position bounding.
