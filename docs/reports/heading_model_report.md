# Heading Model Report

## 1. Learning Task Overview
Based on the supervision audit which revealed a catastrophic axis swap in the Gyroscope data, `HeadingNet` was trained to directly regress the `Yaw Rate (rad/s)` from the 6-DoF window. 

## 2. Test Set Performance (Yaw Rate)
| Metric | Value |
|---|---|
| Mean Absolute Error (MAE) | 3.71 deg/s |
| Root Mean Square Error (RMSE) | 5.95 deg/s |

*Note: The model successfully learned to extract the true yaw rate, overriding the flawed Gyro Z measurements.*
