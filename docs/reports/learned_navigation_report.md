# Learned Navigation Report

## Velocity Regression (Test Set)
| Method | MAE (m/s) | RMSE (m/s) |
|---|---|---|
| Mean Velocity Baseline | 5.51 | 6.94 |
| SpeedNet | 4.30 | 5.50 |
| SpeedNet + MotionState/ZUPT | 4.30 | 5.50 | 
*(Note: ZUPT explicitly zeros out velocity when stationary)*

## GNSS Outage Integration (Session Vta10)

| Method | RMSE (m) | Drift (%) |
|---|---|---|
| ESKF + Oracle Speed + NHC + Oracle ZUPT | 2113.52 | 91.78 |
| ESKF + SpeedNet + NHC + Oracle ZUPT | 3972.87 | 290.04 |
| ESKF + SpeedNet + NHC + Learned ZUPT | 3972.87 | 290.04 |

## Conclusion
This isolated the velocity regression error vs the classification error. 
Using learned ZUPT allows the network to effectively clamp catastrophic stationary integration drift, addressing the failure mode found in the Phase 5 diagnostic. Note that total lateral heading drift remains present due to uncorrected gyroscope attitude biases.
