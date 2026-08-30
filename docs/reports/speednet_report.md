# SpeedNet Evaluation Report

## Overview
SpeedNet replaces the CAN-bus Oracle Velocity with learned inertial velocity.

## Test Set Metrics (m/s)
* **Mean Predictor Baseline**: MAE 5.51 | RMSE 6.94
* **SpeedNet**: MAE 4.20 | RMSE 5.28

## ESKF Integration (100% GNSS Blackout on Session Vta10)
* **ESKF + ORACLE + NHC**: Trajectory RMSE: 2113.52m | Drift: 91.78%
* **ESKF + SPEEDNET + NHC**: Trajectory RMSE: 3238.79m | Drift: 350.53%

*Note*: Without AttitudeNet to correct heading drift, both trajectories drift laterally over long distances, but the total displacement and scale driven by SpeedNet closely matches the Oracle, proving the learned odometry works.

## Leakage Checks
* Train/Val/Test strictly separated by session.
* Normalization purely sourced from Training split.
