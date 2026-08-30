# App ↔ Engine Communication Contract

This document defines the JSON payload schema exchanged between the Mobile Application (Android/Flutter) and the Edge Navigation Engine (`libnavik`).

## Engine to App: `NavStateUpdate`
Sent by the engine at a high frequency (e.g., 10Hz) to update the UI.

```json
{
  "message_type": "NavStateUpdate",
  "version": "1.0",
  "timestamp_ms": 1693400000000,
  "system_mode": "FUSED", // Enum: INIT, GNSS_AIDED, DEAD_RECKON, FUSED, DEGRADED
  
  "position": {
    "frame": "WGS84",
    "lat_deg": 12.971598,
    "lon_deg": 77.594562,
    "alt_m": 920.0
  },
  
  "velocity": {
    "frame": "ENU",
    "speed_m_s": 15.2,
    "heading_deg": 45.0 // True North
  },
  
  "motion_state": {
    "state": "STRAIGHT", // Enum: STATIONARY, STRAIGHT, TURNING, ACCEL, BRAKE, ROUGH_ROAD
    "confidence": 0.95
  },

  "metrics": {
    "outage_distance_m": 150.0, // Distance traveled since last valid GNSS
    "estimated_drift_m": 1.2,   // Expected drift bound
    "filter_latency_ms": 2.5,
    "inference_latency_ms": 15.0
  }
}
```

## App to Engine: `SensorPayload` (If streaming from App to Engine)
Used if the App is the data provider for a remote edge engine.

```json
{
  "message_type": "SensorPayload",
  "version": "1.0",
  "timestamp_ms": 1693400000000,
  
  "imu": {
    "accel_x_m_s2": 0.0,
    "accel_y_m_s2": 9.81,
    "accel_z_m_s2": 0.0,
    "gyro_x_rad_s": 0.01,
    "gyro_y_rad_s": 0.0,
    "gyro_z_rad_s": -0.01
  },
  
  "gnss": {
    "valid": true,
    "lat_deg": 12.971598,
    "lon_deg": 77.594562,
    "speed_m_s": 15.2,
    "accuracy_m": 3.0,
    "satellites": 12
  }
}
```
