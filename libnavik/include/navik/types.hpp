#pragma once

#include <cstdint>
#include <Eigen/Dense>

namespace navik {

/**
 * Coordinate Frame Conventions:
 * - Sensor Frame (s-frame): Right-Handed (x=Right, y=Forward, z=Up) relative to device.
 * - Vehicle Frame (v-frame): Right-Handed (x=Right, y=Forward, z=Up) relative to vehicle.
 * - Navigation Frame (n-frame): East-North-Up (ENU).
 * 
 * Timestamp Convention:
 * - uint64_t timestamp_ms: Milliseconds since Unix Epoch, or synchronized boot time.
 */

// Core Data Types

struct ImuSample {
    uint64_t timestamp_ms;
    Eigen::Vector3d accel_m_s2; // Specific force in s-frame (m/s^2)
    Eigen::Vector3d gyro_rad_s; // Angular velocity in s-frame (rad/s)
};

struct MagSample {
    uint64_t timestamp_ms;
    Eigen::Vector3d mag_uT; // Magnetic field in s-frame (micro-Tesla)
};

struct GnssSample {
    uint64_t timestamp_ms;
    double lat_deg;         // WGS84 Latitude (degrees)
    double lon_deg;         // WGS84 Longitude (degrees)
    double alt_m;           // Height above ellipsoid (meters)
    double speed_m_s;       // Ground speed (m/s)
    double heading_deg;     // Course over ground (degrees, True North)
    double horiz_acc_m;     // Horizontal accuracy estimate (meters)
    int num_satellites;     // Satellites used in solution
};

// Machine Learning Outputs

struct MotionState {
    uint64_t timestamp_ms;
    enum class State {
        STATIONARY = 0,
        STRAIGHT,
        TURNING,
        ACCEL,
        BRAKE,
        ROUGH_ROAD
    } state;
    double confidence;      // Probability [0.0, 1.0]
};

struct SpeedMeasurement {
    uint64_t timestamp_ms;
    double velocity_m_s;    // Forward longitudinal velocity (v-frame y-axis)
    double log_variance;    // Heteroscedastic uncertainty
};

// Filter State

struct NavState {
    uint64_t timestamp_ms;
    Eigen::Vector3d position_enu;   // Position in n-frame (meters) relative to a local origin
    Eigen::Vector3d velocity_enu;   // Velocity in n-frame (m/s)
    Eigen::Quaterniond attitude_ns; // Rotation from s-frame to n-frame
    Eigen::Vector3d accel_bias;     // Accelerometer bias (m/s^2)
    Eigen::Vector3d gyro_bias;      // Gyroscope bias (rad/s)
    
    // Covariance matrix diagonal (variances) for reporting
    Eigen::Matrix<double, 15, 1> state_variances;
};

} // namespace navik
