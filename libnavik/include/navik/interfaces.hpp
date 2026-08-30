#pragma once

#include "types.hpp"
#include <vector>
#include <memory>

namespace navik {

/**
 * @brief Interface for providing sensor data stream.
 */
class ISensorSource {
public:
    virtual ~ISensorSource() = default;
    
    // Non-blocking polls, return false if no new data available
    virtual bool getNextImu(ImuSample& out) = 0;
    virtual bool getNextGnss(GnssSample& out) = 0;
    virtual bool getNextMag(MagSample& out) = 0;
};

/**
 * @brief Interface for ML Model execution.
 * Allows decoupling PyTorch/TFLite/ONNX from the engine.
 */
class IModelRunner {
public:
    virtual ~IModelRunner() = default;

    // Initialize with a model file path
    virtual bool loadModel(const std::string& config_path) = 0;

    // Feed a window of IMU data and predict speed
    virtual bool predictSpeed(const std::vector<ImuSample>& imu_window, SpeedMeasurement& out) = 0;

    // Feed a window of IMU data and predict motion state
    virtual bool predictMotionState(const std::vector<ImuSample>& imu_window, MotionState& out) = 0;
};

/**
 * @brief Abstract filter interface (ESKF, InEKF).
 */
class INavikFilter {
public:
    virtual ~INavikFilter() = default;

    virtual void initialize(const NavState& initial_state) = 0;
    virtual void predict(const ImuSample& imu, double dt_sec) = 0;
    
    // Updates
    virtual void updateGnss(const GnssSample& gnss) = 0;
    virtual void updateSpeed(const SpeedMeasurement& speed) = 0;
    virtual void updateZeroVelocity() = 0; // ZUPT
    virtual void updateZeroHeadingRate() = 0; // ZIHR
    virtual void updateNonHolonomic() = 0; // NHC

    virtual NavState getState() const = 0;
    virtual void reset() = 0;
};

/**
 * @brief Offline OSM Map Matching Interface
 */
class IMapMatcher {
public:
    virtual ~IMapMatcher() = default;

    virtual bool loadMap(const std::string& map_path) = 0;

    // Takes estimated state, returns map-constrained position and heading
    virtual bool matchTrajectory(const NavState& estimated_state, NavState& out) = 0;
};

/**
 * @brief Core Orchestrator Engine.
 */
class NavikEngine {
public:
    NavikEngine(std::shared_ptr<INavikFilter> filter, 
                std::shared_ptr<IModelRunner> model_runner,
                std::shared_ptr<IMapMatcher> map_matcher);
    
    virtual ~NavikEngine() = default;

    void initialize(const std::string& config_path);
    void processImu(const ImuSample& imu);
    void processGnss(const GnssSample& gnss);
    void processMag(const MagSample& mag);

    NavState getCurrentState() const;
    void reset();

private:
    std::shared_ptr<INavikFilter> filter_;
    std::shared_ptr<IModelRunner> model_runner_;
    std::shared_ptr<IMapMatcher> map_matcher_;
    
    // internal buffers and state tracking
};

} // namespace navik
