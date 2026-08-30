#include <navik/interfaces.hpp>
#include <iostream>

using namespace navik;

class MockSensorSource : public ISensorSource {
public:
    bool getNextImu(ImuSample& out) override {
        out.timestamp_ms = 1000;
        out.accel_m_s2 = Eigen::Vector3d(0, 9.81, 0);
        out.gyro_rad_s = Eigen::Vector3d::Zero();
        return true;
    }
    bool getNextGnss(GnssSample& out) override {
        return false;
    }
    bool getNextMag(MagSample& out) override {
        return false;
    }
};

class MockModelRunner : public IModelRunner {
public:
    bool loadModel(const std::string& config_path) override { return true; }
    bool predictSpeed(const std::vector<ImuSample>& imu_window, SpeedMeasurement& out) override {
        out.timestamp_ms = 1000;
        out.velocity_m_s = 15.0;
        out.log_variance = 0.1;
        return true;
    }
    bool predictMotionState(const std::vector<ImuSample>& imu_window, MotionState& out) override {
        return false;
    }
};

class MockFilter : public INavikFilter {
public:
    void initialize(const NavState& initial_state) override {}
    void predict(const ImuSample& imu, double dt_sec) override {}
    void updateGnss(const GnssSample& gnss) override {}
    void updateSpeed(const SpeedMeasurement& speed) override {}
    void updateZeroVelocity() override {}
    void updateZeroHeadingRate() override {}
    void updateNonHolonomic() override {}
    NavState getState() const override { return NavState{}; }
    void reset() override {}
};

int main() {
    std::cout << "Interfaces compiled successfully and mock instances instantiated." << std::endl;
    MockSensorSource sensor;
    ImuSample imu;
    if(sensor.getNextImu(imu)) {
        std::cout << "IMU Accel Y: " << imu.accel_m_s2.y() << std::endl;
    }
    return 0;
}
