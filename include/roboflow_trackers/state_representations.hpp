#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include "kalman_filter.hpp"
#include "converters.hpp"

namespace trackers {
namespace utils {

// Enum to specify which state representation to use
enum class StateRepresentation {
    XCYCSR,
    XYXY,
    XCYCWH
};

// Abstract Base Class
class BaseStateEstimator {
public:
    KalmanFilter kf;

    // Use protected constructor so only derived classes can initialize the KF
    protected:
        BaseStateEstimator(int dim_x, int dim_z) : kf(dim_x, dim_z) {}

    public:
        virtual ~BaseStateEstimator() = default;

        // Core tracker functions
        void predict();
        void update(const cv::Vec4f& bbox);

        // Pure virtual functions that child classes MUST implement
        virtual cv::Mat bbox_to_measurement(const cv::Vec4f& bbox) = 0;
        virtual cv::Vec4f state_to_bbox() = 0;
        virtual void clamp_velocity() = 0;
};

// --- Concrete Implementations ---

class XCYCSRStateEstimator : public BaseStateEstimator {
public:
    XCYCSRStateEstimator(const cv::Vec4f& bbox);
    cv::Mat bbox_to_measurement(const cv::Vec4f& bbox) override;
    cv::Vec4f state_to_bbox() override;
    void clamp_velocity() override;
};

class XCYCWHStateEstimator : public BaseStateEstimator {
public:
    XCYCWHStateEstimator(const cv::Vec4f& bbox);
    cv::Mat bbox_to_measurement(const cv::Vec4f& bbox) override;
    cv::Vec4f state_to_bbox() override;
    void clamp_velocity() override;
};

class XYXYStateEstimator : public BaseStateEstimator {
public:
    XYXYStateEstimator(const cv::Vec4f& bbox);
    cv::Mat bbox_to_measurement(const cv::Vec4f& bbox) override;
    cv::Vec4f state_to_bbox() override;
    void clamp_velocity() override;
};

// --- Factory Function ---
std::unique_ptr<BaseStateEstimator> create_state_estimator(
    StateRepresentation state_repr,
    const cv::Vec4f& bbox
);

} // namespace utils
} // namespace trackers