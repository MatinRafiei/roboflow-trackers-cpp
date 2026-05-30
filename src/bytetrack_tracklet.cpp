#include "bytetrack_tracklet.hpp"
#include <cmath>
#include <algorithm>

namespace trackers {
namespace core {
namespace bytetrack {

ByteTrackTracklet::ByteTrackTracklet(const cv::Vec4f& initial_bbox, utils::StateRepresentation state_repr)
    : utils::BaseTracklet(initial_bbox, state_repr) 
{
    _configure_noise();
    
    // ByteTrack counts the initial box as the first hit
    number_of_successful_consecutive_updates = 1;
}

void ByteTrackTracklet::_configure_noise() {
    // ByteTrack tuning: std weight position (1/20) and std weight velocity (1/160)
    float std_weight_position = 1.0f / 20.0f;
    float std_weight_velocity = 1.0f / 160.0f;

    auto& kf = state_estimator->kf;
    cv::Vec4f bbox = state_estimator->state_to_bbox();
    float w = std::max(bbox[2] - bbox[0], 1e-3f);
    float h = std::max(bbox[3] - bbox[1], 1e-3f);

    // Dynamic Process Noise (Q) and Measurement Noise (R) based on object scale
    kf.Q = cv::Mat::zeros(8, 8, CV_32F);
    kf.Q.at<float>(0, 0) = std::pow(std_weight_position * w, 2);
    kf.Q.at<float>(1, 1) = std::pow(std_weight_position * h, 2);
    kf.Q.at<float>(2, 2) = std::pow(std_weight_position * w, 2);
    kf.Q.at<float>(3, 3) = std::pow(std_weight_position * h, 2);
    kf.Q.at<float>(4, 4) = std::pow(std_weight_velocity * w, 2);
    kf.Q.at<float>(5, 5) = std::pow(std_weight_velocity * h, 2);
    kf.Q.at<float>(6, 6) = std::pow(std_weight_velocity * w, 2);
    kf.Q.at<float>(7, 7) = std::pow(std_weight_velocity * h, 2);

    kf.R = cv::Mat::zeros(4, 4, CV_32F);
    kf.R.at<float>(0, 0) = std::pow(std_weight_position * w, 2);
    kf.R.at<float>(1, 1) = std::pow(std_weight_position * h, 2);
    kf.R.at<float>(2, 2) = std::pow(std_weight_position * w, 2);
    kf.R.at<float>(3, 3) = std::pow(std_weight_position * h, 2);
}

void ByteTrackTracklet::update(const cv::Vec4f& bbox) {
    state_estimator->update(bbox);
    time_since_update = 0;
    number_of_successful_consecutive_updates += 1;
}

cv::Vec4f ByteTrackTracklet::predict() {
    state_estimator->predict();

    if (time_since_update > 0) {
        number_of_successful_consecutive_updates = 0;
    }
    time_since_update += 1;
    age += 1;
    
    return state_estimator->state_to_bbox();
}

} // namespace bytetrack
} // namespace core
} // namespace trackers