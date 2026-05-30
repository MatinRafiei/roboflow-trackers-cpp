#include "sort_tracklet.hpp"

namespace trackers {
namespace core {
namespace sort {

SORTTracklet::SORTTracklet(const cv::Vec4f& initial_bbox, utils::StateRepresentation state_repr)
    : utils::BaseTracklet(initial_bbox, state_repr),
      number_of_successful_updates(1) // SORT counts the initial box as the first hit
{
    _configure_noise();
}

void SORTTracklet::_configure_noise() {
    auto& kf = state_estimator->kf;
    bool is_xcycsr = dynamic_cast<utils::XCYCSRStateEstimator*>(state_estimator.get()) != nullptr;

    if (is_xcycsr) {
        kf.R(cv::Rect(2, 2, 2, 2)) *= 10.0f;
        kf.P = cv::Mat::eye(kf.x.rows, kf.x.rows, CV_32F) * 10.0f;
        kf.P(cv::Rect(4, 4, 3, 3)) *= 1000.0f;
        kf.Q = cv::Mat::eye(kf.x.rows, kf.x.rows, CV_32F) * 0.01f;
        kf.Q.at<float>(6, 6) *= 0.01f;
        kf.Q(cv::Rect(4, 4, 3, 3)) *= 0.01f;
    } else {
        // XYXY Standard SORT Noise
        kf.Q = cv::Mat::eye(8, 8, CV_32F) * 0.01f;
        kf.R = cv::Mat::eye(4, 4, CV_32F);
        kf.R(cv::Rect(2, 2, 2, 2)) *= 10.0f;
        
        kf.P = cv::Mat::eye(8, 8, CV_32F) * 10.0f;
        kf.P(cv::Rect(4, 4, 4, 4)) *= 1000.0f;
    }
}

void SORTTracklet::update(const cv::Vec4f& bbox) {
    state_estimator->update(bbox);
    time_since_update = 0;
    number_of_successful_updates += 1;
}

cv::Vec4f SORTTracklet::predict() {
    state_estimator->predict();
    time_since_update += 1;
    age += 1;
    return state_estimator->state_to_bbox();
}

} // namespace sort
} // namespace core
} // namespace trackers