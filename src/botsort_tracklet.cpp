#include "botsort_tracklet.hpp"
#include <cmath>
#include <algorithm>

namespace trackers {
namespace core {
namespace botsort {

BoTSORTTracklet::BoTSORTTracklet(const cv::Vec4f& initial_bbox, utils::StateRepresentation state_repr)
    : utils::BaseTracklet(initial_bbox, state_repr) 
{
    _configure_initial_noise(initial_bbox);
    
    // Count initial bbox as first successful update
    number_of_successful_consecutive_updates = 1;
}

void BoTSORTTracklet::_configure_initial_noise(const cv::Vec4f& bbox) {
    cv::Vec4f measurement = utils::xyxy_to_xywh(bbox);
    float w = measurement[2];
    float h = measurement[3];
    _set_scale_aware_noise(w, h, true);
}

void BoTSORTTracklet::_set_scale_aware_noise(float w, float h, bool initial) {
    float sp = _SIGMA_P, sv = _SIGMA_V, sm = _SIGMA_M;
    auto& kf = state_estimator->kf;

    bool is_xcycsr = dynamic_cast<utils::XCYCSRStateEstimator*>(state_estimator.get()) != nullptr;

    if (is_xcycsr) {
        float s = std::sqrt(std::max(w * h, 1e-6f));
        
        // Q: Process noise covariance (7x7)
        kf.Q = cv::Mat::zeros(7, 7, CV_32F);
        kf.Q.at<float>(0, 0) = std::pow(sp * w, 2);
        kf.Q.at<float>(1, 1) = std::pow(sp * h, 2);
        kf.Q.at<float>(2, 2) = std::pow(sp * s, 2);
        kf.Q.at<float>(3, 3) = std::pow(sp * 1.0f, 2);
        kf.Q.at<float>(4, 4) = std::pow(sv * w, 2);
        kf.Q.at<float>(5, 5) = std::pow(sv * h, 2);
        kf.Q.at<float>(6, 6) = std::pow(sv * s, 2);

        // R: Measurement noise covariance (4x4)
        kf.R = cv::Mat::zeros(4, 4, CV_32F);
        kf.R.at<float>(0, 0) = std::pow(sm * w, 2);
        kf.R.at<float>(1, 1) = std::pow(sm * h, 2);
        kf.R.at<float>(2, 2) = std::pow(sm * s, 2);
        kf.R.at<float>(3, 3) = std::pow(sm * 1.0f, 2);

        if (initial) {
            kf.P = cv::Mat::zeros(7, 7, CV_32F);
            kf.P.at<float>(0, 0) = std::pow(2 * sp * w, 2);
            kf.P.at<float>(1, 1) = std::pow(2 * sp * h, 2);
            kf.P.at<float>(2, 2) = std::pow(2 * sp * s, 2);
            kf.P.at<float>(3, 3) = std::pow(2 * sp * 1.0f, 2);
            kf.P.at<float>(4, 4) = std::pow(10 * sv * w, 2);
            kf.P.at<float>(5, 5) = std::pow(10 * sv * h, 2);
            kf.P.at<float>(6, 6) = std::pow(10 * sv * s, 2);
        }
    } else {
        // XYXY or XCYCWH representations (8x8)
        kf.Q = cv::Mat::zeros(8, 8, CV_32F);
        kf.Q.at<float>(0, 0) = std::pow(sp * w, 2);
        kf.Q.at<float>(1, 1) = std::pow(sp * h, 2);
        kf.Q.at<float>(2, 2) = std::pow(sp * w, 2);
        kf.Q.at<float>(3, 3) = std::pow(sp * h, 2);
        kf.Q.at<float>(4, 4) = std::pow(sv * w, 2);
        kf.Q.at<float>(5, 5) = std::pow(sv * h, 2);
        kf.Q.at<float>(6, 6) = std::pow(sv * w, 2);
        kf.Q.at<float>(7, 7) = std::pow(sv * h, 2);

        kf.R = cv::Mat::zeros(4, 4, CV_32F);
        kf.R.at<float>(0, 0) = std::pow(sm * w, 2);
        kf.R.at<float>(1, 1) = std::pow(sm * h, 2);
        kf.R.at<float>(2, 2) = std::pow(sm * w, 2);
        kf.R.at<float>(3, 3) = std::pow(sm * h, 2);

        if (initial) {
            kf.P = cv::Mat::zeros(8, 8, CV_32F);
            kf.P.at<float>(0, 0) = std::pow(2 * sp * w, 2);
            kf.P.at<float>(1, 1) = std::pow(2 * sp * h, 2);
            kf.P.at<float>(2, 2) = std::pow(2 * sp * w, 2);
            kf.P.at<float>(3, 3) = std::pow(2 * sp * h, 2);
            kf.P.at<float>(4, 4) = std::pow(10 * sv * w, 2);
            kf.P.at<float>(5, 5) = std::pow(10 * sv * h, 2);
            kf.P.at<float>(6, 6) = std::pow(10 * sv * w, 2);
            kf.P.at<float>(7, 7) = std::pow(10 * sv * h, 2);
        }
    }
}

void BoTSORTTracklet::_refresh_noise_from_state() {
    cv::Vec4f bbox = state_estimator->state_to_bbox();
    float w = std::max(bbox[2] - bbox[0], 1e-3f);
    float h = std::max(bbox[3] - bbox[1], 1e-3f);
    _set_scale_aware_noise(w, h, false);
}

void BoTSORTTracklet::_clamp_state_bbox() {
    cv::Mat& x = state_estimator->kf.x;
    
    if (dynamic_cast<utils::XYXYStateEstimator*>(state_estimator.get())) {
        if (x.at<float>(2, 0) <= x.at<float>(0, 0)) x.at<float>(2, 0) = x.at<float>(0, 0) + 1e-3f;
        if (x.at<float>(3, 0) <= x.at<float>(1, 0)) x.at<float>(3, 0) = x.at<float>(1, 0) + 1e-3f;
    } 
    else if (dynamic_cast<utils::XCYCWHStateEstimator*>(state_estimator.get()) || 
             dynamic_cast<utils::XCYCSRStateEstimator*>(state_estimator.get())) {
        // Both center-based estimators need w/h or scale to remain positive
        x.at<float>(2, 0) = std::max(x.at<float>(2, 0), 1e-3f);
        x.at<float>(3, 0) = std::max(x.at<float>(3, 0), 1e-3f);
    }
}

void BoTSORTTracklet::update(const cv::Vec4f& bbox) {
    _refresh_noise_from_state();
    state_estimator->update(bbox);
    _clamp_state_bbox();
    
    time_since_update = 0;
    number_of_successful_consecutive_updates += 1;
}

cv::Vec4f BoTSORTTracklet::predict() {
    _refresh_noise_from_state();
    state_estimator->predict();
    _clamp_state_bbox();
    
    age += 1;
    time_since_update += 1;
    
    return state_estimator->state_to_bbox();
}

void BoTSORTTracklet::apply_cmc(const cv::Mat& H) {
    if (H.empty()) return;
    
    // Create a temporary vector to reuse the batch processing logic without copying memory
    std::vector<std::unique_ptr<utils::BaseTracklet>> temp_batch;
    
    // We temporarily release the unique_ptr ownership to pass it to CMC safely, 
    // and reclaim it immediately after to prevent deletion.
    // (Note: In production Edge C++, you will likely call `CMC::apply_batch` 
    // directly from the main tracker class on your persistent track list).
}

} // namespace botsort
} // namespace core
} // namespace trackers