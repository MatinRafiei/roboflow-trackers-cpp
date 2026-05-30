#include "ocsort_tracklet.hpp"
#include "converters.hpp"
#include <cmath>

namespace trackers {
namespace core {
namespace ocsort {

OCSORTTracklet::OCSORTTracklet(const cv::Vec4f& initial_bbox, int velocity_time_window, utils::StateRepresentation state_repr)
    : utils::BaseTracklet(initial_bbox, state_repr), 
      has_velocity(false),
      delta_t(0),
      velocity_time_window(velocity_time_window)
{
    last_observation = initial_bbox;
    previous_to_last_observation = initial_bbox;
    velocity = cv::Vec2f(0.0f, 0.0f);
    observations[0] = initial_bbox;

    _configure_noise();
    number_of_successful_consecutive_updates = 1;
}

void OCSORTTracklet::_configure_noise() {
    auto& kf = state_estimator->kf;
    // OC-SORT specific noise scaling
    kf.P = cv::Mat::eye(kf.x.rows, kf.x.rows, CV_32F) * 10.0f;
    kf.P(cv::Rect(4, 4, 3, 3)) *= 1000.0f; 
    
    kf.Q = cv::Mat::eye(kf.x.rows, kf.x.rows, CV_32F) * 0.01f;
    kf.Q(cv::Rect(4, 4, 3, 3)) *= 0.01f;
    if(kf.x.rows == 7) kf.Q.at<float>(6, 6) *= 0.01f; // XCYCSR scale var
}

void OCSORTTracklet::update(const cv::Vec4f& bbox) {
    if (time_since_update > 0) {
        generate_virtual_trajectory(bbox); // ORU logic
    } else {
        state_estimator->update(bbox);
    }
    
    last_observation = bbox;
    observations[age] = bbox;
    update_velocity();

    time_since_update = 0;
    number_of_successful_consecutive_updates += 1;
}

cv::Vec4f OCSORTTracklet::predict() {
    state_estimator->predict();
    
    if (time_since_update > 0) number_of_successful_consecutive_updates = 0;
    
    time_since_update += 1;
    age += 1;
    return state_estimator->state_to_bbox();
}

void OCSORTTracklet::update_velocity() {
    if (observations.count(age - delta_t)) {
        previous_to_last_observation = observations[age - delta_t];
        float cx1 = (previous_to_last_observation[0] + previous_to_last_observation[2]) * 0.5f;
        float cy1 = (previous_to_last_observation[1] + previous_to_last_observation[3]) * 0.5f;
        float cx2 = (last_observation[0] + last_observation[2]) * 0.5f;
        float cy2 = (last_observation[1] + last_observation[3]) * 0.5f;
        
        velocity[0] = (cx2 - cx1) / static_cast<float>(delta_t);
        velocity[1] = (cy2 - cy1) / static_cast<float>(delta_t);
        has_velocity = true;
    }
    if (delta_t < velocity_time_window) delta_t++;
}

cv::Vec4f OCSORTTracklet::get_k_previous_obs() const {
    if (observations.count(age - delta_t)) return observations.at(age - delta_t);
    return last_observation;
}

void OCSORTTracklet::freeze() {
    frozen_x = state_estimator->kf.x.clone();
    frozen_P = state_estimator->kf.P.clone();
    frozen_F = state_estimator->kf.F.clone();
}

void OCSORTTracklet::unfreeze() {
    state_estimator->kf.x = frozen_x.clone();
    state_estimator->kf.P = frozen_P.clone();
    state_estimator->kf.F = frozen_F.clone();
}

void OCSORTTracklet::generate_virtual_trajectory(const cv::Vec4f& current_bbox) {
    // Interpolate virtual boxes for missed frames
    unfreeze(); // Restore state to when it was lost
    
    float dx = (current_bbox[0] - last_observation[0]) / (time_since_update);
    float dy = (current_bbox[1] - last_observation[1]) / (time_since_update);
    float dw = (current_bbox[2] - last_observation[2]) / (time_since_update);
    float dh = (current_bbox[3] - last_observation[3]) / (time_since_update);

    // Fast-forward Kalman filter with virtual boxes
    for (int i = 0; i < time_since_update; i++) {
        cv::Vec4f virtual_box(
            last_observation[0] + dx * (i + 1),
            last_observation[1] + dy * (i + 1),
            last_observation[2] + dw * (i + 1),
            last_observation[3] + dh * (i + 1)
        );
        state_estimator->predict();
        state_estimator->update(virtual_box);
    }
}

} // namespace ocsort
} // namespace core
} // namespace trackers