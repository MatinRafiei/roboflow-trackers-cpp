#include "base_tracklet.hpp"

namespace trackers {
namespace utils {

// Initialize the static ID counter (analogous to cls.count_id = 0)
int BaseTracklet::count_id = 0;

BaseTracklet::BaseTracklet(const cv::Vec4f& bbox, StateRepresentation state_repr)
    : age(0),
      tracker_id(-1),
      time_since_update(0),
      number_of_successful_consecutive_updates(0) 
{
    // Use the factory pattern we built in the previous step to dynamically
    // allocate the correct Kalman Filter state representation.
    state_estimator = create_state_estimator(state_repr, bbox);
}

int BaseTracklet::get_next_tracker_id() {
    int next_id = count_id;
    count_id++;
    return next_id;
}

cv::Vec4f BaseTracklet::get_state_bbox() const {
    if (state_estimator) {
        return state_estimator->state_to_bbox();
    }
    // Safe fallback, though mathematically state_estimator should never be null
    return cv::Vec4f(0.0f, 0.0f, 0.0f, 0.0f); 
}

} // namespace utils
} // namespace trackers