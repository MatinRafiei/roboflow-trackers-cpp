#include "base_tracker.hpp"

namespace trackers {
namespace core {

utils::Detections BaseTracker::get_tracked_objects() const {
    utils::Detections predicted_detections;
    
    // Pre-allocate memory to prevent fragmentation on the Edge CPU
    predicted_detections.reserve(tracks.size());

    for (const auto& track : tracks) {
        if (track->tracker_id != -1) {
            utils::Detection det;
            
            // Get the bounding box predicted by the Kalman Filter
            det.bbox = track->get_state_bbox();
            
            // Assign the tracker ID. 
            // Note: confidence and class_id are left as defaults since this is a pure prediction.
            det.tracker_id = track->tracker_id;
            
            predicted_detections.push_back(det);
        }
    }

    return predicted_detections;
}

} // namespace core
} // namespace trackers