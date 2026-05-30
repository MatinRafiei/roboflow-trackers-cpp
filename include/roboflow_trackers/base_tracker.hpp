#pragma once

#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "detections.hpp"
#include "base_tracklet.hpp"

namespace trackers {
namespace core {

class BaseTracker {
protected:
    // In C++, the base class holds the polymorphic pointers to the active tracks.
    // Derived classes (like BoTSORTTracker) will populate this with their specific tracklets.
    std::vector<std::unique_ptr<utils::BaseTracklet>> tracks;
    
    int maximum_frames_without_update;

public:
    virtual ~BaseTracker() = default;

    /**
     * @brief Process new detections and assign track IDs.
     * Pure virtual function that MUST be implemented by child trackers.
     * * @param detections Current frame detections.
     * @param frame Current video frame (used for CMC). Can be empty.
     * @return Detections enriched with assigned tracker_ids.
     */
    virtual utils::Detections update(const utils::Detections& detections, const cv::Mat& frame = cv::Mat()) = 0;

    /**
     * @brief Clear all internal tracking state.
     */
    virtual void reset() = 0;

    /**
     * @brief Get all confirmed alive tracks with Kalman-predicted bounding boxes.
     * Exposes every confirmed track (tracker_id != -1) that the tracker still considers alive.
     * * @return Detections vector containing predicted locations (no confidence or class_id).
     */
    utils::Detections get_tracked_objects() const;
};

} // namespace core
} // namespace trackers