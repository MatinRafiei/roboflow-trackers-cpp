#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include "state_representations.hpp"

namespace trackers {
namespace utils {

class BaseTracklet {
protected:
public:
    // Static counter shared across ALL tracklets to generate unique IDs
    static int count_id;

    int age;
    std::unique_ptr<BaseStateEstimator> state_estimator;
    
    int tracker_id;
    int time_since_update;
    int number_of_successful_consecutive_updates;

    // Constructor
    BaseTracklet(const cv::Vec4f& bbox, StateRepresentation state_repr);

    // Virtual destructor is REQUIRED when using inheritance and unique_ptr
    virtual ~BaseTracklet() = default;

    // Class method for IDs
    static int get_next_tracker_id();

    // --- Abstract Methods (Pure Virtual) ---
    // Child classes (like BoTSORTTracklet) MUST implement these
    virtual void update(const cv::Vec4f& bbox) = 0;
    virtual cv::Vec4f predict() = 0;
    virtual void mark_lost() = 0;
    virtual void mark_removed() = 0;

    // --- Concrete Methods ---
    cv::Vec4f get_state_bbox() const;
};

} // namespace utils
} // namespace trackers