#pragma once

#include "base_tracker.hpp"
#include "ocsort_tracklet.hpp"
#include "iou.hpp"

namespace trackers {
namespace core {
namespace ocsort {

class OCSORTTracker : public BaseTracker {
private:
    float track_activation_threshold;
    float high_conf_det_threshold;
    float minimum_iou_threshold;
    int minimum_consecutive_frames;
    float direction_consistency_weight;
    int velocity_time_window;
    float max_velocity;
    std::unique_ptr<utils::BaseIoU> iou;

    void linear_sum_assignment(const cv::Mat& sim_mat, std::vector<int>& rows, std::vector<int>& cols);

public:
    OCSORTTracker(
        int lost_track_buffer = 30,
        float frame_rate = 30.0f,
        float track_activation_threshold = 0.3f,
        float high_conf_det_threshold = 0.3f,
        float minimum_iou_threshold = 0.3f,
        int minimum_consecutive_frames = 1,
        float direction_consistency_weight = 0.2f,
        int velocity_time_window = 3
    );

    utils::Detections update(const utils::Detections& detections, const cv::Mat& frame = cv::Mat()) override;
    void reset() override;
};

} // namespace ocsort
} // namespace core
} // namespace trackers