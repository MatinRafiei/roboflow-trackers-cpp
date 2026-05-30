#pragma once

#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "base_tracker.hpp"
#include "sort_tracklet.hpp"
#include "iou.hpp"
#include "detections.hpp"

namespace trackers {
namespace core {
namespace sort {

class SORTTracker : public BaseTracker {
private:
    float track_activation_threshold;
    float high_conf_det_threshold;
    float minimum_iou_threshold;
    int minimum_consecutive_frames;
    std::unique_ptr<utils::BaseIoU> iou;

    void linear_sum_assignment(const cv::Mat& sim_mat, std::vector<int>& rows, std::vector<int>& cols);

public:
    SORTTracker(
        int lost_track_buffer = 30,
        float frame_rate = 30.0f,
        float track_activation_threshold = 0.3f,
        float high_conf_det_threshold = 0.3f,
        float minimum_iou_threshold = 0.3f,
        int minimum_consecutive_frames = 1
    );

    utils::Detections update(const utils::Detections& detections, const cv::Mat& frame = cv::Mat()) override;
    void reset() override;
};

} // namespace sort
} // namespace core
} // namespace trackers