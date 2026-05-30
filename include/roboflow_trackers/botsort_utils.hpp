#pragma once

#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "botsort_tracklet.hpp"

namespace trackers {
namespace core {
namespace botsort {

// Modifies the tracklets vector IN-PLACE to remove dead tracks,
// maximizing performance and memory safety on Edge devices.
void filter_alive_tracklets(
    std::vector<std::unique_ptr<BoTSORTTracklet>>& tracklets,
    int minimum_consecutive_frames,
    int maximum_frames_without_update
);

// Fuses the IoU matrix with the 1D array of detection confidence scores
cv::Mat fuse_score(const cv::Mat& iou_similarity, const std::vector<float>& scores);

} // namespace botsort
} // namespace core
} // namespace trackers