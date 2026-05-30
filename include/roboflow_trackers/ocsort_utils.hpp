#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

namespace trackers {
namespace core {
namespace ocsort {

// Computes the direction consistency matrix (OCM) between tracks and detections
cv::Mat build_direction_consistency_matrix_batch(
    const std::vector<cv::Vec2f>& tracklet_velocities,
    const std::vector<cv::Vec4f>& reference_boxes,
    const std::vector<cv::Vec4f>& detection_boxes,
    const std::vector<float>& velocity_mask,
    float v_max = 0.0f
);

} // namespace ocsort
} // namespace core
} // namespace trackers