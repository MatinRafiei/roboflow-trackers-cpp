#include "ocsort_utils.hpp"
#include <cmath>
#include <algorithm>

namespace trackers {
namespace core {
namespace ocsort {

cv::Mat build_direction_consistency_matrix_batch(
    const std::vector<cv::Vec2f>& tracklet_velocities,
    const std::vector<cv::Vec4f>& reference_boxes,
    const std::vector<cv::Vec4f>& detection_boxes,
    const std::vector<float>& velocity_mask,
    float v_max
) {
    int n_tracks = tracklet_velocities.size();
    int n_dets = detection_boxes.size();

    if (n_tracks == 0 || n_dets == 0) {
        return cv::Mat::zeros(n_tracks, n_dets, CV_32F);
    }

    cv::Mat consistency = cv::Mat::zeros(n_tracks, n_dets, CV_32F);

    for (int i = 0; i < n_tracks; ++i) {
        float cx1 = (reference_boxes[i][0] + reference_boxes[i][2]) * 0.5f;
        float cy1 = (reference_boxes[i][1] + reference_boxes[i][3]) * 0.5f;
        
        float vx = tracklet_velocities[i][0];
        float vy = tracklet_velocities[i][1];
        float v_norm = std::sqrt(vx * vx + vy * vy);
        
        // Normalize track velocity
        float dir_y = (v_norm > 1e-6f) ? (vy / v_norm) : 0.0f;
        float dir_x = (v_norm > 1e-6f) ? (vx / v_norm) : 0.0f;

        for (int j = 0; j < n_dets; ++j) {
            float cx2 = (detection_boxes[j][0] + detection_boxes[j][2]) * 0.5f;
            float cy2 = (detection_boxes[j][1] + detection_boxes[j][3]) * 0.5f;

            float dy = cy2 - cy1;
            float dx = cx2 - cx1;
            float dist = std::sqrt(dx * dx + dy * dy);

            // Normalize distance vector
            float norm_dy = (dist > 1e-6f) ? (dy / dist) : 0.0f;
            float norm_dx = (dist > 1e-6f) ? (dx / dist) : 0.0f;

            // Cosine similarity
            float diff_angle_cos = dir_x * norm_dx + dir_y * norm_dy;
            diff_angle_cos = std::clamp(diff_angle_cos, -1.0f, 1.0f);

            // Scale from [-1, 1] to [0, 1]
            float score = (diff_angle_cos + 1.0f) * 0.5f;

            // Apply velocity magnitude weighting if v_max is provided
            if (v_max > 0.0f) {
                float v_scale = std::exp(-(v_norm / v_max));
                score = score * (1.0f - v_scale) + v_scale;
            }

            // Apply mask (if track has no velocity history, mask = 0)
            consistency.at<float>(i, j) = score * velocity_mask[i];
        }
    }

    return consistency;
}

} // namespace ocsort
} // namespace core
} // namespace trackers