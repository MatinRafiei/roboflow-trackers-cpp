#pragma once

#include <vector>
#include <memory>
#include <tuple>
#include <opencv2/opencv.hpp>
#include "base_tracker.hpp"
#include "bytetrack_tracklet.hpp"
#include "iou.hpp"
#include "detections.hpp"

namespace trackers {
namespace core {
namespace bytetrack {

class ByteTrackTracker : public BaseTracker {
private:
    int frame_id;
    
    // Hyperparameters
    float track_activation_threshold;
    float high_conf_det_threshold;
    float minimum_iou_threshold_first_assoc;
    float minimum_iou_threshold_second_assoc;
    float minimum_iou_threshold_unconfirmed_assoc;
    int minimum_consecutive_frames;
    bool instant_first_frame_activation;

    std::unique_ptr<utils::BaseIoU> iou;

    // Internal helpers
    cv::Mat _get_iou_matrix(const std::vector<utils::BaseTracklet*>& tracklet_pool, 
                            const std::vector<utils::Detection>& detections);

    std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>> 
    _get_associated_indices(const cv::Mat& similarity_matrix, float min_similarity_thresh, int n_tracks, int n_dets);

    void linear_sum_assignment(const cv::Mat& similarity_matrix, std::vector<int>& row_indices, std::vector<int>& col_indices);

public:
    ByteTrackTracker(
        int lost_track_buffer = 30,
        float frame_rate = 30.0f,
        float track_activation_threshold = 0.6f,
        float high_conf_det_threshold = 0.6f,
        float minimum_iou_threshold_first_assoc = 0.2f,
        float minimum_iou_threshold_second_assoc = 0.5f,
        float minimum_iou_threshold_unconfirmed_assoc = 0.3f,
        int minimum_consecutive_frames = 2,
        bool instant_first_frame_activation = true
    );

    utils::Detections update(const utils::Detections& detections, const cv::Mat& frame = cv::Mat()) override;
    void reset() override;
};

} // namespace bytetrack
} // namespace core
} // namespace trackers