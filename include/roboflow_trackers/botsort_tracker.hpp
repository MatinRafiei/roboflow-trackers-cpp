#pragma once

#include <vector>
#include <memory>
#include <tuple>
#include <opencv2/opencv.hpp>
#include "base_tracker.hpp"
#include "botsort_tracklet.hpp"
#include "cmc.hpp"
#include "iou.hpp"
#include "detections.hpp"

namespace trackers {
namespace core {
namespace botsort {

class BoTSORTTracker : public BaseTracker {
private:
    int frame_id;
    
    // Hyperparameters
    int maximum_frames_without_update;
    int minimum_consecutive_frames;
    float minimum_iou_threshold_first_assoc;
    float minimum_iou_threshold_second_assoc;
    float minimum_iou_threshold_unconfirmed_assoc;
    float track_activation_threshold;
    float high_conf_det_threshold;
    bool instant_first_frame_activation;
    bool enable_cmc;

    // Components
    std::unique_ptr<utils::CMC> cmc;
    std::unique_ptr<utils::BaseIoU> iou;

    // Internal helpers
    cv::Mat _get_iou_matrix(const std::vector<utils::BaseTracklet*>& tracklet_pool, 
                            const std::vector<utils::Detection>& detections);

    std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>> 
    _get_associated_indices(const cv::Mat& similarity_matrix, float min_similarity_thresh, int n_tracks, int n_dets);

    void _spawn_new_tracks(
        const std::vector<utils::Detection>& detections,
        const std::vector<int>& unmatched_high_local,
        const std::vector<int>& high_indices,
        std::vector<utils::Detection>& out_detections,
        bool is_first_frame
    );

    // Placeholder wrapper for your chosen C++ Hungarian Algorithm library
    void linear_sum_assignment(const cv::Mat& cost_matrix, std::vector<int>& row_indices, std::vector<int>& col_indices);

public:
    BoTSORTTracker(
        int lost_track_buffer = 30,
        float frame_rate = 30.0f,
        float track_activation_threshold = 0.7f,
        int minimum_consecutive_frames = 2,
        float minimum_iou_threshold_first_assoc = 0.2f,
        float minimum_iou_threshold_second_assoc = 0.5f,
        float minimum_iou_threshold_unconfirmed_assoc = 0.3f,
        float high_conf_det_threshold = 0.6f,
        bool enable_cmc = true,
        utils::CMCMethod cmc_method = utils::CMCMethod::SPARSE_OPTFLOW,
        int cmc_downscale = 2,
        bool instant_first_frame_activation = true
    );

    utils::Detections update(const utils::Detections& detections, const cv::Mat& frame = cv::Mat()) override;
    void reset() override;
};

} // namespace botsort
} // namespace core
} // namespace trackers