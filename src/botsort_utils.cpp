#include "botsort_utils.hpp"
#include <algorithm>

namespace trackers {
namespace core {
namespace botsort {

void filter_alive_tracklets(
    std::vector<std::unique_ptr<BoTSORTTracklet>>& tracklets,
    int minimum_consecutive_frames,
    int maximum_frames_without_update
) {
    // The Erase-Remove Idiom:
    // std::remove_if shifts all "kept" items to the front of the vector,
    // and tracklets.erase() deletes the remaining "dead" pointers at the end.
    // This automatically triggers the destructor (~BaseTracklet) freeing memory safely.
    tracklets.erase(
        std::remove_if(tracklets.begin(), tracklets.end(),
            [=](const std::unique_ptr<BoTSORTTracklet>& tracker) {
                // Note: We use the variable name defined in our BaseTracklet C++ conversion
                bool is_mature = tracker->number_of_successful_consecutive_updates >= minimum_consecutive_frames;
                bool is_active = tracker->time_since_update == 0;
                
                bool keep = (tracker->time_since_update < maximum_frames_without_update) && (is_mature || is_active);
                
                // std::remove_if requires us to return true for items we want to REMOVE
                return !keep; 
            }),
        tracklets.end()
    );
}

cv::Mat fuse_score(const cv::Mat& iou_similarity, const std::vector<float>& scores) {
    if (iou_similarity.empty() || scores.empty()) {
        return iou_similarity.clone();
    }

    int n_tracks = iou_similarity.rows;
    int n_dets = iou_similarity.cols;

    // Allocate the output matrix
    cv::Mat fused = cv::Mat(n_tracks, n_dets, CV_32F);

    // Standard C++ loop is faster than OpenCV broadcasting for small assignment matrices
    for (int i = 0; i < n_tracks; ++i) {
        for (int j = 0; j < n_dets; ++j) {
            fused.at<float>(i, j) = iou_similarity.at<float>(i, j) * scores[j];
        }
    }

    return fused;
}

} // namespace botsort
} // namespace core
} // namespace trackers