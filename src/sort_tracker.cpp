#include "sort_tracker.hpp"
#include "Hungarian.h" 
#include <algorithm>
#include <set>

namespace trackers {
namespace core {
namespace sort {

SORTTracker::SORTTracker(
    int lost_track_buffer, float frame_rate, float track_activation_threshold,
    float high_conf_det_threshold, float minimum_iou_threshold, int minimum_consecutive_frames
) : 
    track_activation_threshold(track_activation_threshold),
    high_conf_det_threshold(high_conf_det_threshold),
    minimum_iou_threshold(minimum_iou_threshold),
    minimum_consecutive_frames(minimum_consecutive_frames)
{
    this->maximum_frames_without_update = static_cast<int>((frame_rate / 30.0f) * lost_track_buffer);
    iou = std::make_unique<utils::IoU>(); 
}

void SORTTracker::reset() {
    tracks.clear();
    utils::BaseTracklet::count_id = 0;
}

utils::Detections SORTTracker::update(const utils::Detections& detections, const cv::Mat& /*frame*/) {
    if (tracks.empty() && detections.empty()) return utils::Detections();

    utils::Detections out_detections;
    out_detections.reserve(detections.size());

    // 1. Predict all tracks
    for (auto& track : tracks) {
        track->predict();
    }

    // 2. Filter high confidence detections
    std::vector<int> high_indices;
    std::vector<utils::Detection> high_dets;
    for (size_t i = 0; i < detections.size(); ++i) {
        if (detections[i].confidence >= high_conf_det_threshold) {
            high_indices.push_back(i);
            high_dets.push_back(detections[i]);
        }
    }

    // 3. Compute IoU Cost Matrix
    std::vector<cv::Vec4f> track_boxes, det_boxes;
    for (const auto& t : tracks) track_boxes.push_back(t->get_state_bbox());
    for (const auto& d : high_dets) det_boxes.push_back(d.bbox);

    cv::Mat iou_mat = iou->compute(track_boxes, det_boxes);

    // 4. Hungarian Match
    std::vector<int> rows, cols;
    linear_sum_assignment(iou_mat, rows, cols);

    std::set<int> matched_dets;
    
    for (size_t i = 0; i < rows.size(); i++) {
        if (iou_mat.at<float>(rows[i], cols[i]) >= minimum_iou_threshold) {
            auto* sort_track = static_cast<SORTTracklet*>(tracks[rows[i]].get());
            sort_track->update(high_dets[cols[i]].bbox);
            
            // Assign ID if mature
            if (sort_track->number_of_successful_updates >= minimum_consecutive_frames && sort_track->tracker_id == -1) {
                sort_track->tracker_id = utils::BaseTracklet::get_next_tracker_id();
            }
            
            utils::Detection out_det = high_dets[cols[i]];
            out_det.tracker_id = sort_track->tracker_id;
            out_detections.push_back(out_det);
            
            matched_dets.insert(cols[i]);
        }
    }

    // 5. Spawn new tracks for unmatched detections
    for (size_t i = 0; i < high_dets.size(); i++) {
        if (matched_dets.find(i) == matched_dets.end() && high_dets[i].confidence >= track_activation_threshold) {
            auto tracklet = std::make_unique<SORTTracklet>(high_dets[i].bbox);
            
            // If minimum frames is 1, assign ID immediately
            if (minimum_consecutive_frames <= 1) {
                tracklet->tracker_id = utils::BaseTracklet::get_next_tracker_id();
            }
            
            utils::Detection out_det = high_dets[i];
            out_det.tracker_id = tracklet->tracker_id;
            tracks.push_back(std::move(tracklet));
            out_detections.push_back(out_det);
        }
    }

    // 6. Prune stale tracks (Replicates Python _get_alive_tracklets directly)
    tracks.erase(std::remove_if(tracks.begin(), tracks.end(),
        [this](const std::unique_ptr<utils::BaseTracklet>& track) {
            auto* sort_track = static_cast<SORTTracklet*>(track.get());
            bool is_mature = sort_track->number_of_successful_updates >= minimum_consecutive_frames;
            bool is_active = sort_track->time_since_update == 0;
            return !(sort_track->time_since_update < maximum_frames_without_update && (is_mature || is_active));
        }),
        tracks.end());

    return out_detections;
}

// Same Hungarian wrapper you used for the other trackers
void SORTTracker::linear_sum_assignment(const cv::Mat& sim_mat, std::vector<int>& rows, std::vector<int>& cols) {
    if (sim_mat.empty()) return;
    int n_rows = sim_mat.rows, n_cols = sim_mat.cols;
    std::vector<std::vector<double>> cost_mat(n_rows, std::vector<double>(n_cols, 0.0));
    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            cost_mat[i][j] = 1.0 - static_cast<double>(sim_mat.at<float>(i, j));
        }
    }
    HungarianAlgorithm hungarian;
    std::vector<int> assignment;
    hungarian.Solve(cost_mat, assignment);
    for (int x = 0; x < n_rows; x++) {
        if (assignment[x] >= 0) { rows.push_back(x); cols.push_back(assignment[x]); }
    }
}

} // namespace sort
} // namespace core
} // namespace trackers