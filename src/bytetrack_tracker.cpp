#include "bytetrack_tracker.hpp"
#include "Hungarian.h" 
#include <algorithm>
#include <set>

namespace trackers {
namespace core {
namespace bytetrack {

ByteTrackTracker::ByteTrackTracker(
    int lost_track_buffer, float frame_rate, float track_activation_threshold,
    float high_conf_det_threshold, float minimum_iou_threshold_first_assoc,
    float minimum_iou_threshold_second_assoc, float minimum_iou_threshold_unconfirmed_assoc,
    int minimum_consecutive_frames, bool instant_first_frame_activation
) : 
    frame_id(0),
    track_activation_threshold(track_activation_threshold),
    high_conf_det_threshold(high_conf_det_threshold),
    minimum_iou_threshold_first_assoc(minimum_iou_threshold_first_assoc),
    minimum_iou_threshold_second_assoc(minimum_iou_threshold_second_assoc),
    minimum_iou_threshold_unconfirmed_assoc(minimum_iou_threshold_unconfirmed_assoc),
    minimum_consecutive_frames(minimum_consecutive_frames),
    instant_first_frame_activation(instant_first_frame_activation)
{
    this->maximum_frames_without_update = static_cast<int>((frame_rate / 30.0f) * lost_track_buffer);
    iou = std::make_unique<utils::IoU>(); 
}

void ByteTrackTracker::reset() {
    tracks.clear();
    frame_id = 0;
    utils::BaseTracklet::count_id = 0;
}

utils::Detections ByteTrackTracker::update(const utils::Detections& detections, const cv::Mat& /*frame*/) {
    frame_id++;

    if (tracks.empty() && detections.empty()) return utils::Detections();

    utils::Detections out_detections;
    out_detections.reserve(detections.size());

    // 1. Predict
    for (auto& track : tracks) track->predict();

    // 2. Split detections
    std::vector<int> high_indices, low_indices;
    std::vector<utils::Detection> high_dets, low_dets;

    for (size_t i = 0; i < detections.size(); ++i) {
        if (detections[i].confidence >= high_conf_det_threshold) {
            high_indices.push_back(i);
            high_dets.push_back(detections[i]);
        } else if (detections[i].confidence > 0.1f) {
            low_indices.push_back(i);
            low_dets.push_back(detections[i]);
        }
    }

    // 3. Split tracks
    std::vector<utils::BaseTracklet*> tracked_pool, unconfirmed_tracks;
    for (auto& track : tracks) {
        if (track->number_of_successful_consecutive_updates >= minimum_consecutive_frames || track->tracker_id != -1) {
            tracked_pool.push_back(track.get());
        } else {
            unconfirmed_tracks.push_back(track.get());
        }
    }

    // 4. First Association (Tracked vs High Dets)
    cv::Mat iou_mat = _get_iou_matrix(tracked_pool, high_dets);
    auto [matched_1, unmatched_tracks_1, unmatched_high] = _get_associated_indices(iou_mat, minimum_iou_threshold_first_assoc, tracked_pool.size(), high_dets.size());

    for (const auto& match : matched_1) {
        auto* track = tracked_pool[match.first];
        int col = match.second;
        track->update(high_dets[col].bbox);
        
        if (track->tracker_id == -1) track->tracker_id = utils::BaseTracklet::get_next_tracker_id();
        
        utils::Detection out_det = high_dets[col];
        out_det.tracker_id = track->tracker_id;
        out_detections.push_back(out_det);
    }

    // 5. Second Association (Remaining Tracked vs Low Dets)
    std::vector<utils::BaseTracklet*> remaining_tracked;
    for (int idx : unmatched_tracks_1) {
        if (tracked_pool[idx]->time_since_update == 1) { 
            remaining_tracked.push_back(tracked_pool[idx]);
        }
    }

    cv::Mat iou_mat_low = _get_iou_matrix(remaining_tracked, low_dets);
    auto [matched_2, unmatched_tracks_2, unmatched_low] = _get_associated_indices(iou_mat_low, minimum_iou_threshold_second_assoc, remaining_tracked.size(), low_dets.size());

    for (const auto& match : matched_2) {
        auto* track = remaining_tracked[match.first];
        int col = match.second;
        track->update(low_dets[col].bbox);
        
        if (track->tracker_id == -1) track->tracker_id = utils::BaseTracklet::get_next_tracker_id();
        
        utils::Detection out_det = low_dets[col];
        out_det.tracker_id = track->tracker_id;
        out_detections.push_back(out_det);
    }

    // 6. Third Association (Unconfirmed vs Remaining High Dets)
    std::vector<utils::Detection> remaining_high_dets;
    for (int idx : unmatched_high) remaining_high_dets.push_back(high_dets[idx]);
    
    std::set<utils::BaseTracklet*> tracks_to_remove;
    std::vector<int> final_unmatched_high = unmatched_high;

    if (!unconfirmed_tracks.empty() && !remaining_high_dets.empty()) {
        cv::Mat iou_mat_uc = _get_iou_matrix(unconfirmed_tracks, remaining_high_dets);
        auto [matched_uc, unmatched_uc_indices, remaining_uh] = _get_associated_indices(iou_mat_uc, minimum_iou_threshold_unconfirmed_assoc, unconfirmed_tracks.size(), remaining_high_dets.size());

        for (const auto& match : matched_uc) {
            auto* track = unconfirmed_tracks[match.first];
            int orig_high_idx = unmatched_high[match.second];
            track->update(high_dets[orig_high_idx].bbox);
            
            if (track->tracker_id == -1) track->tracker_id = utils::BaseTracklet::get_next_tracker_id();
            
            utils::Detection out_det = high_dets[orig_high_idx];
            out_det.tracker_id = track->tracker_id;
            out_detections.push_back(out_det);
        }

        for (int idx : unmatched_uc_indices) tracks_to_remove.insert(unconfirmed_tracks[idx]);

        final_unmatched_high.clear();
        for (int idx : remaining_uh) final_unmatched_high.push_back(unmatched_high[idx]);
    } else {
        for (auto* t : unconfirmed_tracks) tracks_to_remove.insert(t);
    }

    // 7. Prune unconfirmed
    tracks.erase(std::remove_if(tracks.begin(), tracks.end(), 
        [&](const std::unique_ptr<utils::BaseTracklet>& t) { return tracks_to_remove.count(t.get()) > 0; }),
        tracks.end());

    // 8. Spawn new tracks
    for (int local_idx : final_unmatched_high) {
        int global_idx = high_indices[local_idx];
        if (detections[global_idx].confidence >= track_activation_threshold) {
            auto tracklet = std::make_unique<ByteTrackTracklet>(detections[global_idx].bbox);
            if (frame_id == 1 && instant_first_frame_activation) {
                tracklet->tracker_id = utils::BaseTracklet::get_next_tracker_id();
            }
            utils::Detection out_det = detections[global_idx];
            out_det.tracker_id = tracklet->tracker_id;
            tracks.push_back(std::move(tracklet));
            out_detections.push_back(out_det);
        }
    }

    // 9. Prune stale tracks
    tracks.erase(std::remove_if(tracks.begin(), tracks.end(),
        [this](const std::unique_ptr<utils::BaseTracklet>& t) {
            bool is_mature = t->tracker_id != -1 || t->number_of_successful_consecutive_updates >= minimum_consecutive_frames;
            bool is_active = t->time_since_update == 0;
            return !(t->time_since_update < maximum_frames_without_update && (is_mature || is_active));
        }),
        tracks.end());

    return out_detections;
}

// --------------------------------------------------------
// Internal Utils (Identical to BoTSORT)
// --------------------------------------------------------

cv::Mat ByteTrackTracker::_get_iou_matrix(const std::vector<utils::BaseTracklet*>& pool, const std::vector<utils::Detection>& dets) {
    if (pool.empty() || dets.empty()) return cv::Mat::zeros(pool.size(), dets.size(), CV_32F);
    std::vector<cv::Vec4f> track_b, det_b;
    for (const auto* t : pool) track_b.push_back(t->get_state_bbox());
    for (const auto& d : dets) det_b.push_back(d.bbox);
    return iou->compute(track_b, det_b);
}

std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>> 
ByteTrackTracker::_get_associated_indices(const cv::Mat& sim_mat, float thresh, int n_tracks, int n_dets) {
    std::vector<std::pair<int, int>> matched;
    std::set<int> u_tracks; for(int i=0; i<n_tracks; i++) u_tracks.insert(i);
    std::set<int> u_dets;   for(int i=0; i<n_dets; i++) u_dets.insert(i);

    if (n_tracks > 0 && n_dets > 0) {
        std::vector<int> row_indices, col_indices;
        linear_sum_assignment(sim_mat, row_indices, col_indices);
        for (size_t i = 0; i < row_indices.size(); ++i) {
            if (sim_mat.at<float>(row_indices[i], col_indices[i]) >= thresh) {
                matched.push_back({row_indices[i], col_indices[i]});
                u_tracks.erase(row_indices[i]);
                u_dets.erase(col_indices[i]);
            }
        }
    }
    return {matched, std::vector<int>(u_tracks.begin(), u_tracks.end()), std::vector<int>(u_dets.begin(), u_dets.end())};
}

void ByteTrackTracker::linear_sum_assignment(const cv::Mat& sim_mat, std::vector<int>& rows, std::vector<int>& cols) {
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
        if (assignment[x] >= 0) {
            rows.push_back(x);
            cols.push_back(assignment[x]);
        }
    }
}

} // namespace bytetrack
} // namespace core
} // namespace trackers