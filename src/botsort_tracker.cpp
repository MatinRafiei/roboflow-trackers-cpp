#include "botsort_tracker.hpp"
#include "botsort_utils.hpp"
#include <algorithm>
#include <set>
#include "Hungarian.h"

namespace trackers {
namespace core {
namespace botsort {

BoTSORTTracker::BoTSORTTracker(
    int lost_track_buffer, float frame_rate, float track_activation_threshold,
    int minimum_consecutive_frames, float minimum_iou_threshold_first_assoc,
    float minimum_iou_threshold_second_assoc, float minimum_iou_threshold_unconfirmed_assoc,
    float high_conf_det_threshold, bool enable_cmc, utils::CMCMethod cmc_method,
    int cmc_downscale, bool instant_first_frame_activation
) : 
    frame_id(0),
    minimum_consecutive_frames(minimum_consecutive_frames),
    minimum_iou_threshold_first_assoc(minimum_iou_threshold_first_assoc),
    minimum_iou_threshold_second_assoc(minimum_iou_threshold_second_assoc),
    minimum_iou_threshold_unconfirmed_assoc(minimum_iou_threshold_unconfirmed_assoc),
    track_activation_threshold(track_activation_threshold),
    high_conf_det_threshold(high_conf_det_threshold),
    instant_first_frame_activation(instant_first_frame_activation),
    enable_cmc(enable_cmc)
{
    this->maximum_frames_without_update = static_cast<int>((frame_rate / 30.0f) * lost_track_buffer);
    
    if (enable_cmc) {
        utils::CMCConfig cfg;
        cfg.method = cmc_method;
        cfg.downscale = cmc_downscale;
        cmc = std::make_unique<utils::CMC>(cfg);
    }
    
    iou = std::make_unique<utils::IoU>(); // Defaults to standard IoU
}

void BoTSORTTracker::reset() {
    tracks.clear();
    frame_id = 0;
    utils::BaseTracklet::count_id = 0;
    if (cmc) cmc->reset();
}

utils::Detections BoTSORTTracker::update(const utils::Detections& detections, const cv::Mat& frame) {
    frame_id++;

    if (tracks.empty() && detections.empty()) {
        return utils::Detections();
    }

    utils::Detections out_detections;
    out_detections.reserve(detections.size());

    // 1. Predict existing tracks
    for (auto& track : tracks) {
        track->predict();
    }

    // 2. Split detections by confidence
    std::vector<int> high_indices, low_indices;
    std::vector<utils::Detection> high_dets, low_dets;
    std::vector<float> high_scores;

    for (size_t i = 0; i < detections.size(); ++i) {
        if (detections[i].confidence >= high_conf_det_threshold) {
            high_indices.push_back(i);
            high_dets.push_back(detections[i]);
            high_scores.push_back(detections[i].confidence);
        } else if (detections[i].confidence > 0.1f) {
            low_indices.push_back(i);
            low_dets.push_back(detections[i]);
        }
    }

    // 3. Split tracks into categories using raw pointers (views)
    std::vector<utils::BaseTracklet*> confirmed_tracks, unconfirmed_tracks, lost_tracks;
    for (auto& track : tracks) {
        if (track->time_since_update > 1) {
            lost_tracks.push_back(track.get());
        } else if (track->number_of_successful_consecutive_updates >= minimum_consecutive_frames) {
            confirmed_tracks.push_back(track.get());
        } else {
            unconfirmed_tracks.push_back(track.get());
        }
    }

    // 4. CMC Application
    if (enable_cmc && cmc && !frame.empty()) {
        std::vector<cv::Vec4f> mask_boxes;
        for (const auto& d : high_dets) mask_boxes.push_back(d.bbox);
        cv::Mat H = cmc->estimate(frame, mask_boxes);
        utils::CMC::apply_batch(H, tracks);
    }

    // 5. First Association (Confirmed + Lost vs High Conf)
    std::vector<utils::BaseTracklet*> strack_pool = confirmed_tracks;
    strack_pool.insert(strack_pool.end(), lost_tracks.begin(), lost_tracks.end());

    cv::Mat iou_mat = _get_iou_matrix(strack_pool, high_dets);
    iou_mat = fuse_score(iou->normalize_for_fusion(iou_mat), high_scores);
    
    auto [matched_1, unmatched_tracks_1, unmatched_high] = _get_associated_indices(iou_mat, minimum_iou_threshold_first_assoc, strack_pool.size(), high_dets.size());
    
    for (const auto& match : matched_1) {
        auto* track = strack_pool[match.first];
        int col = match.second;
        track->update(high_dets[col].bbox);
        
        if (track->number_of_successful_consecutive_updates >= minimum_consecutive_frames && track->tracker_id == -1) {
            track->tracker_id = utils::BaseTracklet::get_next_tracker_id();
        }
        
        utils::Detection out_det = high_dets[col];
        out_det.tracker_id = track->tracker_id;
        out_detections.push_back(out_det);
    }

    // 6. Second Association (Remaining Tracked vs Low Conf)
    std::vector<utils::BaseTracklet*> remaining_tracked;
    for (int idx : unmatched_tracks_1) {
        if (strack_pool[idx]->time_since_update == 1) { // 1 means matched in prev frame
            remaining_tracked.push_back(strack_pool[idx]);
        }
    }

    cv::Mat iou_mat_low = _get_iou_matrix(remaining_tracked, low_dets);
    auto [matched_2, unmatched_tracks_2, unmatched_low] = _get_associated_indices(iou_mat_low, minimum_iou_threshold_second_assoc, remaining_tracked.size(), low_dets.size());
    
    for (const auto& match : matched_2) {
        auto* track = remaining_tracked[match.first];
        int col = match.second;
        track->update(low_dets[col].bbox);
        
        if (track->number_of_successful_consecutive_updates >= minimum_consecutive_frames && track->tracker_id == -1) {
            track->tracker_id = utils::BaseTracklet::get_next_tracker_id();
        }
        
        utils::Detection out_det = low_dets[col];
        out_det.tracker_id = track->tracker_id;
        out_detections.push_back(out_det);
    }

    // Pass along unmatched low dets (untracked)
    for (int col : unmatched_low) {
        utils::Detection out_det = low_dets[col];
        out_det.tracker_id = -1;
        out_detections.push_back(out_det);
    }

    // 7. Third Association (Unconfirmed vs Remaining High Conf)
    std::vector<utils::Detection> remaining_high_dets;
    std::vector<float> remaining_high_scores;
    for (int idx : unmatched_high) {
        remaining_high_dets.push_back(high_dets[idx]);
        remaining_high_scores.push_back(high_scores[idx]);
    }

    std::vector<int> final_unmatched_high = unmatched_high;
    std::set<utils::BaseTracklet*> tracks_to_remove;

    if (!unconfirmed_tracks.empty() && !remaining_high_dets.empty()) {
        cv::Mat iou_mat_uc = _get_iou_matrix(unconfirmed_tracks, remaining_high_dets);
        iou_mat_uc = fuse_score(iou->normalize_for_fusion(iou_mat_uc), remaining_high_scores);
        
        auto [matched_uc, unmatched_uc_indices, remaining_uh] = _get_associated_indices(iou_mat_uc, minimum_iou_threshold_unconfirmed_assoc, unconfirmed_tracks.size(), remaining_high_dets.size());

        for (const auto& match : matched_uc) {
            auto* track = unconfirmed_tracks[match.first];
            int orig_high_idx = unmatched_high[match.second];
            track->update(high_dets[orig_high_idx].bbox);
            
            if (track->number_of_successful_consecutive_updates >= minimum_consecutive_frames && track->tracker_id == -1) {
                track->tracker_id = utils::BaseTracklet::get_next_tracker_id();
            }
            
            utils::Detection out_det = high_dets[orig_high_idx];
            out_det.tracker_id = track->tracker_id;
            out_detections.push_back(out_det);
        }

        // Mark unmatched unconfirmed tracks for deletion
        for (int idx : unmatched_uc_indices) {
            tracks_to_remove.insert(unconfirmed_tracks[idx]);
        }

        final_unmatched_high.clear();
        for (int idx : remaining_uh) {
            final_unmatched_high.push_back(unmatched_high[idx]);
        }
    } else {
        for (auto* t : unconfirmed_tracks) tracks_to_remove.insert(t);
    }

    // 8. Remove unconfirmed tracks that didn't match
    tracks.erase(
        std::remove_if(tracks.begin(), tracks.end(), 
        [&](const std::unique_ptr<utils::BaseTracklet>& t) { return tracks_to_remove.count(t.get()) > 0; }),
        tracks.end()
    );

    // 9. Spawn new tracks
    _spawn_new_tracks(detections, final_unmatched_high, high_indices, out_detections, (frame_id == 1));

    // 10. Kill stale/lost tracks (cast down to pass to BoTSORT utility)
    // Note: To use `filter_alive_tracklets` seamlessly, we reinterpret cast the vector, or write the loop inline.
    // Writing inline here is safer since BaseTracker holds BaseTracklet.
    tracks.erase(
        std::remove_if(tracks.begin(), tracks.end(),
            [this](const std::unique_ptr<utils::BaseTracklet>& tracker) {
                bool is_mature = tracker->number_of_successful_consecutive_updates >= minimum_consecutive_frames;
                bool is_active = tracker->time_since_update == 0;
                return !((tracker->time_since_update < maximum_frames_without_update) && (is_mature || is_active));
            }),
        tracks.end()
    );

    return out_detections;
}

cv::Mat BoTSORTTracker::_get_iou_matrix(const std::vector<utils::BaseTracklet*>& tracklet_pool, 
                                        const std::vector<utils::Detection>& detections) {
    if (tracklet_pool.empty() || detections.empty()) {
        return cv::Mat::zeros(tracklet_pool.size(), detections.size(), CV_32F);
    }

    std::vector<cv::Vec4f> track_boxes;
    for (const auto* t : tracklet_pool) track_boxes.push_back(t->get_state_bbox());

    std::vector<cv::Vec4f> det_boxes;
    for (const auto& d : detections) det_boxes.push_back(d.bbox);

    return iou->compute(track_boxes, det_boxes);
}

std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>> 
BoTSORTTracker::_get_associated_indices(const cv::Mat& similarity_matrix, float min_similarity_thresh, int n_tracks, int n_dets) {
    std::vector<std::pair<int, int>> matched_indices;
    
    // We now use the explicitly passed n_tracks and n_dets!
    
    std::set<int> unmatched_tracks;
    for(int i=0; i<n_tracks; i++) unmatched_tracks.insert(i);
    std::set<int> unmatched_detections;
    for(int i=0; i<n_dets; i++) unmatched_detections.insert(i);

    if (n_tracks > 0 && n_dets > 0) {
        std::vector<int> row_indices, col_indices;
        
        linear_sum_assignment(similarity_matrix, row_indices, col_indices);

        for (size_t i = 0; i < row_indices.size(); ++i) {
            int row = row_indices[i];
            int col = col_indices[i];
            
            if (similarity_matrix.at<float>(row, col) >= min_similarity_thresh) {
                matched_indices.push_back({row, col});
                unmatched_tracks.erase(row);
                unmatched_detections.erase(col);
            }
        }
    }

    return {
        matched_indices, 
        std::vector<int>(unmatched_tracks.begin(), unmatched_tracks.end()), 
        std::vector<int>(unmatched_detections.begin(), unmatched_detections.end())
    };
}

void BoTSORTTracker::_spawn_new_tracks(
    const std::vector<utils::Detection>& detections,
    const std::vector<int>& unmatched_high_local,
    const std::vector<int>& high_indices,
    std::vector<utils::Detection>& out_detections,
    bool is_first_frame
) {
    for (int local_idx : unmatched_high_local) {
        int global_idx = high_indices[local_idx];
        float conf = detections[global_idx].confidence;
        
        if (conf >= track_activation_threshold) {
            auto tracklet = std::make_unique<BoTSORTTracklet>(detections[global_idx].bbox);
            
            if (is_first_frame && instant_first_frame_activation) {
                tracklet->tracker_id = utils::BaseTracklet::get_next_tracker_id();
            }
            
            utils::Detection out_det = detections[global_idx];
            out_det.tracker_id = tracklet->tracker_id;
            
            tracks.push_back(std::move(tracklet));
            out_detections.push_back(out_det);
        }
    }
}

void BoTSORTTracker::linear_sum_assignment(const cv::Mat& similarity_matrix, std::vector<int>& row_indices, std::vector<int>& col_indices) {
    if (similarity_matrix.empty()) return;

    int n_rows = similarity_matrix.rows;
    int n_cols = similarity_matrix.cols;

    // 1. Convert cv::Mat (float32) to vector<vector<double>>
    // AND convert Similarity (IoU) to Cost (Distance) because Hungarian minimizes.
    std::vector<std::vector<double>> cost_matrix(n_rows, std::vector<double>(n_cols, 0.0));
    
    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            // High similarity (1.0) becomes low cost (0.0)
            cost_matrix[i][j] = 1.0 - static_cast<double>(similarity_matrix.at<float>(i, j));
        }
    }

    // 2. Execute the Hungarian Algorithm
    HungarianAlgorithm hungarian;
    std::vector<int> assignment;
    
    // Solve modifies the assignment vector where index is row, and value is assigned column
    hungarian.Solve(cost_matrix, assignment);

    // 3. Extract the matched pairs
    for (int x = 0; x < n_rows; x++) {
        // Hungarian returns -1 in the assignment vector if a row could not be assigned
        if (assignment[x] >= 0) {
            row_indices.push_back(x);
            col_indices.push_back(assignment[x]);
        }
    }
}

} // namespace botsort
} // namespace core
} // namespace trackers