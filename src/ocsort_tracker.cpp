#include "ocsort_tracker.hpp"
#include "ocsort_utils.hpp"
#include "Hungarian.h" 
#include <set>

namespace trackers {
namespace core {
namespace ocsort {

OCSORTTracker::OCSORTTracker(
    int lost_track_buffer, float frame_rate, float track_activation_threshold,
    float high_conf_det_threshold, float minimum_iou_threshold,
    int minimum_consecutive_frames, float direction_consistency_weight, int velocity_time_window
) : 
    track_activation_threshold(track_activation_threshold),
    high_conf_det_threshold(high_conf_det_threshold),
    minimum_iou_threshold(minimum_iou_threshold),
    minimum_consecutive_frames(minimum_consecutive_frames),
    direction_consistency_weight(direction_consistency_weight),
    velocity_time_window(velocity_time_window)
{
    this->maximum_frames_without_update = static_cast<int>((frame_rate / 30.0f) * lost_track_buffer);
    iou = std::make_unique<utils::IoU>(); 
}

void OCSORTTracker::reset() {
    tracks.clear();
    utils::BaseTracklet::count_id = 0;
}

utils::Detections OCSORTTracker::update(const utils::Detections& detections, const cv::Mat& /*frame*/) {
    if (tracks.empty() && detections.empty()) return utils::Detections();

    utils::Detections out_detections;
    out_detections.reserve(detections.size());

    // 1. Predict and Freeze
    for (auto& track : tracks) {
        auto* ocsort_trk = static_cast<OCSORTTracklet*>(track.get());
        ocsort_trk->freeze(); // Save state before predict
        ocsort_trk->predict();
    }

    // 2. High/Low Conf Split
    std::vector<int> high_indices;
    std::vector<utils::Detection> high_dets;
    for (size_t i = 0; i < detections.size(); ++i) {
        if (detections[i].confidence >= high_conf_det_threshold) {
            high_indices.push_back(i);
            high_dets.push_back(detections[i]);
        }
    }

    // 3. First Association (IoU + Direction Consistency)
    std::vector<OCSORTTracklet*> active_tracks;
    for (auto& track : tracks) {
        active_tracks.push_back(static_cast<OCSORTTracklet*>(track.get()));
    }

    std::set<int> matched_tracks, matched_dets;

    // GUARD: Only perform matrix math if we have BOTH tracks and detections
    if (!active_tracks.empty() && !high_dets.empty()) {
        std::vector<cv::Vec4f> track_boxes, det_boxes;
        for (const auto* t : active_tracks) track_boxes.push_back(t->get_state_bbox());
        for (const auto& d : high_dets) det_boxes.push_back(d.bbox);

        cv::Mat iou_mat = iou->compute(track_boxes, det_boxes);
        
        // Add Observation-Centric Momentum (OCM)
        std::vector<cv::Vec2f> vels;
        std::vector<cv::Vec4f> refs;
        std::vector<float> masks;
        for (const auto* t : active_tracks) {
            vels.push_back(t->velocity);
            refs.push_back(t->get_k_previous_obs());
            masks.push_back(t->has_velocity ? 1.0f : 0.0f);
        }
        
        cv::Mat ocm_mat = build_direction_consistency_matrix_batch(vels, refs, det_boxes, masks);
        cv::Mat cost_mat = iou_mat + direction_consistency_weight * ocm_mat;

        // Hungarian Match
        std::vector<int> rows, cols;
        linear_sum_assignment(cost_mat, rows, cols);

        for (size_t i = 0; i < rows.size(); i++) {
            if (cost_mat.at<float>(rows[i], cols[i]) >= minimum_iou_threshold) { 
                active_tracks[rows[i]]->update(high_dets[cols[i]].bbox);
                if (active_tracks[rows[i]]->tracker_id == -1) {
                    active_tracks[rows[i]]->tracker_id = utils::BaseTracklet::get_next_tracker_id();
                }
                utils::Detection out_det = high_dets[cols[i]];
                out_det.tracker_id = active_tracks[rows[i]]->tracker_id;
                out_detections.push_back(out_det);
                
                matched_tracks.insert(rows[i]);
                matched_dets.insert(cols[i]);
            }
        }
    }

    // 4. Second Association (Low Conf - Omitted for brevity, but similar to ByteTrack)
    
    // 5. Spawn new tracks
    for (size_t i = 0; i < high_dets.size(); i++) {
        if (matched_dets.find(i) == matched_dets.end() && high_dets[i].confidence >= track_activation_threshold) {
            auto tracklet = std::make_unique<OCSORTTracklet>(high_dets[i].bbox, velocity_time_window);
            tracklet->tracker_id = utils::BaseTracklet::get_next_tracker_id();
            utils::Detection out_det = high_dets[i];
            out_det.tracker_id = tracklet->tracker_id;
            tracks.push_back(std::move(tracklet));
            out_detections.push_back(out_det);
        }
    }

    // 6. Prune stale tracks
    tracks.erase(std::remove_if(tracks.begin(), tracks.end(),
        [this](const std::unique_ptr<utils::BaseTracklet>& t) {
            return t->time_since_update >= maximum_frames_without_update;
        }),
        tracks.end());

    return out_detections;
}

// Ensure you include your Hungarian Algorithm wrapper logic here just like in BoTSORT
void OCSORTTracker::linear_sum_assignment(const cv::Mat& sim_mat, std::vector<int>& rows, std::vector<int>& cols) {
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

} // namespace ocsort
} // namespace core
} // namespace trackers