#pragma once

#include "base_tracklet.hpp"
#include "cmc.hpp"
#include <opencv2/opencv.hpp>

namespace trackers {
namespace core {
namespace botsort {

class BoTSORTTracklet : public utils::BaseTracklet {
private:
    // Scale-aware noise constants
    static constexpr float _SIGMA_P = 0.05f;
    static constexpr float _SIGMA_V = 0.00625f;
    static constexpr float _SIGMA_M = 0.05f;

    void _configure_initial_noise(const cv::Vec4f& bbox);
    void _set_scale_aware_noise(float w, float h, bool initial = false);
    void _refresh_noise_from_state();
    void _clamp_state_bbox();

public:
    // Constructor defaults to XCYCWH just like the Python implementation
    BoTSORTTracklet(const cv::Vec4f& initial_bbox, 
                    utils::StateRepresentation state_repr = utils::StateRepresentation::XCYCWH);

    void update(const cv::Vec4f& bbox) override;
    cv::Vec4f predict() override;

    // BaseTracklet pure virtuals (Can be expanded if tracker-state logic is needed here)
    void mark_lost() override {}
    void mark_removed() override {}

    // Apply CMC transform to this specific tracklet
    void apply_cmc(const cv::Mat& H);
};

} // namespace botsort
} // namespace core
} // namespace trackers