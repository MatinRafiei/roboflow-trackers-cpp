#pragma once

#include "base_tracklet.hpp"
#include <opencv2/opencv.hpp>
#include <map>
#include <vector>

namespace trackers {
namespace core {
namespace ocsort {

class OCSORTTracklet : public utils::BaseTracklet {
private:
    void _configure_noise();

public:
    cv::Vec4f last_observation;
    cv::Vec4f previous_to_last_observation;
    cv::Vec2f velocity;
    bool has_velocity;
    std::map<int, cv::Vec4f> observations;
    int delta_t;
    int velocity_time_window;

    // Frozen state components for ORU
    cv::Mat frozen_x;
    cv::Mat frozen_P;
    cv::Mat frozen_F;

    OCSORTTracklet(const cv::Vec4f& initial_bbox, int velocity_time_window = 3, 
                   utils::StateRepresentation state_repr = utils::StateRepresentation::XCYCSR);

    void update(const cv::Vec4f& bbox) override;
    cv::Vec4f predict() override;
    
    void freeze();
    void unfreeze();
    void clear_history();
    void update_velocity();
    cv::Vec4f get_k_previous_obs() const;
    void generate_virtual_trajectory(const cv::Vec4f& current_bbox);

    void mark_lost() override {}
    void mark_removed() override {}
};

} // namespace ocsort
} // namespace core
} // namespace trackers