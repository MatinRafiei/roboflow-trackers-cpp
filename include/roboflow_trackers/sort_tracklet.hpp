#pragma once

#include "base_tracklet.hpp"
#include <opencv2/opencv.hpp>

namespace trackers {
namespace core {
namespace sort {

class SORTTracklet : public utils::BaseTracklet {
private:
    void _configure_noise();

public:
    // SORT tracks total updates cumulatively, not consecutively
    int number_of_successful_updates;

    // SORT defaults to XYXY state representation
    SORTTracklet(const cv::Vec4f& initial_bbox, 
                 utils::StateRepresentation state_repr = utils::StateRepresentation::XYXY);

    void update(const cv::Vec4f& bbox) override;
    cv::Vec4f predict() override;

    void mark_lost() override {}
    void mark_removed() override {}
};

} // namespace sort
} // namespace core
} // namespace trackers