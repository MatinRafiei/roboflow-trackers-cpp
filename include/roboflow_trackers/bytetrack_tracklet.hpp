#pragma once

#include "base_tracklet.hpp"
#include <opencv2/opencv.hpp>

namespace trackers {
namespace core {
namespace bytetrack {

class ByteTrackTracklet : public utils::BaseTracklet {
private:
    void _configure_noise();

public:
    // ByteTrack exclusively uses XYXY state representation by default
    ByteTrackTracklet(const cv::Vec4f& initial_bbox, 
                      utils::StateRepresentation state_repr = utils::StateRepresentation::XYXY);

    void update(const cv::Vec4f& bbox) override;
    cv::Vec4f predict() override;

    void mark_lost() override {}
    void mark_removed() override {}
};

} // namespace bytetrack
} // namespace core
} // namespace trackers