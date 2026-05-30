#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

namespace trackers {
namespace utils {

    // 1. xyxy (x_min, y_min, x_max, y_max) to xywh (x_center, y_center, width, height)
    cv::Vec4f xyxy_to_xywh(const cv::Vec4f& xyxy);
    std::vector<cv::Vec4f> xyxy_to_xywh(const std::vector<cv::Vec4f>& xyxy_batch);

    // 2. xywh to xyxy
    cv::Vec4f xywh_to_xyxy(const cv::Vec4f& xywh);
    std::vector<cv::Vec4f> xywh_to_xyxy(const std::vector<cv::Vec4f>& xywh_batch);

    // 3. xyxy to xcycsr (x_center, y_center, scale/area, aspect_ratio)
    cv::Vec4f xyxy_to_xcycsr(const cv::Vec4f& xyxy);
    std::vector<cv::Vec4f> xyxy_to_xcycsr(const std::vector<cv::Vec4f>& xyxy_batch);

    // 4. xcycsr to xyxy
    cv::Vec4f xcycsr_to_xyxy(const cv::Vec4f& xcycsr);
    std::vector<cv::Vec4f> xcycsr_to_xyxy(const std::vector<cv::Vec4f>& xcycsr_batch);

} // namespace utils
} // namespace trackers

