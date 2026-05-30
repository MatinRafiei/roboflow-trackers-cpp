#include "converters.hpp"
#include <cmath>

namespace trackers {
namespace utils {

// --- 1. xyxy to xywh ---
cv::Vec4f xyxy_to_xywh(const cv::Vec4f& xyxy) {
    float w = xyxy[2] - xyxy[0];
    float h = xyxy[3] - xyxy[1];
    return cv::Vec4f(xyxy[0] + w * 0.5f, xyxy[1] + h * 0.5f, w, h);
}

std::vector<cv::Vec4f> xyxy_to_xywh(const std::vector<cv::Vec4f>& xyxy_batch) {
    std::vector<cv::Vec4f> result;
    result.reserve(xyxy_batch.size()); // Pre-allocate for edge performance
    for (const auto& box : xyxy_batch) {
        result.push_back(xyxy_to_xywh(box));
    }
    return result;
}

// --- 2. xywh to xyxy ---
cv::Vec4f xywh_to_xyxy(const cv::Vec4f& xywh) {
    float hw = xywh[2] * 0.5f;
    float hh = xywh[3] * 0.5f;
    return cv::Vec4f(xywh[0] - hw, xywh[1] - hh, xywh[0] + hw, xywh[1] + hh);
}

std::vector<cv::Vec4f> xywh_to_xyxy(const std::vector<cv::Vec4f>& xywh_batch) {
    std::vector<cv::Vec4f> result;
    result.reserve(xywh_batch.size());
    for (const auto& box : xywh_batch) {
        result.push_back(xywh_to_xyxy(box));
    }
    return result;
}

// --- 3. xyxy to xcycsr ---
cv::Vec4f xyxy_to_xcycsr(const cv::Vec4f& xyxy) {
    float w = xyxy[2] - xyxy[0];
    float h = xyxy[3] - xyxy[1];
    // Adding 1e-6 to avoid division by zero, exactly like the Python code
    return cv::Vec4f(xyxy[0] + w * 0.5f, xyxy[1] + h * 0.5f, w * h, w / (h + 1e-6f));
}

std::vector<cv::Vec4f> xyxy_to_xcycsr(const std::vector<cv::Vec4f>& xyxy_batch) {
    std::vector<cv::Vec4f> result;
    result.reserve(xyxy_batch.size());
    for (const auto& box : xyxy_batch) {
        result.push_back(xyxy_to_xcycsr(box));
    }
    return result;
}

// --- 4. xcycsr to xyxy ---
cv::Vec4f xcycsr_to_xyxy(const cv::Vec4f& xcycsr) {
    float w = std::sqrt(xcycsr[2] * xcycsr[3]);
    float h = xcycsr[2] / w;
    float hw = w * 0.5f;
    float hh = h * 0.5f;
    return cv::Vec4f(xcycsr[0] - hw, xcycsr[1] - hh, xcycsr[0] + hw, xcycsr[1] + hh);
}

std::vector<cv::Vec4f> xcycsr_to_xyxy(const std::vector<cv::Vec4f>& xcycsr_batch) {
    std::vector<cv::Vec4f> result;
    result.reserve(xcycsr_batch.size());
    for (const auto& box : xcycsr_batch) {
        result.push_back(xcycsr_to_xyxy(box));
    }
    return result;
}

} // namespace utils
} // namespace trackers