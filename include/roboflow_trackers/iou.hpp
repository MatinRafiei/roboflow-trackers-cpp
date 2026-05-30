#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include <cmath>

namespace trackers {
namespace utils {

class BaseIoU {
public:
    virtual ~BaseIoU() = default;

    // Main entry point - handles edge cases like empty inputs
    cv::Mat compute(const std::vector<cv::Vec4f>& boxes_1, const std::vector<cv::Vec4f>& boxes_2);

    // Normalize for fusion shifts [-1, 1] metrics into [0, 1]
    virtual cv::Mat normalize_for_fusion(const cv::Mat& similarity_matrix);

protected:
    // Pure virtual hook that child classes must implement
    virtual cv::Mat _compute(const std::vector<cv::Vec4f>& boxes_1, const std::vector<cv::Vec4f>& boxes_2) = 0;
};

// --- IoU Variants ---

class IoU : public BaseIoU {
protected:
    cv::Mat _compute(const std::vector<cv::Vec4f>& boxes_1, const std::vector<cv::Vec4f>& boxes_2) override;
};

class BIoU : public BaseIoU {
private:
    float buffer_ratio;
public:
    BIoU(float buffer_ratio = 0.1f);
protected:
    cv::Mat _compute(const std::vector<cv::Vec4f>& boxes_1, const std::vector<cv::Vec4f>& boxes_2) override;
};

class GIoU : public BaseIoU {
public:
    cv::Mat normalize_for_fusion(const cv::Mat& similarity_matrix) override;
protected:
    cv::Mat _compute(const std::vector<cv::Vec4f>& boxes_1, const std::vector<cv::Vec4f>& boxes_2) override;
};

class DIoU : public BaseIoU {
private:
    static constexpr float _EPS = 1e-7f;
public:
    cv::Mat normalize_for_fusion(const cv::Mat& similarity_matrix) override;
protected:
    cv::Mat _compute(const std::vector<cv::Vec4f>& boxes_1, const std::vector<cv::Vec4f>& boxes_2) override;
};

class CIoU : public BaseIoU {
private:
    static constexpr float _EPS = 1e-7f;
public:
    cv::Mat normalize_for_fusion(const cv::Mat& similarity_matrix) override;
protected:
    cv::Mat _compute(const std::vector<cv::Vec4f>& boxes_1, const std::vector<cv::Vec4f>& boxes_2) override;
};

} // namespace utils
} // namespace trackers