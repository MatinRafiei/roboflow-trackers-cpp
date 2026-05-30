#include "iou.hpp"
#include <algorithm>

namespace trackers {
namespace utils {

// --- BaseIoU Implementation ---
cv::Mat BaseIoU::compute(const std::vector<cv::Vec4f>& boxes_1, const std::vector<cv::Vec4f>& boxes_2) {
    if (boxes_1.empty() || boxes_2.empty()) {
        return cv::Mat::zeros(boxes_1.size(), boxes_2.size(), CV_32F);
    }
    return _compute(boxes_1, boxes_2);
}

cv::Mat BaseIoU::normalize_for_fusion(const cv::Mat& similarity_matrix) {
    // Default implementation returns the matrix unchanged
    return similarity_matrix.clone();
}

// Helper function to calculate raw intersection and union for a single pair of boxes
inline void calculate_inter_union(const cv::Vec4f& b1, const cv::Vec4f& b2, float& iou, float& union_area, 
                                  float& enc_area, float& enc_diag_sq) {
    // Areas
    float area1 = (b1[2] - b1[0]) * (b1[3] - b1[1]);
    float area2 = (b2[2] - b2[0]) * (b2[3] - b2[1]);

    // Intersection
    float inter_x1 = std::max(b1[0], b2[0]);
    float inter_y1 = std::max(b1[1], b2[1]);
    float inter_x2 = std::min(b1[2], b2[2]);
    float inter_y2 = std::min(b1[3], b2[3]);
    
    float inter_w = std::max(0.0f, inter_x2 - inter_x1);
    float inter_h = std::max(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;

    union_area = area1 + area2 - inter_area;
    iou = (union_area > 0) ? (inter_area / union_area) : 0.0f;

    // Smallest enclosing box
    float enc_x1 = std::min(b1[0], b2[0]);
    float enc_y1 = std::min(b1[1], b2[1]);
    float enc_x2 = std::max(b1[2], b2[2]);
    float enc_y2 = std::max(b1[3], b2[3]);
    
    float enc_w = enc_x2 - enc_x1;
    float enc_h = enc_y2 - enc_y1;
    
    enc_area = enc_w * enc_h;
    enc_diag_sq = enc_w * enc_w + enc_h * enc_h;
}

// --- Standard IoU ---
cv::Mat IoU::_compute(const std::vector<cv::Vec4f>& boxes_1, const std::vector<cv::Vec4f>& boxes_2) {
    int N = boxes_1.size();
    int M = boxes_2.size();
    cv::Mat result(N, M, CV_32F);

    for (int i = 0; i < N; ++i) {
        float area1 = (boxes_1[i][2] - boxes_1[i][0]) * (boxes_1[i][3] - boxes_1[i][1]);
        for (int j = 0; j < M; ++j) {
            float inter_x1 = std::max(boxes_1[i][0], boxes_2[j][0]);
            float inter_y1 = std::max(boxes_1[i][1], boxes_2[j][1]);
            float inter_x2 = std::min(boxes_1[i][2], boxes_2[j][2]);
            float inter_y2 = std::min(boxes_1[i][3], boxes_2[j][3]);
            
            float inter_w = std::max(0.0f, inter_x2 - inter_x1);
            float inter_h = std::max(0.0f, inter_y2 - inter_y1);
            float inter_area = inter_w * inter_h;

            float area2 = (boxes_2[j][2] - boxes_2[j][0]) * (boxes_2[j][3] - boxes_2[j][1]);
            float union_area = area1 + area2 - inter_area;
            
            result.at<float>(i, j) = (union_area > 0) ? (inter_area / union_area) : 0.0f;
        }
    }
    return result;
}

// --- BIoU (Buffered IoU) ---
BIoU::BIoU(float buffer_ratio) : buffer_ratio(buffer_ratio) {
    if (buffer_ratio < 0) throw std::invalid_argument("buffer_ratio must be non-negative");
}

cv::Mat BIoU::_compute(const std::vector<cv::Vec4f>& boxes_1, const std::vector<cv::Vec4f>& boxes_2) {
    if (buffer_ratio == 0.0f) {
        IoU standard_iou;
        return standard_iou.compute(boxes_1, boxes_2); // reuse standard computation
    }

    auto apply_buffer = [this](const std::vector<cv::Vec4f>& boxes) {
        std::vector<cv::Vec4f> buffered(boxes.size());
        for (size_t i = 0; i < boxes.size(); ++i) {
            float w = boxes[i][2] - boxes[i][0];
            float h = boxes[i][3] - boxes[i][1];
            buffered[i] = cv::Vec4f(
                boxes[i][0] - buffer_ratio * w,
                boxes[i][1] - buffer_ratio * h,
                boxes[i][2] + buffer_ratio * w,
                boxes[i][3] + buffer_ratio * h
            );
        }
        return buffered;
    };

    IoU standard_iou;
    return standard_iou.compute(apply_buffer(boxes_1), apply_buffer(boxes_2));
}

// --- GIoU Implementation ---
cv::Mat GIoU::normalize_for_fusion(const cv::Mat& similarity_matrix) {
    return (similarity_matrix + 1.0f) / 2.0f;
}

cv::Mat GIoU::_compute(const std::vector<cv::Vec4f>& boxes_1, const std::vector<cv::Vec4f>& boxes_2) {
    int N = boxes_1.size();
    int M = boxes_2.size();
    cv::Mat result(N, M, CV_32F);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            float iou, union_area, enc_area, enc_diag_sq;
            calculate_inter_union(boxes_1[i], boxes_2[j], iou, union_area, enc_area, enc_diag_sq);
            
            float penalty = (enc_area > 0) ? ((enc_area - union_area) / enc_area) : 0.0f;
            result.at<float>(i, j) = iou - penalty;
        }
    }
    return result;
}

// --- DIoU Implementation ---
cv::Mat DIoU::normalize_for_fusion(const cv::Mat& similarity_matrix) {
    return (similarity_matrix + 1.0f) / 2.0f;
}

cv::Mat DIoU::_compute(const std::vector<cv::Vec4f>& boxes_1, const std::vector<cv::Vec4f>& boxes_2) {
    int N = boxes_1.size();
    int M = boxes_2.size();
    cv::Mat result(N, M, CV_32F);

    for (int i = 0; i < N; ++i) {
        float cx1 = (boxes_1[i][0] + boxes_1[i][2]) / 2.0f;
        float cy1 = (boxes_1[i][1] + boxes_1[i][3]) / 2.0f;
        
        for (int j = 0; j < M; ++j) {
            float iou, union_area, enc_area, enc_diag_sq;
            calculate_inter_union(boxes_1[i], boxes_2[j], iou, union_area, enc_area, enc_diag_sq);

            float cx2 = (boxes_2[j][0] + boxes_2[j][2]) / 2.0f;
            float cy2 = (boxes_2[j][1] + boxes_2[j][3]) / 2.0f;
            
            float dx = cx1 - cx2;
            float dy = cy1 - cy2;
            float center_dist_sq = dx * dx + dy * dy;

            result.at<float>(i, j) = iou - (center_dist_sq / (enc_diag_sq + _EPS));
        }
    }
    return result;
}

// --- CIoU Implementation ---
cv::Mat CIoU::normalize_for_fusion(const cv::Mat& similarity_matrix) {
    return (similarity_matrix + 1.0f) / 2.0f;
}

cv::Mat CIoU::_compute(const std::vector<cv::Vec4f>& boxes_1, const std::vector<cv::Vec4f>& boxes_2) {
    int N = boxes_1.size();
    int M = boxes_2.size();
    cv::Mat result(N, M, CV_32F);
    
    const float pi_sq = static_cast<float>(M_PI * M_PI);
    const float four_over_pi_sq = 4.0f / pi_sq;

    for (int i = 0; i < N; ++i) {
        float cx1 = (boxes_1[i][0] + boxes_1[i][2]) / 2.0f;
        float cy1 = (boxes_1[i][1] + boxes_1[i][3]) / 2.0f;
        float w1 = boxes_1[i][2] - boxes_1[i][0];
        float h1 = std::max(boxes_1[i][3] - boxes_1[i][1], _EPS);
        float arctan1 = std::atan(w1 / h1);

        for (int j = 0; j < M; ++j) {
            float iou, union_area, enc_area, enc_diag_sq;
            calculate_inter_union(boxes_1[i], boxes_2[j], iou, union_area, enc_area, enc_diag_sq);

            float cx2 = (boxes_2[j][0] + boxes_2[j][2]) / 2.0f;
            float cy2 = (boxes_2[j][1] + boxes_2[j][3]) / 2.0f;
            float dx = cx1 - cx2;
            float dy = cy1 - cy2;
            float center_dist_sq = dx * dx + dy * dy;
            
            float diou = iou - (center_dist_sq / (enc_diag_sq + _EPS));

            float w2 = boxes_2[j][2] - boxes_2[j][0];
            float h2 = std::max(boxes_2[j][3] - boxes_2[j][1], _EPS);
            float arctan2 = std::atan(w2 / h2);

            float v = four_over_pi_sq * std::pow(arctan1 - arctan2, 2);
            float alpha = 0.0f;
            
            if ((1.0f - iou + v) > 0) {
                alpha = v / (1.0f - iou + v);
            }

            result.at<float>(i, j) = diou - alpha * v;
        }
    }
    return result;
}

} // namespace utils
} // namespace trackers