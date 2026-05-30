#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <vector>
#include <string>
#include <memory>
#include "base_tracklet.hpp"
#include "state_representations.hpp"

namespace trackers {
namespace utils {

enum class CMCMethod {
    ORB,
    SIFT,
    SPARSE_OPTFLOW,
    ECC
};

struct CMCConfig {
    CMCMethod method = CMCMethod::SPARSE_OPTFLOW;
    int downscale = 2;

    // Shared Feature params
    float ransac_reproj_threshold = 3.0f;
    float max_spatial_distance_frac = 0.25f;
    float roi_min_frac = 0.02f;
    float roi_max_frac = 0.98f;

    // ORB
    int fast_threshold = 20;

    // SIFT
    int sift_n_octave_layers = 3;
    float sift_contrast_threshold = 0.02f;
    int sift_edge_threshold = 20;

    // Sparse Optical Flow
    int sof_max_corners = 1000;
    double sof_quality_level = 0.01;
    double sof_min_distance = 1.0;
    int sof_block_size = 3;
    bool sof_use_harris = false;
    double sof_k = 0.04;

    // ECC
    int ecc_number_of_iterations = 50;
    double ecc_termination_eps = 1e-4;
    int ecc_gaussian_filter_size = 1;
};

class CMC {
private:
    CMCConfig cfg;
    bool _initialized;
    int frames_failed;

    // Buffers for avoiding reallocation
    cv::Mat _prev_frame_gray;
    cv::Mat _gray_downscaled;

    // ORB/SIFT state
    cv::Ptr<cv::Feature2D> detector;
    cv::Ptr<cv::Feature2D> extractor;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    std::vector<cv::KeyPoint> _prev_kps;
    cv::Mat _prev_desc;

    // SparseOptFlow state
    std::vector<cv::Point2f> _prev_points;

    // Internal estimators
    cv::Mat _estimate_sparse_optflow(const cv::Mat& frame_bgr);
    cv::Mat _estimate_ecc(const cv::Mat& frame_bgr);
    cv::Mat _estimate_feature_affine(const cv::Mat& frame_bgr, const std::vector<cv::Vec4f>& dets_xyxy);

public:
    CMC(const CMCConfig& config = CMCConfig());

    void reset();
    
    // Estimate affine transform mapping previous frame to current
    cv::Mat estimate(const cv::Mat& frame_bgr, const std::vector<cv::Vec4f>& dets_xyxy = {});

    // Utility: Apply affine warp to 4 corners of an XYXY box to maintain axis-alignment
    static cv::Vec4f warp_xyxy_corners(const cv::Vec4f& xyxy, const cv::Mat& R, const cv::Mat& t);

    // Apply computed affine transform to the Kalman Filter states of all active tracks
    static void apply_batch(const cv::Mat& H, std::vector<std::unique_ptr<BaseTracklet>>& tracklets);
};

} // namespace utils
} // namespace trackers