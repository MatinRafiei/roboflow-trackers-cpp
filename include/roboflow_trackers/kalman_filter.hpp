#pragma once
#include <opencv2/opencv.hpp>

namespace trackers {
namespace utils {

class KalmanFilter {
public:
    int dim_x;
    int dim_z;

    // Core matrices
    cv::Mat x; // State vector (dim_x, 1)
    cv::Mat P; // State covariance matrix (dim_x, dim_x)
    cv::Mat F; // State transition matrix (dim_x, dim_x)
    cv::Mat H; // Measurement function matrix (dim_z, dim_x)
    cv::Mat Q; // Process noise covariance (dim_x, dim_x)
    cv::Mat R; // Measurement noise covariance (dim_z, dim_z)

    // Tracked priors and posteriors
    cv::Mat x_prior;
    cv::Mat P_prior;
    cv::Mat x_post;
    cv::Mat P_post;

    // Internal calculation matrices (kept as members to avoid reallocation)
    cv::Mat y;
    cv::Mat S;
    cv::Mat K;

private:
    cv::Mat _I; // Identity matrix cache

public:
    KalmanFilter(int dim_x, int dim_z);

    void predict();
    void update(const cv::Mat& z);
};

} // namespace utils
} // namespace trackers