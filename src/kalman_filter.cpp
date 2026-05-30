#include "kalman_filter.hpp"

namespace trackers {
namespace utils {

KalmanFilter::KalmanFilter(int dim_x, int dim_z) : dim_x(dim_x), dim_z(dim_z) {
    // Initialize all matrices with 32-bit floats (CV_32F) for edge performance.
    // Python usually defaults to float64, but float32 is significantly faster on ARM/Edge CPUs.
    x = cv::Mat::zeros(dim_x, 1, CV_32F);
    P = cv::Mat::eye(dim_x, dim_x, CV_32F);
    Q = cv::Mat::eye(dim_x, dim_x, CV_32F);
    F = cv::Mat::eye(dim_x, dim_x, CV_32F);
    H = cv::Mat::zeros(dim_z, dim_x, CV_32F);
    R = cv::Mat::eye(dim_z, dim_z, CV_32F);
    _I = cv::Mat::eye(dim_x, dim_x, CV_32F);
}

void KalmanFilter::predict() {
    // x = F @ x
    x = F * x;
    
    // P = F @ P @ F.T + Q
    P = F * P * F.t() + Q;

    // Save prior
    x_prior = x.clone();
    P_prior = P.clone();
}

void KalmanFilter::update(const cv::Mat& z) {
    // Ensure the incoming measurement is a column vector (dim_z, 1)
    cv::Mat z_col = z;
    if (z_col.cols != 1) {
        z_col = z.reshape(1, dim_z);
    }

    // Residual: y = z - H @ x
    y = z_col - H * x;

    // System uncertainty: S = H @ P @ H.T + R
    cv::Mat PHT = P * H.t();
    S = H * PHT + R;

    // Kalman gain: K = P @ H.T @ S^-1
    // OpenCV's .inv() is highly optimized for small matrices
    K = PHT * S.inv();

    // State update: x = x + K @ y
    x = x + K * y;

    // Covariance update (Joseph form for numerical stability):
    // P = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T
    cv::Mat I_KH = _I - K * H;
    P = I_KH * P * I_KH.t() + K * R * K.t();

    // Save posterior
    x_post = x.clone();
    P_post = P.clone();
}

} // namespace utils
} // namespace trackers