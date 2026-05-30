#include "state_representations.hpp"
#include <stdexcept>

namespace trackers {
namespace utils {

// --- BaseStateEstimator Implementation ---
void BaseStateEstimator::predict() {
    kf.predict();
    clamp_velocity();
}

void BaseStateEstimator::update(const cv::Vec4f& bbox) {
    cv::Mat z = bbox_to_measurement(bbox);
    kf.update(z);
    clamp_velocity();
}

// --- XCYCSR Implementation ---
XCYCSRStateEstimator::XCYCSRStateEstimator(const cv::Vec4f& bbox) : BaseStateEstimator(7, 4) {
    // F: State transition matrix
    kf.F = cv::Mat::eye(7, 7, CV_32F);
    kf.F.at<float>(0, 4) = 1.0f;
    kf.F.at<float>(1, 5) = 1.0f;
    kf.F.at<float>(2, 6) = 1.0f;

    // H: Measurement matrix (4x7)
    kf.H = cv::Mat::eye(4, 7, CV_32F);

    // P: Covariance matrix (Python: P[4:, 4:] *= 1000; P *= 10.0)
    kf.P = cv::Mat::eye(7, 7, CV_32F) * 10.0f;
    kf.P(cv::Rect(4, 4, 3, 3)) *= 1000.0f; 

    // R: Measurement noise
    kf.R = cv::Mat::eye(4, 4, CV_32F) * 10.0f;

    // Q: Process noise
    kf.Q = cv::Mat::eye(7, 7, CV_32F) * 0.01f;
    kf.Q.at<float>(6, 6) *= 0.01f; // Python: kf.Q[-1, -1] *= 0.01
    kf.Q(cv::Rect(4, 4, 3, 3)) *= 0.01f; // Python: kf.Q[4:, 4:] *= 0.01

    // Initialize state
    cv::Mat z = bbox_to_measurement(bbox);
    z.copyTo(kf.x(cv::Rect(0, 0, 1, 4))); 
}

cv::Mat XCYCSRStateEstimator::bbox_to_measurement(const cv::Vec4f& bbox) {
    cv::Vec4f xcycsr = xyxy_to_xcycsr(bbox);
    cv::Mat z(4, 1, CV_32F);
    z.at<float>(0, 0) = xcycsr[0];
    z.at<float>(1, 0) = xcycsr[1];
    z.at<float>(2, 0) = xcycsr[2];
    z.at<float>(3, 0) = xcycsr[3];
    return z;
}

cv::Vec4f XCYCSRStateEstimator::state_to_bbox() {
    cv::Vec4f xcycsr(
        kf.x.at<float>(0, 0),
        kf.x.at<float>(1, 0),
        kf.x.at<float>(2, 0),
        kf.x.at<float>(3, 0)
    );
    return xcycsr_to_xyxy(xcycsr);
}

void XCYCSRStateEstimator::clamp_velocity() {
    // In XCYCSR, scale ratio change velocity is set to 0 to prevent drastic shape shifts
    kf.x.at<float>(6, 0) = 0.0f;
}

// --- XCYCWH Implementation ---
XCYCWHStateEstimator::XCYCWHStateEstimator(const cv::Vec4f& bbox) : BaseStateEstimator(8, 4) {
    kf.F = cv::Mat::eye(8, 8, CV_32F);
    kf.F.at<float>(0, 4) = 1.0f;
    kf.F.at<float>(1, 5) = 1.0f;
    kf.F.at<float>(2, 6) = 1.0f;
    kf.F.at<float>(3, 7) = 1.0f;

    kf.H = cv::Mat::eye(4, 8, CV_32F);

    kf.P = cv::Mat::eye(8, 8, CV_32F) * 10.0f;
    kf.P(cv::Rect(4, 4, 4, 4)) *= 1000.0f;

    kf.R = cv::Mat::eye(4, 4, CV_32F) * 10.0f;
    kf.Q = cv::Mat::eye(8, 8, CV_32F) * 0.01f;

    cv::Mat z = bbox_to_measurement(bbox);
    z.copyTo(kf.x(cv::Rect(0, 0, 1, 4)));
}

cv::Mat XCYCWHStateEstimator::bbox_to_measurement(const cv::Vec4f& bbox) {
    cv::Vec4f xywh = xyxy_to_xywh(bbox);
    cv::Mat z(4, 1, CV_32F);
    z.at<float>(0, 0) = xywh[0];
    z.at<float>(1, 0) = xywh[1];
    z.at<float>(2, 0) = xywh[2];
    z.at<float>(3, 0) = xywh[3];
    return z;
}

cv::Vec4f XCYCWHStateEstimator::state_to_bbox() {
    cv::Vec4f xywh(
        kf.x.at<float>(0, 0),
        kf.x.at<float>(1, 0),
        kf.x.at<float>(2, 0),
        kf.x.at<float>(3, 0)
    );
    // Prevent negative width/height exactly as Python does
    if (xywh[2] < 0) xywh[2] = 0;
    if (xywh[3] < 0) xywh[3] = 0;
    
    return xywh_to_xyxy(xywh);
}

void XCYCWHStateEstimator::clamp_velocity() {
    // No clamping needed
}

// --- XYXY Implementation ---
XYXYStateEstimator::XYXYStateEstimator(const cv::Vec4f& bbox) : BaseStateEstimator(8, 4) {
    kf.F = cv::Mat::eye(8, 8, CV_32F);
    kf.F.at<float>(0, 4) = 1.0f;
    kf.F.at<float>(1, 5) = 1.0f;
    kf.F.at<float>(2, 6) = 1.0f;
    kf.F.at<float>(3, 7) = 1.0f;

    kf.H = cv::Mat::eye(4, 8, CV_32F);

    kf.P = cv::Mat::eye(8, 8, CV_32F) * 10.0f;
    kf.P(cv::Rect(4, 4, 4, 4)) *= 1000.0f;

    kf.R = cv::Mat::eye(4, 4, CV_32F) * 10.0f;
    kf.Q = cv::Mat::eye(8, 8, CV_32F) * 0.01f;

    cv::Mat z = bbox_to_measurement(bbox);
    z.copyTo(kf.x(cv::Rect(0, 0, 1, 4)));
}

cv::Mat XYXYStateEstimator::bbox_to_measurement(const cv::Vec4f& bbox) {
    cv::Mat z(4, 1, CV_32F);
    z.at<float>(0, 0) = bbox[0];
    z.at<float>(1, 0) = bbox[1];
    z.at<float>(2, 0) = bbox[2];
    z.at<float>(3, 0) = bbox[3];
    return z;
}

cv::Vec4f XYXYStateEstimator::state_to_bbox() {
    return cv::Vec4f(
        kf.x.at<float>(0, 0),
        kf.x.at<float>(1, 0),
        kf.x.at<float>(2, 0),
        kf.x.at<float>(3, 0)
    );
}

void XYXYStateEstimator::clamp_velocity() {
    // No clamping needed
}

// --- Factory Helper ---
std::unique_ptr<BaseStateEstimator> create_state_estimator(
    StateRepresentation state_repr,
    const cv::Vec4f& bbox) 
{
    switch (state_repr) {
        case StateRepresentation::XCYCSR:
            return std::make_unique<XCYCSRStateEstimator>(bbox);
        case StateRepresentation::XCYCWH:
            return std::make_unique<XCYCWHStateEstimator>(bbox);
        case StateRepresentation::XYXY:
            return std::make_unique<XYXYStateEstimator>(bbox);
        default:
            throw std::invalid_argument("Unknown state representation requested.");
    }
}

} // namespace utils
} // namespace trackers