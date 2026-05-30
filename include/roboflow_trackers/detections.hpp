#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace trackers {
namespace utils {

/**
 * @brief C++ replacement for supervision.Detections
 * Represents a single detected object in a frame.
 */
struct Detection {
    // Bounding box in [x_min, y_min, x_max, y_max] format
    cv::Vec4f bbox; 
    
    // Default to 1.0f. This completely replaces the Python `default_confidences` function!
    float confidence = 1.0f; 
    
    // Optional: Classification info from your object detector (e.g., YOLO)
    int class_id = -1;       
    
    // Will be populated by the tracker (-1 means unconfirmed/unassigned)
    int tracker_id = -1;     
};

// Define 'Detections' as a standard vector of Detection structs
using Detections = std::vector<Detection>;

// Utility function to extract an array of confidences if needed for vectorized math
std::vector<float> extract_confidences(const Detections& detections);

} // namespace utils
} // namespace trackers