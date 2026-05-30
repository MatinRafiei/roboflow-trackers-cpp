#include "detections.hpp"

namespace trackers {
namespace utils {

std::vector<float> extract_confidences(const Detections& detections) {
    std::vector<float> confs;
    
    // Edge optimization: pre-allocate memory to avoid fragmentation
    confs.reserve(detections.size()); 
    
    for (const auto& det : detections) {
        confs.push_back(det.confidence);
    }
    
    return confs;
}

} // namespace utils
} // namespace trackers