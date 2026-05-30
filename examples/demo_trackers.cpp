#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

// Include all tracker headers
#include "base_tracker.hpp"
#include "botsort_tracker.hpp"
#include "bytetrack_tracker.hpp"
#include "ocsort_tracker.hpp"
#include "sort_tracker.hpp"
#include "detections.hpp"

using namespace trackers::core;
using namespace trackers::utils;

// A reusable test function that accepts ANY tracker
void run_tracker_test(BaseTracker& tracker, const std::string& tracker_name) {
    std::cout << "\n==================================================" << std::endl;
    std::cout << " Testing: " << tracker_name << std::endl;
    std::cout << "==================================================" << std::endl;

    cv::Mat empty_frame;

    // --- FRAME 1 ---
    std::cout << "[FRAME 1] Simulating 1 detection at [100, 100, 150, 150]..." << std::endl;
    Detections frame1_dets;
    Detection d1;
    d1.bbox = cv::Vec4f(100.0f, 100.0f, 150.0f, 150.0f); 
    d1.confidence = 0.9f;
    frame1_dets.push_back(d1);

    Detections out1 = tracker.update(frame1_dets, empty_frame);
    for (const auto& d : out1) {
        std::cout << "  -> Track ID: " << d.tracker_id << " | BBox: [" 
                  << d.bbox[0] << ", " << d.bbox[1] << ", " << d.bbox[2] << ", " << d.bbox[3] << "]\n";
    }

    // --- FRAME 2 ---
    std::cout << "\n[FRAME 2] Moving object slightly to [105, 105, 155, 155]..." << std::endl;
    Detections frame2_dets;
    Detection d2;
    d2.bbox = cv::Vec4f(105.0f, 105.0f, 155.0f, 155.0f); 
    d2.confidence = 0.85f;
    frame2_dets.push_back(d2);

    Detections out2 = tracker.update(frame2_dets, empty_frame);
    for (const auto& d : out2) {
        std::cout << "  -> Track ID: " << d.tracker_id << " | BBox: [" 
                  << d.bbox[0] << ", " << d.bbox[1] << ", " << d.bbox[2] << ", " << d.bbox[3] << "]\n";
    }
    std::cout << "--------------------------------------------------\n";
}

int main() {
    std::cout << "Initializing Tracker Suite...\n";

    // 1. Initialize BoTSORT
    botsort::BoTSORTTracker botsort_tracker(30, 30.0f, 0.5f, 2, 0.2f, 0.5f, 0.3f, 0.6f, false);
    run_tracker_test(botsort_tracker, "BoTSORT Tracker");

    // 2. Initialize ByteTrack
    bytetrack::ByteTrackTracker bytetrack_tracker(30, 30.0f, 0.6f, 0.6f, 0.2f, 0.5f, 0.3f, 2, true);
    run_tracker_test(bytetrack_tracker, "ByteTrack Tracker");

    // 3. Initialize OC-SORT
    ocsort::OCSORTTracker ocsort_tracker(30, 30.0f, 0.3f, 0.3f, 0.3f, 1, 0.2f, 3);
    run_tracker_test(ocsort_tracker, "OC-SORT Tracker");

    // 4. Initialize Standard SORT
    sort::SORTTracker sort_tracker(30, 30.0f, 0.3f, 0.3f, 0.3f, 1);
    run_tracker_test(sort_tracker, "Standard SORT Tracker");

    std::cout << "\nAll tracking tests completed successfully!" << std::endl;
    return 0;
}