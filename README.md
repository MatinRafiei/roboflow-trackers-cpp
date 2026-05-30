# 🚀 EdgeTrackers-CPP: C++ Multi-Object Tracking Library

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![C++](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)
[![CMake](https://img.shields.io/badge/CMake-Build-success.svg)]()

A highly optimized, hardware-agnostic C++ library for **Multi-Object Tracking (MOT)**, built specifically for real-time Edge AI deployments such as Raspberry Pi, NVIDIA Jetson, RK3588, and other embedded platforms.

This repository provides native C++ implementations of four state-of-the-art tracking algorithms:

- **BoTSORT**
- **ByteTrack**
- **OC-SORT**
- **SORT**

---

## 🤝 Acknowledgments & Original Work

This repository is an unofficial, highly-optimized C++ translation of the excellent multi-object tracking ecosystem found in **Roboflow Trackers**. 
* **Original Python Framework:** [Roboflow Trackers (GitHub)](https://github.com/roboflow/trackers)
* **License:** Ported and distributed under the Apache License 2.0, adhering to the original project's terms.

We are deeply grateful to the Roboflow team and the original researchers for their phenomenal work in the open-source computer vision community. Please see our [`CREDITS.md`](CREDITS.md) file for full attribution regarding OC-SORT and the Hungarian Algorithm implementations.

---

## 🧠 Python to C++ Conversion Notes

While Python is excellent for training and prototyping, running complex Deep-SORT/BoTSORT logic on Edge CPUs can create severe bottlenecks. This C++ port was designed to bridge that gap with the following architectural principles:

* **Hardware Agnostic:** This library does *not* require a specific Neural Processing Unit (NPU). It strictly handles the mathematical tracking associations. You can feed it bounding boxes from Hailo, TensorRT, NCNN, ONNXRuntime, or standard OpenCV.
* **OpenCV Native (`cv::Mat`):** NumPy array manipulations have been strictly mapped to OpenCV's `cv::Mat` and `cv::Vec4f` data structures.
* **Decoupled Architecture:** Just like the original Roboflow design, the mathematical layers (Kalman Filters, IoU Matrices) are decoupled from the base interfaces (`BaseTracker`, `BaseTracklet`), making it incredibly easy to add new tracking algorithms in the future.
* **Memory Safety:** Heavy utilization of `std::unique_ptr` and proper C++17 memory management ensures zero memory leaks during continuous 24/7 video streams.

---

## 📦 Supported Trackers

1. **BoTSORT:** Robust tracking utilizing Camera Motion Compensation (CMC).
2. **ByteTrack:** Extremely fast, two-stage "High-Low" association pipeline (no CMC required).
3. **OC-SORT:** Observation-Centric SORT, relying purely on historical observation data to correct Kalman filter drift during occlusions.
4. **SORT:** The classic Simple Online and Realtime Tracking algorithm.

---

## 🛠️ Installation Guide

### Prerequisites
* **C++17 Compiler** (GCC, Clang, etc.)
* **CMake** (3.10 or higher)
* **OpenCV** (4.x recommended)

### Build and Install
Clone the repository and build the shared library to install it system-wide:

```bash
git clone https://github.com/MatinRafiei/roboflow-trackers-cpp.git
cd roboflow-trackers-cpp

mkdir build
cd build
cmake ..
make -j$(nproc)

# Install the library to your system (requires sudo)
sudo make install
sudo ldconfig
```

*Note*: This creates libroboflow_trackers.so and copies the headers to your system's include path.

---

## 💻 Usage / Quick Start

Because you installed the library system-wide, you can easily use it in any completely separate C++ project.

### 1. Include the headers in your C++ code:

```cpp
#include <iostream>
#include <opencv2/opencv.hpp>
#include <roboflow_trackers/bytetrack_tracker.hpp>

using namespace trackers::core::bytetrack;

int main()
{
    // Initialize ByteTrack
    ByteTrackTracker tracker(
        30,      // lost buffer
        30.0f,   // frame rate
        0.6f,    // track activation threshold
        0.6f,    // high confidence threshold
        0.2f,    // minimum iou threshold for first association
        0.5f,    // minimum iou threshold for second association
        0.3f,    // minimum iou threshold for unconfirmed association
        2,       // minimum consecutive frames
        true     // first frame activation flag
    );

    std::cout << "Successfully loaded ByteTrack Tracker!" << std::endl;

    // tracker.update(detections, frame);

    return 0;
}
```

### 2. Link the library in your custom CMakeLists.txt

```cmake
find_package(OpenCV REQUIRED)

add_executable(my_app main.cpp)

target_link_libraries(
    my_app
    ${OpenCV_LIBS}
    roboflow_trackers
)
```

---

## 🧪 Running the Examples

This repository includes a standalone test executable that runs a 2-frame simulation across all four trackers to verify mathematical parity and system stability.

If you enabled building examples during CMake (which is ON by default), you can run the test directly from the build folder:

```bash
./examples/demo_trackers
```

### Expected Output

```text
Initializing Tracker Suite...

==================================================
 Testing: BoTSORT Tracker
==================================================
[FRAME 1] Simulating 1 detection at [100, 100, 150, 150]...
  -> Track ID: 0 | BBox: [100, 100, 150, 150]

[FRAME 2] Moving object slightly to [105, 105, 155, 155]...
  -> Track ID: 0 | BBox: [105, 105, 155, 155]
--------------------------------------------------

==================================================
 Testing: ByteTrack Tracker
==================================================
[FRAME 1] Simulating 1 detection at [100, 100, 150, 150]...
  -> Track ID: 1 | BBox: [100, 100, 150, 150]

[FRAME 2] Moving object slightly to [105, 105, 155, 155]...
  -> Track ID: 1 | BBox: [105, 105, 155, 155]
--------------------------------------------------

==================================================
 Testing: OC-SORT Tracker
==================================================
[FRAME 1] Simulating 1 detection at [100, 100, 150, 150]...
  -> Track ID: 2 | BBox: [100, 100, 150, 150]

[FRAME 2] Moving object slightly to [105, 105, 155, 155]...
  -> Track ID: 2 | BBox: [105, 105, 155, 155]
--------------------------------------------------

==================================================
 Testing: Standard SORT Tracker
==================================================
[FRAME 1] Simulating 1 detection at [100, 100, 150, 150]...
  -> Track ID: 3 | BBox: [100, 100, 150, 150]

[FRAME 2] Moving object slightly to [105, 105, 155, 155]...
  -> Track ID: 3 | BBox: [105, 105, 155, 155]
--------------------------------------------------

All tracking tests completed successfully!
```

---

## 📂 Repository Structure

```text
├── include/roboflow_trackers/ # Header files (.hpp)
├── src/                       # Core C++ implementations (.cpp)
├── examples/                  # Demo executables and tests
├── CMakeLists.txt             # Master build configuration
├── LICENSE                    # Apache 2.0 License
└── CREDITS.md                 # Open-Source Attribution
```

---

## ⭐ Contributing

Contributions are welcome.

Whether you're:

- Fixing bugs
- Improving performance
- Adding new trackers
- Enhancing documentation

feel free to submit a pull request or open an issue.

---

## 📄 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

---

Built for the Edge AI community ❤️

---
