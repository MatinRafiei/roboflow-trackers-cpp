# Credits & Acknowledgments

This repository is a C++ translation of algorithms and architectures originally developed by others in the open-source computer vision community. We are deeply grateful to the original authors for their phenomenal research and engineering.

## 1. Roboflow Trackers
The core object-oriented architecture, Base Tracker abstractions, Kalman Filter scaling logic, and bounding box math in this repository are directly ported from the Python implementation by **Roboflow**.
* **Repository:** [https://github.com/roboflow/trackers](https://github.com/roboflow/trackers)
* **License:** Apache License 2.0
* **Copyright:** (c) 2026 Roboflow. All Rights Reserved.

## 2. OC-SORT (Observation-Centric SORT)
The implementation of Observation-Centric Momentum (OCM) and Observation-centric Re-Update (ORU) within the OC-SORT tracker logic is adapted based on the original OC-SORT research and implementation.
* **Original Repository:** [https://github.com/noahcao/OC_SORT](https://github.com/noahcao/OC_SORT)
* **License:** MIT License

    MIT License
    Copyright (c) 2022 Jinkun Cao

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

## 3. Hungarian Algorithm (C++ Implementation)
The Linear Sum Assignment solving required for the multi-object tracking association steps relies on the C++ implementation of the Hungarian Algorithm by **Cong Ma**, which builds upon work by **Markus Buehren**.
* **Repository:** [https://github.com/mcximing/hungarian-algorithm-cpp](https://github.com/mcximing/hungarian-algorithm-cpp)
* **License:** BSD 2-Clause License

    Copyright (c) 2014, Markus Buehren
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

        * Redistributions of source code must retain the above copyright
          notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
          notice, this list of conditions and the following disclaimer in
          the documentation and/or other materials provided with the distribution

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.