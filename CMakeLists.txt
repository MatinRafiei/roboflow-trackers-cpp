cmake_minimum_required(VERSION 3.10)
project(RoboflowTrackersCPP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -fPIC")

# Allow users to choose whether to build the examples (Default is ON)
option(BUILD_EXAMPLES "Build the example demonstration programs" ON)

find_package(OpenCV REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/roboflow_trackers
    ${OpenCV_INCLUDE_DIRS}
)

# 1. Grab all the core tracker files
file(GLOB TRACKER_SOURCES "src/*.cpp")

# 2. Create the Shared Library
add_library(roboflow_trackers SHARED ${TRACKER_SOURCES})
target_link_libraries(roboflow_trackers ${OpenCV_LIBS})

# 3. System Install Rules (The "pip install" equivalent)
install(TARGETS roboflow_trackers DESTINATION lib)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION include)

# 4. Route to the Examples folder (if the user wants them)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()