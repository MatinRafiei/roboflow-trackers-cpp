# We don't need project() or cmake_minimum_required() here 
# because the root CMake already handled it!

# 1. Create the executable
add_executable(demo_trackers demo_trackers.cpp)

# 2. Link OpenCV and our brand new custom library
target_link_libraries(demo_trackers 
    roboflow_trackers 
    ${OpenCV_LIBS}
)