cmake_minimum_required(VERSION 2.8)
project( Histogram )
set(CMAKE_CXX_STANDARD 11)
find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

add_library( Filter src/Filter.cpp src/Filter.hpp src/BilateralFilter.cpp src/GaussianFilter.cpp src/MedianFilter.cpp)

add_executable( Histogram src/Histogram.cpp)
target_link_libraries( Histogram ${OpenCV_LIBS} )
target_link_libraries( Histogram Filter )
