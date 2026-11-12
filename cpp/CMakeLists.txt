cmake_minimum_required(VERSION 3.16)
project(chord_detector CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Detector core: no platform headers. fft_simple.cpp is the desktop FFT backend;
# on the ESP32 replace it with a file implementing the same rfft_magnitude().
add_library(detector src/detector.cpp src/fft_simple.cpp)
target_include_directories(detector PUBLIC src)
target_compile_options(detector PRIVATE -Wall -Wextra)

add_executable(test_detector test/test_main.cpp)
target_link_libraries(test_detector PRIVATE detector)