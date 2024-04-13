cmake_minimum_required(VERSION 3.11)
project(SPH-FLUID)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Find OpenMP package
find_package(OpenMP REQUIRED)

# Add executable and link OpenMP
add_executable(openmp_3d main.cpp openmp.cpp common.h sph.h)
add_executable(serial_3d main.cpp serial.cpp common.h sph.h)

target_link_libraries(openmp_3d PUBLIC OpenMP::OpenMP_CXX)

# Include additional directories if needed
# target_include_directories(openmp_3d PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Optionally, set compiler flags
# target_compile_options(openmp_3d PRIVATE -Wall -Wextra)
