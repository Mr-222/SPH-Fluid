cmake_minimum_required(VERSION 3.11)
project(SPH-FLUID)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Find OpenMP package
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)


# Add executable and link OpenMP
add_executable(openmp_3d main.cpp openmp.cpp common.h sph.h)
add_executable(serial_3d main.cpp serial.cpp common.h sph.h)

target_link_libraries(openmp_3d PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(openmp_3d PUBLIC Threads::Threads)
target_link_libraries(serial_3d PUBLIC Threads::Threads)


