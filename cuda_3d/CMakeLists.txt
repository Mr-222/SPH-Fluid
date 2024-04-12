cmake_minimum_required(VERSION 3.11)
project(SPH-FLUID)
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

add_executable(gpu main.cu gpu.cu common.h sph.cuh)
target_compile_features(gpu PRIVATE cxx_std_17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_70")
set_target_properties(gpu PROPERTIES CUDA_ARCHITECTURES "70")

configure_file(salloc.sh salloc.sh COPYONLY)
configure_file(env.sh env.sh COPYONLY)