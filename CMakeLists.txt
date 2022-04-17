cmake_minimum_required(VERSION 3.20)

set(CMAKE_CUDA_STANDARD 14)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 30)
endif()

project(set_mandelbrot_direct2d LANGUAGES CUDA)

add_executable(set_mandelbrot_direct2d main.cu)

set_target_properties(set_mandelbrot_direct2d PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
