cmake_minimum_required(VERSION 3.17.5)

project(mangoct CXX CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

find_package(CUDA REQUIRED)

# since rapidjson is header-only
include_directories(rapidjson/include)
# and we use CUDA
include_directories(${CUDA_INCLUDE_DIRS})

add_subdirectory(mgfbp)
add_subdirectory(mgfpj)
