cmake_minimum_required(VERSION 3.17.5)

project(mgfpj CXX CUDA)

set(SOURCES
    src/mgfbp.cpp
    src/stdafx.cpp
    src/FbpClass.cu
    src/FbpClass_Agent.cu
)
include_directories(mgfbp PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_executable(mgfbp ${SOURCES})

target_compile_features(mgfbp PUBLIC cxx_std_17)
target_link_libraries(mgfbp
    ${CUDA_LIBRARIES}
    stdc++fs # Remove this line for GCC > 8.2 which fully supports <filesystem>
)
set_target_properties(mgfbp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

install(TARGETS mgfbp)
