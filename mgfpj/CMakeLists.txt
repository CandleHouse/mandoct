cmake_minimum_required(VERSION 3.17.5)

project(mgfpj CXX CUDA)

set(SOURCES
    src/mgfpj.cpp
    src/stdafx.cpp  
    src/FpjClass.cpp
    src/FpjClass_Agent.cu
)
include_directories(mgfpj PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_executable(mgfpj ${SOURCES})

target_compile_features(mgfpj PUBLIC cxx_std_17)
target_link_libraries(mgfpj
    ${CUDA_LIBRARIES}
    stdc++fs # Remove this line for GCC > 8.2 which fully supports <filesystem>
)
set_target_properties(mgfpj PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

install(TARGETS mgfpj)
