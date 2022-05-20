// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#include <cstdio>
#include <fstream>
#include <regex>

// RapidJSON
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>

// GCC 7.5 <filesystem> workaround
#ifdef __cpp_lib_filesystem
  #include <filesystem>
  namespace fs = std::filesystem;
#else
  #include <experimental/filesystem>
  namespace fs = std::experimental::filesystem;
#endif

// CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
