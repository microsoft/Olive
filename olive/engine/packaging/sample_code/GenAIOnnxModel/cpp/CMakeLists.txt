# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.10)
project(olive-genai-cpp-sample)
set(CMAKE_CXX_STANDARD 20)

add_executable(olive-genai-cpp-sample code_sample.cpp)
target_include_directories(olive-genai-cpp-sample
    PRIVATE include
    PRIVATE include/onnxruntime-genai
)
target_link_libraries(olive-genai-cpp-sample
    PRIVATE onnxruntime-genai
)
target_link_directories(olive-genai-cpp-sample
    PRIVATE lib
)

if (MSVC)
  # MSVC doesn't report correct value for __cplusplus without the explicit flag
  # Ref: https://devblogs.microsoft.com/cppblog/msvc-now-correctly-reports-__cplusplus/
  target_compile_options(olive-genai-cpp-sample PRIVATE "/Zc:__cplusplus")

  add_custom_command(TARGET olive-genai-cpp-sample POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${PROJECT_SOURCE_DIR}/lib/onnxruntime.dll"
          $<TARGET_FILE_DIR:olive-genai-cpp-sample>
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${PROJECT_SOURCE_DIR}/lib/onnxruntime-genai.dll"
          $<TARGET_FILE_DIR:olive-genai-cpp-sample>
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${PROJECT_SOURCE_DIR}/lib/onnxruntime_providers_shared.dll"
          $<TARGET_FILE_DIR:olive-genai-cpp-sample>
  )
endif()
