# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Minimum CMake required
cmake_minimum_required(VERSION 3.26)

# Project
project(bnb_custom_op C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
if(WIN32)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS 1)
ENDIF()

include(custom_op_lib.cmake)
