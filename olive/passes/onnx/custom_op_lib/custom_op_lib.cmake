# always build cuda for now
# enable_language(CUDA)

include(FetchContent)

# ort package is missing onnxruntime_float16.h
# download and make available onnxruntime
# set(ONNXRUNTIME_VER "1.16.0")
# if(WIN32)
#     set(ONNXRUNTIME_URL "v${ONNXRUNTIME_VER}/onnxruntime-win-x64-gpu-${ONNXRUNTIME_VER}.zip")
# else()
#     set(ONNXRUNTIME_URL "v${ONNXRUNTIME_VER}/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VER}.tgz")
# endif()
# set(ort_fetch_URL "https://github.com/microsoft/onnxruntime/releases/download/${ONNXRUNTIME_URL}")

# message(STATUS "ONNX Runtime URL: ${ort_fetch_URL}")
# FetchContent_Declare(
# onnxruntime
# URL ${ort_fetch_URL}
# )

# FetchContent_makeAvailable(onnxruntime)

# local copy with missing header
# set(onnxruntime_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-src)

# message(STATUS "ONNX Runtime source dir: ${onnxruntime_SOURCE_DIR}")

# set(ONNXRUNTIME_INCLUDE_DIR ${onnxruntime_SOURCE_DIR}/include)
# set(ONNXRUNTIME_LIB_DIR ${onnxruntime_SOURCE_DIR}/lib)
set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_DIR}/include/onnxruntime/core/session)

# set(custom_op_src_patterns
#     "${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.h"
#     "${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cc"
#     "${CMAKE_CURRENT_SOURCE_DIR}/csrc/cuda/*.cuh"
#     "${CMAKE_CURRENT_SOURCE_DIR}/csrc/cuda/*.h"
#     "${CMAKE_CURRENT_SOURCE_DIR}/csrc/cuda/*.cu"
#     "${CMAKE_CURRENT_SOURCE_DIR}/csrc/cuda/*.cc"
# )

set(custom_op_lib_include ${ONNXRUNTIME_INCLUDE_DIR})
set(custom_op_lib_option)
set(custom_op_lib_link)
set(custom_op_lib_link_dir)
# set(custom_op_lib_link_dir ${ONNXRUNTIME_LIB_DIR})
# set(custom_op_lib_link onnxruntime)


# always build cuda for now
list(APPEND custom_op_lib_include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
list(APPEND custom_op_lib_link_dir ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
list(APPEND custom_op_lib_link cudart)
# list(APPEND custom_op_lib_link cudart cublas cublasLt cusparse)

# add custom op library
# file(GLOB custom_op_src ${custom_op_src_patterns})

# add_library(custom_op_lib SHARED ${custom_op_src})
add_library(custom_op_lib SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/custom_op_library.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/custom_op_library.h
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/cuda/cuda_ops.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/cuda/cuda_ops.h
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/cuda/cuda_ops.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/cuda/cuda_ops.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/cuda/kernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/cuda/kernels.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/cuda/common.h
)

target_compile_options(custom_op_lib PRIVATE ${custom_op_lib_option})
target_include_directories(custom_op_lib PRIVATE ${custom_op_lib_include})
target_link_directories(custom_op_lib PRIVATE ${custom_op_lib_link_dir})
target_link_libraries(custom_op_lib PRIVATE ${custom_op_lib_link})

if (WIN32)
    set(ONNXRUNTIME_CUSTOM_OP_LIB_LINK_FLAG "-DEF:${CMAKE_CURRENT_SOURCE_DIR}/csrc/custom_op_library.def")
else()
    set(ONNXRUNTIME_CUSTOM_OP_LIB_LINK_FLAG "-Xlinker --version-script=${CMAKE_CURRENT_SOURCE_DIR}/csrc/custom_op_library.lds -Xlinker --no-undefined -Xlinker --gc-sections -z noexecstack")
endif()
set_property(TARGET custom_op_lib APPEND_STRING PROPERTY LINK_FLAGS ${ONNXRUNTIME_CUSTOM_OP_LIB_LINK_FLAG})
