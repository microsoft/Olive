# always build cuda for now
# enable_language(CUDA)

set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_DIR}/include/onnxruntime/core/session)

set(custom_op_lib_include ${ONNXRUNTIME_INCLUDE_DIR})
set(custom_op_lib_option)
set(custom_op_lib_link)
set(custom_op_lib_link_dir)

# always build cuda for now
list(APPEND custom_op_lib_include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
list(APPEND custom_op_lib_link_dir ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
list(APPEND custom_op_lib_link cudart cublas)

# add custom op library
add_library(custom_op_lib SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/custom_op_library.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/custom_op_library.h
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/cuda/matmul_bnb4.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/cuda/matmul_bnb4.h
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/cuda/matmul_bnb4.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/cuda/matmul_bnb4.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/cuda/dequantize_blockwise_bnb4.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/cuda/dequantize_blockwise_bnb4.cuh
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
