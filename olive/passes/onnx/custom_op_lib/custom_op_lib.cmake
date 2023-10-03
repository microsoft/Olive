# always build cuda for now
enable_language(CUDA)

include(FetchContent)
find_package(Patch)

# add gsl
FetchContent_Declare(
    GSL
    URL https://github.com/microsoft/GSL/archive/refs/tags/v4.0.0.zip
    URL_HASH SHA1=cf368104cd22a87b4dd0c80228919bb2df3e2a14
    PATCH_COMMAND ${Patch_EXECUTABLE} --binary --ignore-whitespace -p1 < ${ONNXRUNTIME_DIR}/cmake/patches/gsl/1064.patch
)
FetchContent_MakeAvailable(GSL)
set(GSL_TARGET Microsoft.GSL::GSL)


set(custom_op_src_patterns
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/cuda/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/cuda/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/cuda/*.cuh"
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/cuda/*.cu"
)

set(custom_op_lib_include ${ONNXRUNTIME_DIR}/include/onnxruntime)
set(custom_op_lib_option)
set(custom_op_lib_link_dir)
set(custom_op_lib_link ${GSL_TARGET})


# always build cuda for now
list(APPEND custom_op_lib_include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
# list(APPEND custom_op_lib_link cudart)

# add custom op library
file(GLOB custom_op_src ${custom_op_src_patterns})

add_library(custom_op_lib SHARED ${custom_op_src})
target_compile_options(custom_op_lib PRIVATE ${custom_op_lib_option})
target_include_directories(custom_op_lib PRIVATE ${custom_op_lib_include})
target_link_libraries(custom_op_lib PRIVATE ${custom_op_lib_link})

if (UNIX)
    if (APPLE)
        set(ONNXRUNTIME_CUSTOM_OP_LIB_LINK_FLAG "-Xlinker -dead_strip")
    else()
        set(ONNXRUNTIME_CUSTOM_OP_LIB_LINK_FLAG "-Xlinker --version-script=${CMAKE_CURRENT_SOURCE_DIR}/csrc/custom_op_library.lds -Xlinker --no-undefined -Xlinker --gc-sections -z noexecstack")
    endif()
else()
    set(ONNXRUNTIME_CUSTOM_OP_LIB_LINK_FLAG "-DEF:${CMAKE_CURRENT_SOURCE_DIR}/csrc/custom_op_library.def")
endif()
set_property(TARGET custom_op_lib APPEND_STRING PROPERTY LINK_FLAGS ${ONNXRUNTIME_CUSTOM_OP_LIB_LINK_FLAG})
