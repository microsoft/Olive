// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <cuda_runtime.h>

template <typename T>
bool TryMatMulBnb4(
    T* output,
    const T* a_data,
    const unsigned char* b_data_quant,
    const float* absmax,
    const float* datatype,
    int m,
    int n,
    int k,
    int block_size,
    cudaStream_t stream);
