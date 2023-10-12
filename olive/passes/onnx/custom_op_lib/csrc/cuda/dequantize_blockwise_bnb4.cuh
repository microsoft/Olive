// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <cuda_runtime.h>

void addOffset(float *out, const float *offset, int n, cudaStream_t stream);

template<typename T, int DATA_TYPE>
void dequantizeBlockwise(const float *quant_map, const unsigned char *A, const float *absmax, T *out, int block_size, const int n, cudaStream_t stream);
