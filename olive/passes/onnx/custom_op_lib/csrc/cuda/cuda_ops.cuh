// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_runtime.h>

#ifndef cuda_ops
#define cuda_ops

void addOffset(float *out, const float *offset, int n, cudaStream_t stream);

template<typename T, int DATA_TYPE>
void dequantizeBlockwise(const float *code, const unsigned char *A, const float *absmax, T *out, int block_size, const int n, cudaStream_t stream);

#endif
