// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_runtime.h>

#ifndef cuda_ops
#define cuda_ops

template<typename T, int DATA_TYPE>
void dequantizeBlockwise(float *code, const unsigned char *A, const float *absmax, T *out, int block_size, const int n, cudaStream_t stream);

#endif
