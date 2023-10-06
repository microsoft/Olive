// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_runtime.h>

#ifndef kernels
#define kernels

__global__ void kAddOffset(float *out, const float *offset, int n);

template<typename T, int TILE_SIZE, int THREADS, int NUM_PER_TH, int DATA_TYPE>
__global__ void kDequantizeBlockwise(const float *code, const unsigned char *A, const float *absmax, T *out, const int blocksize, const int n);

#endif
