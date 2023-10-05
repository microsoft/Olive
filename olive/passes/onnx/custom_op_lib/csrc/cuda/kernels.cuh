// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef kernels
#define kernels

template<typename T, int TILE_SIZE, int THREADS, int NUM_PER_TH, int DATA_TYPE>
__global__ void kDequantizeBlockwise(float *code, const unsigned char *A, const float *absmax, T *out, const int blocksize, const int n);

#endif
