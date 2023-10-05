// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>

#include "cuda_ops.cuh"
#include "kernels.cuh"

#define CUDA_CHECK_RETURN(value) {                      \
  cudaError_t _m_cudaStat = value;                    \
  if (_m_cudaStat != cudaSuccess) {                   \
    fprintf(stderr, "Error %s at line %d in file %s\n",         \
        cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);   \
    exit(1);                              \
  } }

template<typename T, int DATA_TYPE>
void dequantizeBlockwise(float *code, const unsigned char *A, const float *absmax, T *out, int blocksize, const int n)
{
  int num_blocks = n/blocksize;
  num_blocks = n % blocksize == 0 ? num_blocks : num_blocks + 1;
  int tile_size = (DATA_TYPE > 0) ? 1024 : 512;

  if(DATA_TYPE > 0)
    kDequantizeBlockwise<T, 512, 64, 8, DATA_TYPE><<<(n+tile_size-1)/tile_size, 64>>>(code, A, absmax, out, blocksize/2, n);
  else
    kDequantizeBlockwise<T, 512, 64, 8, DATA_TYPE><<<(n+tile_size-1)/tile_size, 64>>>(code, A, absmax, out, blocksize, n);

  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}
