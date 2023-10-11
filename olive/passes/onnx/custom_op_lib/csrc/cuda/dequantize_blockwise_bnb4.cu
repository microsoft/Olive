// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <float.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cuda_fp16.h>

#include "common.h"
#include "dequantize_blockwise_bnb4.cuh"

__global__ void kAddOffset(float *out, const float *offset, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) {
    out[idx] += *offset;
  }
}


void addOffset(float *out, const float *offset, int n, cudaStream_t stream)
{
  kAddOffset<<<((n+1024-1)/1024), 1024, 0, stream>>>(out, offset, n);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}


__device__ float dDequantizeFP4Tree(unsigned char val, float absmax)
{
  float sign = (val & 0b1000) == 8 ? -1.0f : 1.0f;
  if((val & 0b0100) == 4) // 0
    if((val & 0b0010) == 2) //01
      if((val & 0b0001) == 1) // 111
        return 0.25000000f*absmax*sign; // 1111
      else
        return 0.16666667f*absmax*sign; // 1110
    else
      if((val & 0b0001) == 1) // 110
        return 0.50000000f*absmax*sign; // 1101
      else
        return 0.33333333f*absmax*sign; // 1100
  else
    if((val & 0b0010) == 2) //10
      if((val & 0b0001) == 1) // 101
        return 1.00000000f*absmax*sign; // 1011
      else
        return 0.66666667f*absmax*sign; // 1010
    else
      if((val & 0b0001) == 1) // 100
        return 5.208333333e-03f*absmax*sign; // 1001
      else
        return 0.00000000f*absmax*sign; // 1000
}

__device__ float dDequantizeNF4(unsigned char val)
{

  // the values for this tree was generated by test_normal_map_tree
  // in the file tests/test_functional.py
  if((val & 0b1000) == 8)
    if((val & 0b0100) == 4) // 1
      if((val & 0b0010) == 2) // 11
        if((val & 0b0001) == 1) // 111
          return 1.0f;
        else
          return 0.7229568362236023f;
      else
        if((val & 0b0001) == 1) // 110
          return 0.5626170039176941f;
        else
          return 0.44070982933044434f;
    else
      if((val & 0b0010) == 2) //10
        if((val & 0b0001) == 1) // 101
          return 0.33791524171829224f;
        else
          return 0.24611230194568634f;
      else
        if((val & 0b0001) == 1) // 100
          return 0.16093020141124725f;
        else
          return 0.07958029955625534f;

  else
    if((val & 0b0100) == 4) // 0
      if((val & 0b0010) == 2) //01
        if((val & 0b0001) == 1) // 011
          return 0.0f;
        else
          return -0.09105003625154495f;
      else
        if((val & 0b0001) == 1) // 010
          return -0.18477343022823334f;
        else
          return -0.28444138169288635f;
    else
      if((val & 0b0010) == 2) //00
        if((val & 0b0001) == 1) // 001
          return -0.39491748809814453f;
        else
          return -0.5250730514526367f;
      else
        if((val & 0b0001) == 1) // 000
          return -0.6961928009986877f;
        else
          return -1.0f;

}


template<typename T, int TILE_SIZE, int THREADS, int NUM_PER_TH, int DATA_TYPE>
__global__ void kDequantizeBlockwise(const float *code, const unsigned char *A, const float *absmax, T *out, const int blocksize, const int n)
{
  const int n_load = (gridDim.x * TILE_SIZE);
  int valid_items_load = 0;
  int valid_items_store = 0;
  const int base_idx = (blockIdx.x * TILE_SIZE);

  T vals[NUM_PER_TH*((DATA_TYPE > 0) ? 2 : 1)];
  unsigned char qvals[NUM_PER_TH];
  float local_abs_max = -FLT_MAX;

  typedef cub::BlockLoad<unsigned char, THREADS, NUM_PER_TH, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadChar;
  typedef cub::BlockStore<T, THREADS, NUM_PER_TH*((DATA_TYPE > 0) ? 2 : 1), cub::BLOCK_STORE_WARP_TRANSPOSE> StoreT;

  __shared__ typename LoadChar::TempStorage loadchar;
  __shared__ typename StoreT::TempStorage storet;

  for (unsigned int i = base_idx; i < n_load; i += gridDim.x*TILE_SIZE)
  {
    if(DATA_TYPE > 0)
    {
      valid_items_load = (n+1)/2 - i > TILE_SIZE ? TILE_SIZE : (n+1)/2 - i;
      valid_items_store = n - i*2 > TILE_SIZE*2 ? TILE_SIZE*2 : n - i*2;
    }
    else
    {
      valid_items_load = n - i > TILE_SIZE ? TILE_SIZE : n - i;
      valid_items_store = n - i > TILE_SIZE ? TILE_SIZE : n - i;
    }
    local_abs_max = __ldg(&absmax[(i+threadIdx.x*NUM_PER_TH)/(blocksize)]);

    __syncthreads();
    LoadChar(loadchar).Load(&(A[i]), qvals, valid_items_load, 128);

    switch(DATA_TYPE)
    {
      case General8bit:
        // load code through read-only cache via __ldg
        #pragma unroll NUM_PER_TH
        for(int j = 0; j < NUM_PER_TH; j++)
          vals[j] = __ldg(&code[qvals[j]])*local_abs_max;
        break;
      case FP4:
        #pragma unroll NUM_PER_TH
        for(int j = 0; j < NUM_PER_TH; j++)
        {
          vals[j*2] = dDequantizeFP4Tree(qvals[j] >> 4, local_abs_max);
          vals[j*2 + 1] = dDequantizeFP4Tree(qvals[j] & 0x0F, local_abs_max);
        }
        break;
      case NF4:
        #pragma unroll NUM_PER_TH
        for(int j = 0; j < NUM_PER_TH; j++)
        {
          vals[j*2] = dDequantizeNF4(qvals[j] >> 4)* local_abs_max;
          vals[j*2 + 1] = dDequantizeNF4(qvals[j] & 0x0F)* local_abs_max;
        }
        break;
    }

    __syncthreads();
    StoreT(storet).Store(&(out[(DATA_TYPE > 0) ? i*2 : i]), vals, valid_items_store);
  }
}


template<typename T, int DATA_TYPE>
void dequantizeBlockwise(const float *code, const unsigned char *A, const float *absmax, T *out, int blocksize, const int n, cudaStream_t stream)
{
  // int num_blocks = n/blocksize;
  // num_blocks = n % blocksize == 0 ? num_blocks : num_blocks + 1;
  int tile_size = (DATA_TYPE > 0) ? 1024 : 512;

  if(DATA_TYPE > 0)
    kDequantizeBlockwise<T, 512, 64, 8, DATA_TYPE><<<(n+tile_size-1)/tile_size, 64, 0, stream>>>(code, A, absmax, out, blocksize/2, n);
  else
    kDequantizeBlockwise<T, 512, 64, 8, DATA_TYPE><<<(n+tile_size-1)/tile_size, 64, 0, stream>>>(code, A, absmax, out, blocksize, n);

  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

template void dequantizeBlockwise<float, General8bit>(const float *code, const unsigned char *A, const float *absmax, float *out, int blocksize, const int n, cudaStream_t stream);
template void dequantizeBlockwise<float, FP4>(const float *code, const unsigned char *A, const float *absmax, float *out, int blocksize, const int n, cudaStream_t stream);
template void dequantizeBlockwise<float, NF4>(const float *code, const unsigned char *A, const float *absmax, float *out, int blocksize, const int n, cudaStream_t stream);

template void dequantizeBlockwise<half, General8bit>(const float *code, const unsigned char *A, const float *absmax, half *out, int blocksize, const int n, cudaStream_t stream);
template void dequantizeBlockwise<half, FP4>(const float *code, const unsigned char *A, const float *absmax, half *out, int blocksize, const int n, cudaStream_t stream);
template void dequantizeBlockwise<half, NF4>(const float *code, const unsigned char *A, const float *absmax, half *out, int blocksize, const int n, cudaStream_t stream);
