// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

template<typename T, int DATA_TYPE>
void dequantizeBlockwise(float *code, const unsigned char *A, const float *absmax, T *out, int block_size, const int n);
