#!/bin/sh

export model_location=$1
export model_filename=$2
wget https://olivewheels.blob.core.windows.net/repo/onnxruntime_olive-0.4.0-py3-none-any.whl
wget https://olivewheels.blob.core.windows.net/repo/onnxruntime_gpu_tensorrt-1.11.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
echo wget $model_location -O $model_filename
wget $model_location -O $model_filename