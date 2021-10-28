#!/bin/bash

export context_dir=`dirname "$0"`
export acr_url=$1
export build_num=$2
onnxruntime_version=$3
use_gpu=$4

cd "$context_dir"

if [ "$use_gpu" == "True" ]
then
  full_image_name="${acr_url}/olive_optimization:${onnxruntime_version}_gpu_${build_num}"
  docker build -t "$full_image_name" --no-cache --build-arg ort_version="$onnxruntime_version" -f Dockerfile.gpu .
else
  full_image_name="${acr_url}/olive_optimization:${onnxruntime_version}_cpu_${build_num}"
  docker build -t "$full_image_name" --no-cache --build-arg ort_version="$onnxruntime_version" -f Dockerfile.cpu .
fi

echo "$full_image_name"
echo "$full_image_name" >> images.txt