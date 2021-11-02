#!/bin/bash

export context_dir=`dirname "$0"`
export acr_url=$1
export build_num=$2
export framework=$3
export framework_version=$4

cd "$context_dir"

cp requirements/"${framework}_${framework_version}.txt" ./requirements.txt
full_image_name="${acr_url}/olive_conversion:${framework}_${framework_version}_${build_num}"
docker build -t "$full_image_name" --no-cache .

echo "$full_image_name"
echo "$full_image_name" >> images.txt
