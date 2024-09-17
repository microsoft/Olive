#!/bin/bash

# get following instructions from https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/phi-3-vision.md

# output_dir argument
if [ -z "$1" ]
then
    echo "Usage: prepare_phi3_vision_for_olive.sh <output_dir>"
    exit 1
else
    base_output_dir="$1"
    pytorch_output_dir="$base_output_dir"/pytorch
fi

mkdir -p "$pytorch_output_dir"
echo "downloading microsoft/Phi-3-vision-128k-instruct"
huggingface-cli download microsoft/Phi-3-vision-128k-instruct --local-dir "$pytorch_output_dir"

cd "$base_output_dir" || exit
echo "downloading microsoft/Phi-3-vision-128k-instruct-onnx-cpu"
huggingface-cli download microsoft/Phi-3-vision-128k-instruct-onnx-cpu --include "onnx/*" --local-dir .
huggingface-cli download microsoft/Phi-3-vision-128k-instruct-onnx-cpu --include "cpu-int4-rtn-block-32-acc-level-4/*.json" --local-dir .

# In our `config.json`, we replaced `flash_attention_2` with `eager` in `_attn_implementation`
mv onnx/config.json pytorch/

# In our `modeling_phi3_v.py`, we replaced `from .image_embedding_phi3_v import Phi3ImageEmbedding`
# with `from .image_embedding_phi3_v_for_onnx import Phi3ImageEmbedding`
mv onnx/modeling_phi3_v.py pytorch/

# In our `image_embedding_phi3_v_for_onnx.py`, we created a copy of `image_embedding_phi3_v.py` and modified it for exporting to ONNX
mv onnx/image_embedding_phi3_v_for_onnx.py pytorch/

# Move the builder script to the root directory
mv onnx/builder.py .

# mv genai-config.json, preprocess_config.json to output_dir
mv cpu-int4-rtn-block-32-acc-level-4/*.json "$1"

# Delete empty `onnx` directory
rm -rf onnx/ cpu-int4-rtn-block-32-acc-level-4/
