#!/bin/sh

export model_filename=$1 
export model_name=$2
export model_type=$3
export in_names=$4
export in_shapes=$5
export in_types=$6
export out_names=$7

pip install pip==22.2.1
# Setup OLive
pip install onnxruntime_olive-0.5.0-py3-none-any.whl
pip install onnxruntime_gpu_tensorrt-1.11.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# Setup model analyzer
pip install git+https://github.com/triton-inference-server/model_analyzer@r22.07

# Setup for conversion
pip install transformers

# Creating entry for the given model on the model repository
mkdir model_repository/${model_name}_default
mkdir model_repository/${model_name}_default/1

# Run OLive conversion
echo olive convert --model_path $model_filename --model_framework $model_type --framework_version 1.11 --input_names $in_names --output_names $out_names --onnx_opset 11 --input_shapes $in_shapes  --input_types $in_types --onnx_model_path model_repository/${model_name}_default/1/model.onnx 
olive convert --model_path $model_filename --model_framework $model_type --framework_version 1.11 --input_names $in_names --output_names $out_names --onnx_opset 11 --input_shapes $in_shapes  --input_types $in_types --onnx_model_path model_repository/${model_name}_default/1/model.onnx 

cp config_model_name-auto-complete.yml config_${model_name}-auto-complete.yml
sed -i "s/model_name/${model_name}/g" config_${model_name}-auto-complete.yml

echo model-analyzer -v profile -f config_${model_name}-auto-complete.yml
model-analyzer -v profile -f config_${model_name}-auto-complete.yml

cp output_model_repository/output_${model_name}_auto_complete/${model_name}_default_config_0/config.pbtxt model_repository/${model_name}_default/
sed -i "s/${model_name}_default_config_0/${model_name}_default/g" model_repository/${model_name}_default/config.pbtxt

# Run OLive optimization
olive optimize --model_path model_repository/${model_name}_default/1/model.onnx --model_analyzer_config model_repository/${model_name}_default/config.pbtxt --providers_list tensorrt,cuda --trt_fp16_enabled --fp32_enabled --result_path ${model_name}_model_analyzer

# Copy OLive Optimized model to Model Repository
mkdir model_repository/${model_name}
mkdir model_repository/${model_name}/1
cp ${model_name}_model_analyzer/optimized_model.onnx model_repository/${model_name}/1/model.onnx
cp ${model_name}_model_analyzer/olive_result.pbtxt model_repository/${model_name}/config.pbtxt

# Create Model Analyzer config file for the given model
cp config_model_name.yml config_${model_name}.yml
sed -i "s/model_name/${model_name}/g" config_${model_name}.yml

# Run Model Analyzer Optimization
model-analyzer -v profile -f config_${model_name}.yml

cp config_model_name_analyze.yml config_${model_name}_analyze.yml
sed -i "s/model_name/${model_name}/g" config_${model_name}_analyze.yml

# Run Model Analyzer Analyze command that would create metrics files
model-analyzer analyze -f config_${model_name}_analyze.yml

pip install pandas
# Process Metrics Files, plot results and find location of optimal config file
python3 Plot_Performance_Results.py --output_repository ./output_model_repository/output_${model_name} --inference_results_file results/metrics-model-inference-${model_name}.csv --output_figure_file Optimal_Results.png --optimal_location_file Optimal_ConfigFile_Location.txt

