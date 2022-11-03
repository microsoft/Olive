#!/bin/sh

conda_env_name=$1
python_version=$2
model_framework=$3
framework_version=$4
cpt_args_str=$5

# create conda env
conda create -n $conda_env_name python=$python_version -y

# activate conda env
conda activate $conda_env_name

# install dependencies
pip install numpy onnx psutil coloredlogs sympy onnxconverter_common docker==5.0.0 six

# install olive
pip install --extra-index-url https://olivewheels.azureedge.net/oaas onnxruntime-olive==0.5.0

# conversion setup in conda env
olive setup --model_framework $model_framework --framework_version $framework_version

# run conversion in conda env
olive convert $cpt_args_str

# deactivate conda env
conda deactivate

conda env remove -n $conda_env_name
