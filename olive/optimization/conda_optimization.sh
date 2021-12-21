#!/bin/sh

conda_env_name=$1
python_version=$2
use_gpu=$3
opt_args_str=$4
onnxruntime_version=$5

# create conda env
conda create -n $conda_env_name python=$python_version -y

# activate conda env
conda activate $conda_env_name

# install dependencies
pip install numpy onnx psutil coloredlogs sympy onnxconverter_common docker==5.0.0 six

# install olive
pip install --index-url https://olivewheels.azureedge.net/test onnxruntime-olive==0.2.0

# optimization setup in conda env
if [ $use_gpu == "True" ]
then
  olive setup --onnxruntime_version $onnxruntime_version --use_gpu
else
  olive setup --onnxruntime_version $onnxruntime_version
fi

# run optimization in conda env
olive optimize $opt_args_str

# deactivate conda env
conda deactivate

conda env remove -n $conda_env_name
