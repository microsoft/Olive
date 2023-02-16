# Nvidia Model Analyzer Support

To use OLive optimization result in Nvidia Model Analyzer, user needs to provide ONNX model and config.pbtxt file to be used in Model Analyzer.

## Setup Environment
### OLive Setup 
OLive package can be installed with command `pip install onnxruntime_olive==0.5.0 -f https://olivewheels.azureedge.net/oaas/onnxruntime-olive` 

ONNX Runtime package can be installed with

`pip install onnxruntime_openvino_dnnl==1.11.0 -f https://olivewheels.azureedge.net/oaas/onnxruntime-openvino-dnnl` for cpu

or 

`pip install onnxruntime_gpu_tensorrt==1.11.0 -f https://olivewheels.azureedge.net/oaas/onnxruntime-gpu-tensorrt` for gpu

### Model Analyzer Setup
Please refer to [Model Analyzer Installation Guide](https://github.com/triton-inference-server/model_analyzer/blob/main/docs/install.md)


## Run OLive optimization
To enable optimization for Model Analyzer, user needs to provide argument `--model_analyzer_config path_to_model_analyzer_config_pbtxt`.

For example, to optimize model with TensorRT backend and precision mode FP16, use can call 

`olive optimize --model_path bertaquad.onnx --model_analyzer_config config.pbtxt --providers_list tensorrt --trt_fp16_enabled --result_path bertsquad_model_analyzer`

Optimized model and new generated configuration file will be saved in bertaquad_model_analyzer. User needs to rename model and configuration files, and change location of the files with structure rules of Model Analyzer. 

## Run Model Analyzer
User can run model analyzer profile with command 

`model-analyzer profile --model-repository models --profile-models bertsquad_trt_fp16`

Then user can run model analyzer analyze to generate report for proflie result with command 

`model-analyzer analyze --analysis-models bertsquad_trt_fp16/ -e analysis_result`
