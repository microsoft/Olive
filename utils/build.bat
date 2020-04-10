:: Copyright (c) Microsoft Corporation.
:: Licensed under the MIT License.

:: pull onnx-converter and perf-tuning docker images from mcr
call docker pull mcr.microsoft.com/onnxruntime/onnx-converter
call docker pull mcr.microsoft.com/onnxruntime/perf-tuning

:: install python dependency modules
call python -m pip install docker pandas pickle onnx
