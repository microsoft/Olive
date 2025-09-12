# Examples

|Scenario| Model|Examples|Hardware Targeted Optimization|
|---|-----------|-----------|-----------|
||llama2|[Link](https://github.com/microsoft/Olive/tree/main/examples/llama2)|`CPU`: with ONNX Runtime optimizations for optimized FP32 ONNX model<br>`CPU`: with ONNX Runtime optimizations for optimized INT8 ONNX model<br>`CPU`: with ONNX Runtime optimizations for optimized INT4 ONNX model<br>`GPU`: with ONNX Runtime optimizations for optimized FP16 ONNX model<br>`GPU`: with ONNX Runtime optimizations for optimized INT4 ONNX model<br>`GPU`: with QLoRA for model fine tune and ONNX Runtime optimizations for optimized ONNX model
||audio spectrogram<br>transformer|[Link](https://github.com/microsoft/Olive/tree/main/examples/ast)|`CPU`: with ONNX Runtime optimizations and quantization for optimized INT8 ONNX model
||mobilenet|[Link](https://github.com/microsoft/Olive/tree/main/examples/mobilenet)|`QNN EP`: with ONNX Runtime static QDQ quantization for ONNX Runtime QNN EP
||resnet|[Link](https://github.com/microsoft/Olive/tree/main/examples/resnet)|`CPU`: with ONNX Runtime static/dynamic Quantization for ONNX INT8 model<br>`QDQ`: with ONNX Runtime static Quantization for ONNX INT8 model with QDQ format<br>`AMD DPU`: with AMD Vitis-AI Quantizationg
