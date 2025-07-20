# AMD Quark for ONNX Model Support

The following models and techniques are demonstrated in the AMD Quark for ONNX examples:

| Quantization Technique                            | Models                                                                            |
| ------------------------------------------------- | --------------------------------------------------------------------------------- |
| Accuracy improvement¹                             | densenet121, Llama2-7B, mobilenetv2_050, opt-125m, resnet50-v1-12, timm/resnet152 |
| Auto-Search                                       | Yolov3                                                                            |
| Dynamic Quantization                              | Llama-2-7b, opt-125m                                                              |
| Hugging Face TIMM Quantization                    | mobilenetv2_100, various                                                          |
| Image Classification                              | Resnet50-v1-12                                                                    |
| Language Model Quantization                       | opt-125m                                                                          |
| Auto-Search for AMD Ryzen AI                      | Resnet50-v1-12, yolo_nas_s, yolov3                                                |
| Post Training Quantization (PTQ) for AMD Ryzen AI | Resnet50-v1-12                                                                    |
| Object Detection PTQ for AMD Ryzen AI             | yolov8n-face                                                                      |

¹Accuracy improvement includes examples with: AdaQuant, AdaRound, Block Floating Point (BFP), Cross-Layer Equalization (CLE), GPTQ, Mixed Precision, Microscaling (MX) data types, QuaRot, and SmoothQuant.
