# OLive - ONNX Go Live

OLive, meaning ONNX Go Live, is a sequence of docker images that automates the process of ONNX model shipping. It integrates model conversion, correctness test, and performance tuning into a single pipeline, while each component is a standalone docker image and can be scaled out. 

There are three ways to use OLive:

1. [Use With Command Line Tool](/cmd-tool): Run the OLive with command line using Python. 

2. [Use With Local Web App](/web): A web application with visualization to use OLive on your local machine. 

3. [Use With Jupyter Notebook](/notebook): Quickstart of the OLive with tutorial using Jupyter Notebook. 

4. [Use Pipeline With Kubeflow](/kubeflow): Portable and rapid solution with Kubeflow on Kubernetes to deploy easily manageable 

end-to-end workflow.

The backend of OLive mainly contains two docker images, ONNX converter and performance tuning image. 
1. [ONNX Converter Image](/docker-images/onnx-converter): Converts models from different frameworks to ONNX, generates random inputs, and verifies the correctness of the converted model. The current supported frameworks are Tensorflow, PyTorch, Keras, Scikit-learn, CNTK, and CoreML. 

2. [Performance Tuning Image](/docker-images/perf-tuning): Tunes different execution providers and environment variable options for the converted ONNX model with ONNX Runtime. Selects and outputs the option combinations with the best performance. 

## Contributing
We’d love to embrace your contribution to OLive. Please refer to [CONTRIBUTING.md](./CONTRIBUTING.md).

## License
Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the [MIT](./LICENSE) License.
