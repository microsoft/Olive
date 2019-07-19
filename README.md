# ONNX Automation Pipeline

ONNX Automation Pipeline is a sequence of docker images that automates the process of ONNX model shipping. It integrates model conversion, correctness test, and performance tuning into a single pipeline, while each component is a standalone docker image and can be scaled out. 

There are three ways to use the pipeline:

1. [Use Pipeline With Kubeflow](/pipelines): Portable and rapid solution with Kubeflow on Kubernetes to deploy easily manageable end-to-end workflow.

2. [Use With Jupyter Notebook](/notebook): Quickstart of the pipeline with tutorial using Jupyter Notebook. 

3. [Use With Command Line Tool](/windows): Run the pipeline with command line using Python. 