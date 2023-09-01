Model Transformations and Optimizations
=======================================

Olive provides multiple transformations and optimizations, also known as Olive Passes, to improve
model performance. Typically, the input model goes through series of transformations before it is
ready for the production. The Olive Passes are designed to receive an input model that is the output
of another Olive Pass.

PyTorch transformations can be applied on the PyTorch model before it is converted to ONNX. After
applying series of ONNX transformations the model could be converted, if needed, to the format
preferred by the native hardware specific SDK.

Olive receives PyTorch or ONNX model as an input. Olive can produce PyTorch or ONNX or hardware
native model as an output.

.. toctree::
   :maxdepth: 2

   passes/pytorch
   passes/onnx
   passes/openvino
   passes/snpe
