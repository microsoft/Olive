Best Practice for Ryzen AI in Quark ONNX
========================================

This topic outlines best practice for Post-Training Quantization (PTQ) in Quark ONNX. It provides guidance on fine-tuning your quantization strategy to meet target quantization accuracy.


.. figure:: ../_static/best_practice_in_quark_onnx.png
   :align: center
   :width: 85%

   **Figure 1. Best Practices for Quark ONNX Quantization**

Pip Requirements
----------------

Install the necessary python packages:

.. code-block:: bash

   python -m pip install -r requirements.txt

Prepare model from Torch to ONNX (Optional)
-------------------------------------------

Download the Yolov8 pytorch float model:

.. code-block:: bash


   wget -P models https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt

Export the Yolov8 pytorch model to onnx model:

.. code-block:: bash


   python export_yolo_to_onnx.py --input_model_path models/yolov8n-face.pt --output_model_path yolov8n-face.onnx


Prepare model
-------------

Download the Yolov8 ONNX float model:

.. code-block:: bash

   wget -P models https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.onnx


Prepare Calibration Data
------------------------

You can provide a folder containing PNG or JPG files as calibration data folder. For example, you can download images from https://github.com/microsoft/onnxruntime-inference-examples/tree/main/quantization/image_classification/cpu/test_images as a quick start. Specifically, you can provide the preprocessing code at line 63 in quantize_quark.py

.. code-block:: bash

    mkdir calib_images
    wget -O calib_images/daisy.jpg https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/test_images/daisy.jpg?raw=true


Quantization
------------

For object detection models like YOLO, A16W8, BF16, and Excluding nodes often achieve good results.


- **XINT8**

XINT8 uses symmetric INT8 activation and weights quantization with power-of-two scales. Typically, the calibration method uses MinMSE.

.. code-block:: bash

   python quantize_quark.py  --input_model_path models/yolov8n-face.onnx \
                             --calib_data_path calib_images \
                             --output_model_path models/yolov8n-face_quantized.onnx \
                             --config XINT8

- **A8W8**

A8W8 uses symmetric INT8 activation and weights quantization with float scales. Typically, the calibration method uses MinMax.

.. code-block:: bash

   python quantize_quark.py  --input_model_path models/yolov8n-face.onnx \
                             --calib_data_path calib_images \
                             --output_model_path models/yolov8n-face_quantized.onnx \
                             --config A8W8

- **A16W8**

A16W8 uses symmetric INT16 activation and symmetric INT8 weights quantization with float scales. Typically, the calibration method uses MinMax.

.. code-block:: bash

   python quantize_quark.py  --input_model_path models/yolov8n-face.onnx \
                             --calib_data_path calib_images \
                             --output_model_path models/yolov8n-face_quantized.onnx \
                             --config A16W8

- **BF16**

BFLOAT16 (BF16) is a 16-bit floating-point format designed for machine learning. It has the same exponent size as FP32, allowing a wide dynamic range, but with reduced precision to save memory and speed up computations.

.. code-block:: bash

   python quantize_quark.py  --input_model_path models/yolov8n-face.onnx \
                             --calib_data_path calib_images \
                             --output_model_path models/yolov8n-face_quantized.onnx \
                             --config BF16

- **BFP16**

Block Floating Point (BFP) quantization computational complexity by grouping numbers to share a common exponent, preserving accuracy efficiently. BFP has both reduced storage requirements and high quantization precision.

.. code-block:: bash

   python quantize_quark.py  --input_model_path models/yolov8n-face.onnx \
                             --calib_data_path calib_images \
                             --output_model_path models/yolov8n-face_quantized.onnx \
                             --config BFP16

- **CLE**

The CLE (Cross Layer Equalization) algorithm is a quantization technique that balances weights across layers by scaling them proportionally, aiming to reduce accuracy loss and improve robustness in low-bit quantized neural networks. Taking XINT8 as the example:

.. code-block:: bash

   python quantize_quark.py  --input_model_path models/yolov8n-face.onnx \
                             --calib_data_path calib_images \
                             --output_model_path models/yolov8n-face_quantized.onnx \
                             --config XINT8 \
                             --cle

- **ADAROUND**

ADAROUND (Adaptive Rounding) is a quantization algorithm that optimizes the rounding of weights by minimizing the reconstruction error, ensuring better accuracy retention for neural networks in post-training quantization. Taking XINT8 as the example:

Note: ADAROUND does not support BF16 and BFP16 configurations.

.. code-block:: bash

   python quantize_quark.py  --input_model_path models/yolov8n-face.onnx \
                             --calib_data_path calib_images \
                             --output_model_path models/yolov8n-face_quantized.onnx \
                             --config XINT8 \
                             --adaround \
                             --learning_rate 0.1 \
                             --num_iters 3000

- **ADAQUANT**

ADAQUANT (Adaptive Quantization) is a post-training quantization algorithm that optimizes quantization parameters by minimizing layer-wise reconstruction errors, enabling improved accuracy for low-bit quantized neural networks. Taking XINT8 as the example:

.. code-block:: bash

   python quantize_quark.py  --input_model_path models/yolov8n-face.onnx \
                             --calib_data_path calib_images \
                             --output_model_path models/yolov8n-face_quantized.onnx \
                             --config XINT8 \
                             --adaquant \
                             --learning_rate 0.00001 \
                             --num_iters 10000

- **Exclude Nodes**

Excluding some nodes means that these nodes will be quantized. The method can improve quantization accuracy. Taking XINT8 as the example:

.. code-block:: bash

   python quantize_quark.py  --input_model_path models/yolov8n-face.onnx \
                             --calib_data_path calib_images \
                             --output_model_path models/yolov8n-face_quantized.onnx \
                             --config XINT8 \
                             --exclude_nodes "/model.22/Concat_5"

- **Exclude Subgraphs**

Excluding some subgraphs can improve quantization accuracy significantly, especially for excluding postprocess of detection models. Taking XINT8 as the example:

.. code-block:: bash

   python quantize_quark.py  --input_model_path models/yolov8n-face.onnx \
                             --calib_data_path calib_images \
                             --output_model_path models/yolov8n-face_quantized.onnx \
                             --config XINT8 \
                             --exclude_subgraphs "[/model.22/cv3.2/cv3.2.1/conv/Conv, /model.22/cv3.2/cv3.2.1/act/Sigmoid], [/model.22/cv3.2/cv3.2.1/act/Mul]; [/model.22/Slice, /model.22/Slice_1], [/model.22/Sub_1, /model.22/Div_1]"

Inference
---------

The command below shows how to input an image and output an image with detection.

.. code-block:: bash

   mkdir detection_images
   python onnx_evaluate.py --input_model_path models/yolov8n-face.onnx --input_image [INPUT_IMAGE] --output_image [DETECTION_IMAGE]

.. raw:: html

   <!--
   ## License
   Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
