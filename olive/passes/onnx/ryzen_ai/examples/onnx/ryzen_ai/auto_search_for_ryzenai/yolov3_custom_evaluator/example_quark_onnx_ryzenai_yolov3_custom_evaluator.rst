Auto-Search for Ryzen AI Yolov3 ONNX Model Quantization with Custom Evaluator
=============================================================================

This guide explains how to use the Auto Search framework to perform optimal quantization of an ONNX model on the AMD Ryzen AI platform. The framework automatically searches for the best configuration to balance accuracy, search time by adjusting quantization settings.

Search Config Settings
----------------------

**a. Search Space Building**

The search space defines the range of potential quantization configurations that the Auto Search framework will explore. The goal is to explore different combinations of quantization parameters to find the best trade-off between accuracy, latency and quantization complexity.

- **one example**: You can specify different bit widths (e.g., 8-bit, 4-bit) for weights and activations, calibration algorithms, FastFinetune hyper-parameters, etc.

.. code-block:: python

    search_space: dict[str, any] = {
        "calibrate_method": [CalibrationMethod.MinMax, CalibrationMethod.Percentile],
        "activation_type": [QuantType.QInt8, QuantType.QInt16,],
        "weight_type": [QuantType.QInt8,],
        "include_cle": [False],
        "include_fast_ft": [False],
        "extra_options": {
            'ActivationSymmetric': [False,],
            'WeightSymmetric': [True],
            "CalibMovingAverage": [False, True],
            "CalibMovingAverageConstant": [0.01],
        }
    }


- **multi-space settings**: Combine multi-search spaces into one, where you can control the search order and avoid some bad config combination.

.. code-block:: python

    space1 = auto_search_ins.build_all_configs(auto_search_config.search_space_XINT8)
    space2 = auto_search_ins.build_all_configs(auto_search_config.search_space)
    auto_search_ins.all_configs = space1 + space2


The search space can be configured manually or based on predefined templates.

- **GPU setting**: For taking advantage of GPU resources, we could set GPU to do model optimization and inference. Specifically, set the 'OptimDevice' and 'InferDevice' to be the GPU number in the 'FastFinetune' item. In order to call the GPU smoothly, we need to insall onnxruntime-gpu after uninstalling onnxruntime.

.. code-block:: python

    search_space_with_GPU: dict[str, any] = {
        "calibrate_method": [PowerOfTwoMethod.MinMSE],
        "activation_type": [QuantType.QUInt8,],
        "weight_type": [QuantType.QInt8,],
        "enable_npu_cnn": [True],
        "include_cle": [False, True],
        "include_fast_ft": [True],
        "extra_options": {
            'ActivationSymmetric': [True,],
            'WeightSymmetric': [True],
            'FastFinetune': {
                'OptimDevice': ['cuda:0'],
                'InferDevice': ['cuda:0'],
                }
        }
    }

**b. Search Metric Setting or Evaluator Setting**

The evaluator measures the performance of each quantization configuration. It is essential to set the right metrics to optimize for your deployment needs.

- **Built-in Metric**: Typically, the framework uses the drop in accuracy between the original floating-point model and the quantized model. It is important to keep this drop within an acceptable threshold.The Built-in metrics now support L2, L1, cos, psnr, ssim, which calculate the similarity between the float onnx output and quantized onnx output.
- **Custom Metric in auto_search_config's Evaluator**: Measure the task's metric between the float onnx output and the quantized model output, which may include a post-processing. This configuration is designed for the situation where we need a more concrete metric such as MaP value in YOLOV series models.

There are two ways to define evaluator function:
- defined in auto_search_config as a static method:

.. code-block:: python

    class AutoSearchConfig_Default:
        # 1) define search space
        # 2) define search_metric, search_algo
        # 3) define search_metric_tolerance, search_cache_dir, etc

        @staticmethod
        def custom_evaluator(onnx_path, **args):
            # step 1) build onnx inference session
            # step 2) model post-processing if needed
            # step 3) build evaluation dataloader
            # step 4) calcuate the metric
            # step 5) clean cache if needed
            # step 6) return the metric

        search_evaluator = custom_evaluator

- instance an auto_search_config and assign the evaluator function:

.. code-block:: python

    def custom_evaluator(onnx_path, **args):
            # step 1) build onnx inference session
            # step 2) model post-processing if needed
            # step 3) build evaluation dataloader
            # step 4) calcuate the metric
            # step 5) clean cache if needed
            # step 6) return the metric

    auto_search_conig = AutoSearchConfig_Default()
    auto_search_config.search_evaluator = custom_evaluator


You can specify which metric should be prioritized during the search. For example, if your application demands high accuracy, the evaluator will prioritize configurations that minimize accuracy loss.

**c. Search Tolerance Setting**

The search tolerance is the acceptable margin between the accuracy of the original floating-point model and the quantized model. When the quantized model's accuracy loss exceeds the set tolerance, the Auto Search framework will stop further searches.

- **Tolerance Threshold**: This is a value representing the maximum acceptable accuracy drop from the floating-point model.
- **Auto-Stop Condition**: When the search reaches a configuration with accuracy loss below the tolerance threshold, the framework will halt, saving the best configuration and corresponding quantized model.

Example:
If the floating-point model has 95% accuracy and the tolerance is set to 1%, the Auto Search will stop if a configuration causes an accuracy drop greater than 1% (i.e., below 94%).

Model Quantization Preparation
------------------------------

Before initiating the Auto Search process, ensure that you have the following components ready:

**a. Float ONNX Model**

This is the pre-trained floating-point ONNX model that you intend to quantize.

- **Model File**: model.onnx
  - Ensure the model is trained and exported in the ONNX format. Download the yolov3 model from huggingface url:

::

   https://huggingface.co/amd/yolov3/tree/main

**b. Calibration DataReader**

The calibration data is used during the post-training quantization (PTQ) process to adjust the quantization parameters (e.g., scale and zero-point).

- **Dataset**: Use a dataset that closely represents the input data the model will encounter during inference.
- **DataLoader**: Ensure the calibration data is properly loaded into the framework.

**c. Default Quantization Config**

A default quantization configuration file that defines the starting parameters for the search process. This file may include:
As usual, you can set

.. code-block:: python

    default_config = "S8S8_AAWS"

Call the Auto Search Process
----------------------------

After configuring the search settings, model, and calibration data, you can start the auto search process. Use the following command to trigger the search:

.. code-block:: bash

    python quark_quantize.py --input_model_path [INPUT_MODEL_PATH] --calibration_dataset_path [CALIB_DATA_PATH]

.. raw:: html

   <!--
   ## License
   Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
