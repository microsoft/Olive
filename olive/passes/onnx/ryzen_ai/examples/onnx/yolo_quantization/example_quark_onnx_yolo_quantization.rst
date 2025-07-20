.. raw:: html

   <!-- omit in toc -->

Yolo_nas and Yolox Quantization
===============================

This example introduces how to quantize a yolo_nas/yolox model with Quark ONNX and evaluate its accuracy.

The Quark ONNX example has the following parts:

-  `Pip requirements <#pip-requirements>`__
-  `Prepare data <#prepare-data>`__
-  `Infer the Float Model (Optional) <#infer-the-float-model-optional>`__
-  `Provide Custom Calibration Data and Evaluation Data (Optional) <#provide-custom-calibration-data-and-evaluation-data-optional>`__
-  `Quantization <#quantization>`__
-  `Results <#results>`__

Pip Requirements
----------------

Install the necessary Python packages (using CPU as default):

::

   bash env.sh

If you want to use GPU, use the following command instead:

.. code:: bash

   bash env.sh
   pip uninstall onnxruntime
   pip uninstall onnxruntime-gpu
   pip install onnxruntime-gpu==1.18.1

.. note::

   The difference between using CPU and GPU lies in whether CUDA acceleration is used during model evaluation.


Prepare Data
------------

To use this Dataset you need to:

- Download coco dataset:
   annotations: http://images.cocodataset.org/annotations/annotations_trainval2017.zip
   train2017: http://images.cocodataset.org/zips/train2017.zip
   val2017: http://images.cocodataset.org/zips/val2017.zip

- Unzip and organize it as below:

-  coco

   -  annotations

      -  instances_train2017.json
      -  instances_val2017.json
      -  …

   -  images

      -  train2017

         -  000000000001.jpg
         -  …

      -  val2017

         - …


- Install CoCo API: https://github.com/pdollar/coco/tree/master/PythonAPI


Infer the Float Model (Optional)
--------------------------------

You can use this command to infer the float model with evaluation data.

.. code:: bash


   python infer_float_yolo_model.py --model_name yolo_nas_s --eval_data_path [EVAL_DATA]


If you want to use GPU, use the following command instead:

.. code:: bash


   python infer_float_yolo_model.py --model_name yolo_nas_s --eval_data_path [EVAL_DATA] --gpu


Quantization
------------

We take yolo_nas_s as an example.

This command first generates a float model in the **models** folder. Next, it prepares the evaluation data **val_data**. Then, the model is quantized using the calibration data (prepared in the first inference process) and the specified configuration (you can specify **XINT8**, **A8W8** or **A16W8**). Finally, the quantized model is evaluated using the evaluation data.

.. code:: bash


   python quantize_yolo.py --model_name yolo_nas_s --calib_data_path [CALIB_DATA_PATH] --eval_data_path [EVAL_DATA] --config XINT8


If you want to use GPU, use the following command instead:

.. code:: bash


   python quantize_yolo.py --model_name yolo_nas_s --calib_data_path [CALIB_DATA_PATH] --eval_data_path [EVAL_DATA] --config XINT8 --gpu


If you customize the calibration data and evaluation data, please use the command below.

.. code:: bash


   python quantize_yolo.py --model_name yolo_nas_s --calib_data_path [CALIB_DATA_PATH] --eval_data_path [CUSTOME_EVAL_DATA] --config XINT8


If you want to use GPU, use the following command instead:

.. code:: bash

   python quantize_yolo.py --model_name yolo_nas_s --calib_data_path [CALIB_DATA_PATH] --eval_data_path [CUSTOME_EVAL_DATA] --config XINT8 --gpu


.. note::

   If using AdaRound or AdaQuant, utilizing a GPU will also accelerate the finetune process.


Results
-------

As seen in the table, generally, the Top-1 accuracy of **A16W8** is higher than that of **A8W8**, which in turn is higher than **XINT8**. Among them, **A16W8** often achieves accuracy close to that of the float model. You can also use some accuracy enhancement methods to improve accuracy, such as specifying the **XINT8_ADAROUND**, **A8W8_ADAROUND** or **A16W8_ADAROUND** configuration.

.. list-table::
   :header-rows: 1

   * -
     - Yolox_s Float Model
     - Quantized with XINT8 config
     - Quantized with A8W8 config
     - Quantized with A16W8 config
   * - Model Size
     - 36 MB
     - 9 MB
     - 9 MB
     - 9 MB
   * - mAP0.5:0.95
     - 40.51 %
     - 29.40 %
     - 30.92 %
     - 36.99 %



.. list-table::
   :header-rows: 1

   * -
     - Yolo_nas_s Float Model
     - Quantized with XINT8 config
     - Quantized with A8W8 config
     - Quantized with A16W8 config
   * - Model Size
     - 46.5 MB
     - 12.0 MB
     - 12.0 MB
     - 12.0 MB
   * - mAP0.5
     - 60.11 %
     - 25.97 %
     - 51.51 %
     - 59.35 %
