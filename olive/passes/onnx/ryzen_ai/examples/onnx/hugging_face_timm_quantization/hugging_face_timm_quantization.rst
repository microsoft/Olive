.. raw:: html

   <!-- omit in toc -->

Hugging Face TIMM Quantization
==============================

This example introduces how to quantize a timm model with Quark ONNX and evaluate its accuracy. PyTorch Image Models (TIMM) is a library containing SOTA computer vision models, layers, utilities, optimizers, schedulers, data-loaders, augmentations, and training/evaluation scripts. It comes packaged with >700 pretrained models, and is designed to be flexible and easy to use.

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

   python -m pip install -r requirements.txt

If you want to use GPU, use the following command instead:

.. code:: bash

   python -m pip install -r requirements_gpu.txt

.. note::

   The difference between using CPU and GPU lies in whether CUDA acceleration is used during model evaluation.


Prepare Data
------------

ILSVRC 2012, commonly known as 'ImageNet'. This dataset provides access to ImageNet (ILSVRC) 2012, which is the most commonly used subset of ImageNet. This dataset spans 1000 object classes and contains 50,000 validation images.

If you already have an ImageNet dataset, you can directly use your dataset path.

To prepare the test data, please check the download section of the main website: https://huggingface.co/datasets/imagenet-1k/tree/main/data. You need to register and download **val_images.tar.gz**.

Provide Custom Calibration Data and Evaluation Data (Optional)
--------------------------------------------------------------

You can also customize the calibration data and evaluation data, the storage format of the calibration / evaluation data is organized as follows:

-  eval_data

   -  n01440764

      -  ILSVRC2012_val_00000293.JPEG
      -  ILSVRC2012_val_00002138.JPEG
      -  …

   -  n01443537

      -  ILSVRC2012_val_00000236.JPEG
      -  ILSVRC2012_val_00000262.JPEG
      -  …

   -  …
   -  n15075141

      -  ILSVRC2012_val_00001079.JPEG
      -  ILSVRC2012_val_00002663.JPEG
      -  …

Infer the Float Model (Optional)
--------------------------------

You can use this command to infer the float model with evaluation data.

.. code:: bash

   python infer_float_timm_model.py --model_name mobilenetv2_100.ra_in1k --eval_data_path eval_data

If you want to use GPU, use the following command instead:

.. code:: bash

   python infer_float_timm_model.py --model_name mobilenetv2_100.ra_in1k --eval_data_path eval_data --gpu

Quantization
------------

We take mobilenetv2_100.ra_in1k (https://huggingface.co/timm/mobilenetv2_100.ra_in1k) as example.

This command first generates a float model in the **models** folder. Next, it prepares the calibration data **calib_data** and evaluation data **val_data**. Then, the model is quantized using the calibration data and the specified configuration (you can specify **XINT8**, **A8W8** or **A16W8**). Finally, the quantized model is evaluated using the evaluation data.

::

   python quantize_timm.py --model_name mobilenetv2_100.ra_in1k --data_path val_images.tar.gz --config XINT8

If you want to use GPU, use the following command instead:

.. code:: bash

   python quantize_timm.py --model_name mobilenetv2_100.ra_in1k --data_path val_images.tar.gz --config XINT8 --gpu

If you customize the calibration data and evaluation data, please use the command below. Assume that **calib_100** and **eval_50000** are your calibration and evaluation folders, respectively.

::

   python quantize_timm.py --model_name mobilenetv2_100.ra_in1k --calib_data_path calib_100 --eval_data_path eval_50000 --config XINT8

If you want to use GPU, use the following command instead:

.. code:: bash

   python quantize_timm.py --model_name mobilenetv2_100.ra_in1k --calib_data_path calib_100 --eval_data_path eval_50000 --config XINT8 --gpu

.. note::

   If using AdaRound or AdaQuant, utilizing a GPU will also accelerate the finetune process.


Results
-------

As seen in the table, generally, the Top-1 accuracy of **A16W8** is higher than that of **A8W8**, which in turn is higher than **XINT8**. Among them, **A16W8** often achieves accuracy close to that of the float model. You can also use some accuracy enhancement methods to improve accuracy, such as specifying the **XINT8_ADAROUND**, **A8W8_ADAROUND** or **A16W8_ADAROUND** configuration.

.. list-table::
   :header-rows: 1

   * -
     - Float Model
     - Quantized with XINT8 config
     - Quantized with A8W8 config
     - Quantized with A16W8 config
   * - Model Size
     - 15 MB
     - 4.0 MB
     - 4.0 MB
     - 4.0 MB
   * - Prec@1
     - 72.890 %
     - 66.640 %
     - 70.504 %
     - 70.556 %
   * - Prec@5
     - 90.996 %
     - 87.122 %
     - 89.656 %
     - 89.592 %
