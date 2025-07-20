.. raw:: html

   <!-- omit in toc -->

Quark ONNX Example for CrossLayerEqualization (CLE)
===================================================

This folder contains an example of quantizing a resnet152 model using
the ONNX quantizer of Quark. The example has the following parts:

-  `Pip requirements <#pip-requirements>`__
-  `Prepare model <#prepare-model>`__
-  `Prepare data <#prepare-data>`__
-  `Quantization without CLE <#quantization-without-cle>`__
-  `Quantization with CLE <#quantization-with-cle>`__
-  `Evaluation <#evaluation>`__

Pip Requirements
----------------

Install the necessary Python packages:

::

   python -m pip install -r ../utils/requirements.txt

Prepare Model
-------------

Export ONNX model from resnet152 torch model. The corresponding model link is https://huggingface.co/timm/resnet152.a1h_in1k:

::

   mkdir models && python ../utils/export_onnx.py resnet152

Prepare Data
------------

ILSVRC 2012, commonly known as 'ImageNet'. This dataset provides access to ImageNet (ILSVRC) 2012, which is the most commonly used subset of ImageNet. This dataset spans 1000 object classes and contains 50,000 validation images.

If you already have an ImageNet dataset, you can directly use your dataset path.

To prepare the test data, please check the download section of the main website: https://huggingface.co/datasets/imagenet-1k/tree/main/data. You need to register and download **val_images.tar.gz**.

Then, create the validation dataset and calibration dataset:

::

   mkdir val_data && tar -xzf val_images.tar.gz -C val_data
   python ../utils/prepare_data.py val_data calib_data

The storage format of the val_data of the ImageNet dataset is organized as follows:

-  val_data

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

The storage format of the calib_data of the ImageNet dataset is organized as follows:

-  calib_data

   -  n01440764

      -  ILSVRC2012_val_00000293.JPEG

   -  n01443537

      -  ILSVRC2012_val_00000236.JPEG

   -  …
   -  n15075141

      -  ILSVRC2012_val_00001079.JPEG

Quantization Without CLE
------------------------

The quantizer takes the float model and produces a quantized model without CLE.

::

   python quantize_model.py --model_name resnet152 \
                            --input_model_path models/resnet152.onnx \
                            --output_model_path models/resnet152_quantized.onnx \
                            --calibration_dataset_path calib_data

This command will generate a quantized model under the **models** folder, which was quantized by the S8S8_AAWS configuration (Int8 symmetric quantization) without CLE.

Quantization With CLE
---------------------

The quantizer takes the float model and produces a quantized model with CLE.

::

   python quantize_model.py --model_name resnet152 \
                            --input_model_path models/resnet152.onnx \
                            --output_model_path models/resnet152_cle_quantized.onnx \
                            --include_cle \
                            --calibration_dataset_path calib_data

This command will generate a quantized model under the **models** folder, which was quantized by the S8S8_AAWS configuration (Int8 symmetric quantization) with CLE.

Evaluation
----------

Test the accuracy of the float model on the ImageNet val dataset:

::

   python ../utils/onnx_validate.py val_data --model-name resnet152 --batch-size 1 --onnx-input models/resnet152.onnx

Test the accuracy of the quantized model without CLE on the ImageNet val dataset:

::

   python ../utils/onnx_validate.py val_data --model-name resnet152 --batch-size 1 --onnx-input models/resnet152_quantized.onnx

Test the accuracy of the quantized model with CLE on the ImageNet val dataset:

::

   python ../utils/onnx_validate.py val_data --model-name resnet152 --batch-size 1 --onnx-input models/resnet152_cle_quantized.onnx

.. list-table::
   :header-rows: 1

   * -
     - Float Model
     - Quantized Model without CLE
     - Quantized Model with CLE
   * - Model Size
     - 232 MB
     - 59 MB
     - 59 MB
   * - Prec@1
     - 83.456 %
     - 70.042 %
     - 79.664 %
   * - Prec@5
     - 96.580 %
     - 88.502 %
     - 94.854 %
