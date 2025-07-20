.. raw:: html

   <!-- omit in toc -->

Quantizing a ResNet50-v1-12 Model
=================================

This folder contains an example of quantizing a `Resnet50-v1-12 image
classification
model <https://github.com/onnx/models/blob/new-models/vision/classification/resnet/model/resnet50-v1-12.onnx>`__
using the ONNX quantizer of Quark. The example has the following parts:

-  `Pip requirements <#pip-requirements>`__
-  `Prepare data and model <#prepare-data-and-model>`__
-  `Model Quantization <#model-quantization>`__
-  `Evaluation <#evaluation>`__

Pip requirements
----------------

Install the necessary python packages:

::

   python -m pip install -r requirements.txt

Prepare data and model
----------------------

ILSVRC 2012, commonly known as 'ImageNet'. This dataset provides access
to ImageNet (ILSVRC) 2012 which is the most commonly used subset of
ImageNet. This dataset spans 1000 object classes and contains 50,000
validation images.

If you already have an ImageNet datasets, you can directly use your
dataset path.

To prepare the test data, please check the download section of the main
website: https://huggingface.co/datasets/imagenet-1k/tree/main/data. You
need to register and download **val_images.tar.gz**.

Then, create the validation dataset and calibration dataset:

::

   mkdir val_data && tar -xzf val_images.tar.gz -C val_data
   python prepare_data.py val_data calib_data

The storage format of the val_data of the ImageNet dataset organized as
follows:

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

The storage format of the calib_data of the ImageNet dataset organized
as follows:

-  calib_data

   -  n01440764

      -  ILSVRC2012_val_00000293.JPEG

   -  n01443537

      -  ILSVRC2012_val_00000236.JPEG

   -  …
   -  n15075141

      -  ILSVRC2012_val_00001079.JPEG

Finally, download the onnx float model from onnx/models repo.

::

   wget -P models https://github.com/onnx/models/raw/new-models/vision/classification/resnet/model/resnet50-v1-12.onnx

Model Quantization
------------------

The quantizer takes the float model and produce a quantized model.

::

   python quantize_model.py --input_model_path models/resnet50-v1-12.onnx \
                            --output_model_path models/resnet50-v1-12_quantized.onnx \
                            --calibration_dataset_path calib_data

This command will generate a quantized model under the **models**
folder, which was quantized by XINT8 configuration (Int8 symmetric
quantization using power-of-2 scale).

Evaluation
----------

Test the accuracy of the float model on ImageNet val dataset:

::

   python onnx_validate.py val_data --batch-size 1 --onnx-input models/resnet50-v1-12.onnx

Test the accuracy of the quantized model on ImageNet val dataset:

::

   python onnx_validate.py val_data --batch-size 1 --onnx-input models/resnet50-v1-12_quantized.onnx

+----------+----------------------------+------------------------------+
|          | Float Model                | Quantized Model              |
+==========+============================+==============================+
| Model    | 97.82 MB                   | 25.62 MB                     |
| Size     |                            |                              |
+----------+----------------------------+------------------------------+
| Prec@1   | 74.114 %                   | 73.444 %                     |
+----------+----------------------------+------------------------------+
| Prec@5   | 91.716 %                   | 91.274 %                     |
+----------+----------------------------+------------------------------+

.. raw:: html

   <!--
   ## License
   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
