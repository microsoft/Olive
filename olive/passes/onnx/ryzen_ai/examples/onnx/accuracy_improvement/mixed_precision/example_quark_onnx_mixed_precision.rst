Quantization using Mixed Precision
==================================

This folder contains an example of quantizing a `densenet121.ra_in1k`
model using the ONNX quantizer of Quark. The example has the following
parts:

-  `Pip requirements <#pip-requirements>`__
-  `Prepare model <#prepare-model>`__
-  `Prepare data <#prepare-data>`__
-  `Quantization without
   Mixed_Precision <#quantization-without-mixed_precision>`__
-  `Quantization
   with_Mixed_Precision <#quantization-with-mixed_precision>`__
-  `Evaluation <#evaluation>`__

Pip requirements
----------------

Install the necessary python packages:

::

   python -m pip install -r ../utils/requirements.txt

Prepare model
-------------

Export ONNX model from `densenet121.ra_in1k` torch model. The corresponding model link is https://huggingface.co/timm/densenet121.ra_in1k:

::

   mkdir models && python ../utils/export_onnx.py densenet121.ra_in1k

Prepare data
------------

ILSVRC 2012, commonly known as 'ImageNet'. This dataset provides access to ImageNet (ILSVRC) 2012 which is the most commonly used subset of ImageNet. This dataset spans 1000 object classes and contains 50,000 validation images.

If you already have an ImageNet dataset, you can directly use your dataset path.

To prepare the test data, please check the download section of the main website: https://huggingface.co/datasets/imagenet-1k/tree/main/data. You need to register and download **val_images.tar.gz**.

Then, create the validation dataset and calibration dataset:

::

   mkdir val_data && tar -xzf val_images.tar.gz -C val_data
   python ../utils/prepare_data.py val_data calib_data

The storage format of the val_data of the ImageNet dataset is organized as follows:

- val_data

  - n01440764

    - ILSVRC2012_val_00000293.JPEG
    - ILSVRC2012_val_00002138.JPEG
    - …

  - n01443537

    - ILSVRC2012_val_00000236.JPEG
    - ILSVRC2012_val_00000262.JPEG
    - …

  - …
  - n15075141

    - ILSVRC2012_val_00001079.JPEG
    - ILSVRC2012_val_00002663.JPEG
    - …

The storage format of the calib_data of the ImageNet dataset is organized as follows:

- calib_data

  - n01440764

    - ILSVRC2012_val_00000293.JPEG

  - n01443537

    - ILSVRC2012_val_00000236.JPEG

  - …
  - n15075141

    - ILSVRC2012_val_00001079.JPEG

Quantization Without Mixed Precision
------------------------------------

The quantizer takes the float model and produces a quantized model without Mixed_Precision.

::

   python quantize_model.py --model_name densenet121.ra_in1k \
                            --input_model_path models/densenet121.ra_in1k.onnx \
                            --output_model_path models/densenet121.ra_in1k_quantized.onnx \
                            --calibration_dataset_path calib_data \
                            --config S8S8_AAWS

This command will generate a quantized model under the **models** folder, which was quantized by S8S8_AAWS configuration (Int8 symmetric quantization) without Mixed_Precision.

Quantization With Mixed Precision
---------------------------------

The quantizer takes the float model and produces a quantized model with Mixed_Precision.

::

   python quantize_model.py --model_name densenet121.ra_in1k \
                            --input_model_path models/densenet121.ra_in1k.onnx \
                            --output_model_path models/densenet121.ra_in1k_mixed_precision_quantized.onnx \
                            --calibration_dataset_path calib_data \
                            --config S16S16_MIXED_S8S8

This command will generate a quantized model under the **models** folder, which was quantized by S16S16_MIXED_S8S8_AAWS configuration (Int16 and Int8 mixed symmetric quantization) with Mixed_Precision.

Evaluation
~~~~~~~~~~

Test the accuracy of the float model on ImageNet val dataset:

::

   python ../utils/onnx_validate.py val_data --model-name densenet121.ra_in1k --batch-size 1 --onnx-input models/densenet121.ra_in1k.onnx

Test the accuracy of the quantized model without Mixed_Precision on ImageNet val dataset:

::

   python ../utils/onnx_validate.py val_data --model-name densenet121.ra_in1k --batch-size 1 --onnx-input models/densenet121.ra_in1k_quantized.onnx

Test the accuracy of the quantized model with Mixed_Precision on ImageNet val dataset:

::

   python ../utils/onnx_validate.py val_data --model-name densenet121.ra_in1k --batch-size 1 --onnx-input models/densenet121.ra_in1k_mixed_precision_quantized.onnx

.. list-table::
   :header-rows: 1

   * -
     - Float Model
     - Quantized Model without Mixed_Precision
     - Quantized Model with Mixed_Precision
   * - Model Size
     - 33 MB
     - 10 MB
     - 17 MB
   * - Prec@1
     - 76.602 %
     - 0.486 %
     - 74.938 %
   * - Prec@5
     - 93.440 %
     - 1.536 %
     - 92.618 %
