.. raw:: html

   <!-- omit in toc -->

Quark ONNX Example for LayerWisePercentile
==========================================

This folder contains an example of quantizing a vit_small_patch16_224 model using
the ONNX quantizer of Quark. The example has the following parts:

-  `Pip requirements <#pip-requirements>`__
-  `Prepare model <#prepare-model>`__
-  `Prepare data <#prepare-data>`__
-  `Quantization with MinMax and Percentile <#quantization-with-minmax-and-percentile>`__
-  `Quantization with LayerWisePercentile <#quantization-with-layerwisepercentile>`__
-  `Evaluation <#evaluation>`__

Pip Requirements
----------------

Install the necessary Python packages:

::

   python -m pip install -r ../utils/requirements.txt

Prepare Model
-------------

Export ONNX model from vit_small_patch16_224 torch model. The corresponding model link is https://huggingface.co/timm/vit_small_patch16_224.augreg_in1k:

::

   mkdir models && python ../utils/export_onnx.py vit_small_patch16_224.augreg_in1k

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

Quantization With MinMax and Percentile
---------------------------------------

The quantizer takes the float model and produces a quantized model with MinMax and Percentile.

::

   python quantize_model.py --model_name vit_small_patch16_224.augreg_in1k \
                            --input_model_path models/vit_small_patch16_224.augreg_in1k.onnx \
                            --output_model_path models/vit_small_patch16_224.augreg_in1k_minmax_quantized.onnx \
                            --calib_method minmax \
                            --calibration_dataset_path calib_data


::

   python quantize_model.py --model_name vit_small_patch16_224.augreg_in1k \
                            --input_model_path models/vit_small_patch16_224.augreg_in1k.onnx \
                            --output_model_path models/vit_small_patch16_224.augreg_in1k_percentile_quantized.onnx \
                            --calib_method percentile \
                            --calibration_dataset_path calib_data

This command will generate a quantized model under the **models** folder, which was quantized by the S8S8_AAWS configuration (Int8 symmetric quantization) without Layerwise Percentile.

Quantization With LayerWisePercentile
-------------------------------------

The quantizer takes the float model and produces a quantized model with Layerwise Percentile.

::

   python quantize_model.py --model_name vit_small_patch16_224.augreg_in1k \
                            --input_model_path models/vit_small_patch16_224.augreg_in1k.onnx \
                            --output_model_path models/vit_small_patch16_224.augreg_in1k_lwp_quantized.onnx \
                            --calib_method layerwise_percentile \
                            --calibration_dataset_path calib_data

This command will generate a quantized model under the **models** folder, which was quantized by the S8S8_AAWS configuration (Int8 symmetric quantization) with Layerwise Percentile.

Evaluation
----------

Test the accuracy of the float model on the ImageNet val dataset:

::

   python ../utils/onnx_validate.py val_data --model-name vit_small_patch16_224.augreg_in1k --batch-size 1 --onnx-input models/vit_small_patch16_224.augreg_in1k.onnx

Test the accuracy of the quantized model with Minmax on the ImageNet val dataset:

::

   python ../utils/onnx_validate.py val_data --model-name vit_small_patch16_224.augreg_in1k --batch-size 1 --onnx-input models/vit_small_patch16_224.augreg_in1k_minmax_quantized.onnx

Test the accuracy of the quantized model with Percentile on the ImageNet val dataset:

::

   python ../utils/onnx_validate.py val_data --model-name vit_small_patch16_224.augreg_in1k --batch-size 1 --onnx-input models/vit_small_patch16_224.augreg_in1k_percentile_quantized.onnx

Test the accuracy of the quantized model with Layerwise Percentile on the ImageNet val dataset:

::

   python ../utils/onnx_validate.py val_data --model-name vit_small_patch16_224.augreg_in1k --batch-size 1 --onnx-input models/vit_small_patch16_224.augreg_in1k_lwp_quantized.onnx


.. list-table::
   :header-rows: 1

   * -
     - Float Model
     - Quantized with MinMax config
     - Quantized with Percentile config
     - Quantized with LayerWisePercentile config
   * - Model Size
     - 88 MB
     - 23 MB
     - 23 MB
     - 23 MB
   * - Prec@1
     - 74.842 %
     - 45.978 %
     - 55.898 %
     - 71.004 %
   * - Prec@5
     - 92.206 %
     - 69.508 %
     - 78.946 %
     - 89.932 %
