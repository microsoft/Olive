Quark ONNX Quantization Example
===============================

This folder contains an example of quantizing a
mobilenetv2_050.lamb_in1k model using the ONNX quantizer of Quark.
Per-tensor quantization performs poorly on the model, but ADAROUND can
significantly mitigate the quantization loss. The example has the
following parts:

-  `Pip requirements <#pip-requirements>`__
-  `Prepare model <#prepare-model>`__
-  `Prepare data <#prepare-data>`__
-  `Quantization without ADAROUND <#quantization-without-adaround>`__
-  `Quantization with ADAROUND <#quantization-with-adaround>`__
-  `Evaluation <#evaluation>`__

Pip Requirements
^^^^^^^^^^^^^^^^^

Install the necessary Python packages:

::

   python -m pip install -r ../utils/requirements.txt

Prepare Model
^^^^^^^^^^^^^

Export ONNX model from mobilenetv2_050.lamb_in1k torch model. The corresponding model link is https://huggingface.co/timm/mobilenetv2_050.lamb_in1k:

::

   mkdir models && python ../utils/export_onnx.py mobilenetv2_050.lamb_in1k

Prepare Data
^^^^^^^^^^^^

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

Quantization without ADAROUND
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The quantizer takes the float model and produces a quantized model without ADAROUND.

::

   python quantize_model.py --model_name mobilenetv2_050.lamb_in1k \
                            --input_model_path models/mobilenetv2_050.lamb_in1k.onnx \
                            --output_model_path models/mobilenetv2_050.lamb_in1k_quantized.onnx \
                            --calibration_dataset_path calib_data \
                            --config S8S8_AAWS

This command generates a quantized model under the **models** folder, which was quantized by the S8S8_AAWS configuration (Int8 symmetric quantization) without ADAROUND.

Quantization with ADAROUND
^^^^^^^^^^^^^^^^^^^^^^^^^^

The quantizer takes the float model and produces a quantized model with ADAROUND.

::

   python quantize_model.py --model_name mobilenetv2_050.lamb_in1k \
                            --input_model_path models/mobilenetv2_050.lamb_in1k.onnx \
                            --output_model_path models/mobilenetv2_050.lamb_in1k_adaround_quantized.onnx \
                            --calibration_dataset_path calib_data \
                            --config S8S8_AAWS_ADAROUND

This command generates a quantized model under the **models** folder, which was quantized by the S8S8_AAWS configuration (Int8 symmetric quantization) with ADAROUND.

Evaluation
^^^^^^^^^^

Test the accuracy of the float model on the ImageNet val dataset:

::

   python ../utils/onnx_validate.py val_data --model-name mobilenetv2_050.lamb_in1k --batch-size 1 --onnx-input models/mobilenetv2_050.lamb_in1k.onnx

Test the accuracy of the quantized model without ADAROUND on the ImageNet val dataset:

::

   python ../utils/onnx_validate.py val_data --model-name mobilenetv2_050.lamb_in1k --batch-size 1 --onnx-input models/mobilenetv2_050.lamb_in1k_quantized.onnx

Test the accuracy of the quantized model with ADAROUND on the ImageNet val dataset:

::

   python ../utils/onnx_validate.py val_data --model-name mobilenetv2_050.lamb_in1k --batch-size 1 --onnx-input models/mobilenetv2_050.lamb_in1k_adaround_quantized.onnx

.. list-table::
   :header-rows: 1

   * -
     - Float Model
     - Quantized Model without ADAROUND
     - Quantized Model with ADAROUND
   * - Model Size
     - 8.4 MB
     - 2.3 MB
     - 2.4 MB
   * - P rec@1
     - 65.424 %
     - 1.708 %
     - 41.420 %
   * - P rec@5
     - 85.788 %
     - 5.690 %
     - 64.802 %

.. note::

   Different machine models can lead to minor variations in the accuracy of the quantized model with ADAROUND.
