Quark ONNX Quantization Example
===============================

This folder contains an example of quantizing a
mobilenetv2_050.lamb_in1k model using the ONNX quantizer of Quark.
Per-tensor quantization performs poorly on the model, but ADAQUANT can
significantly mitigate the quantization loss. The example has the
following parts:

-  `Pip requirements <#pip-requirements>`__
-  `Prepare model <#prepare-model>`__
-  `Prepare data <#prepare-data>`__
-  `Quantization without ADAQUANT <#quantization-without-adaquant>`__
-  `Quantization with ADAQUANT <#quantization-with-adaquant>`__
-  `Evaluation <#evaluation>`__

Pip requirements
----------------

Install the necessary python packages:

::

   python -m pip install -r ../utils/requirements.txt

Prepare model
-------------

Export onnx model from mobilenetv2_050.lamb_in1k torch model. The corresponding model link is https://huggingface.co/timm/mobilenetv2_050.lamb_in1k:

::

   mkdir models && python ../utils/export_onnx.py mobilenetv2_050.lamb_in1k

Prepare data
------------

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
   python ../utils/prepare_data.py val_data calib_data

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

Quantization without ADAQUANT
-----------------------------

The quantizer takes the float model and produce a quantized model
without ADAQUANT.

::

   python quantize_model.py --model_name mobilenetv2_050.lamb_in1k \
                            --input_model_path models/mobilenetv2_050.lamb_in1k.onnx \
                            --output_model_path models/mobilenetv2_050.lamb_in1k_quantized.onnx \
                            --calibration_dataset_path calib_data \
                            --config S8S8_AAWS

This command will generate a quantized model under the **models**
folder, which was quantized by S8S8_AAWS configuration (Int8 symmetric
quantization) without ADAQUANT.

Quantization with ADAQUANT
--------------------------

The quantizer takes the float model and produce a quantized model with
ADAQUANT.

::

   python quantize_model.py --model_name mobilenetv2_050.lamb_in1k \
                            --input_model_path models/mobilenetv2_050.lamb_in1k.onnx \
                            --output_model_path models/mobilenetv2_050.lamb_in1k_adaquant_quantized.onnx \
                            --calibration_dataset_path calib_data \
                            --config S8S8_AAWS_ADAQUANT

This command will generate a quantized model under the **models**
folder, which was quantized by S8S8_AAWS configuration (Int8 symmetric
quantization) with ADAQUANT.

Evaluation
----------

Test the accuracy of the float model on ImageNet val dataset:

::

   python ../utils/onnx_validate.py val_data --model-name mobilenetv2_050.lamb_in1k --batch-size 1 --onnx-input models/mobilenetv2_050.lamb_in1k.onnx

Test the accuracy of the quantized model without ADAQUANT on ImageNet
val dataset:

::

   python ../utils/onnx_validate.py val_data --model-name mobilenetv2_050.lamb_in1k --batch-size 1 --onnx-input models/mobilenetv2_050.lamb_in1k_quantized.onnx

Test the accuracy of the quantized model with ADAQUANT on ImageNet val
dataset:

::

   python ../utils/onnx_validate.py val_data --model-name mobilenetv2_050.lamb_in1k --batch-size 1 --onnx-input models/mobilenetv2_050.lamb_in1k_adaquant_quantized.onnx

+-------+-------------------+---------------------+-------------------+
|       | Float Model       | Quantized Model     | Quantized Model   |
|       |                   | without ADAQUANT    | with ADAQUANT     |
+=======+===================+=====================+===================+
| Model | 8.4 MB            | 2.3 MB              | 2.4 MB            |
| Size  |                   |                     |                   |
+-------+-------------------+---------------------+-------------------+
| P     | 65.424 %          | 1.708 %             | 52.322 %          |
| rec@1 |                   |                     |                   |
+-------+-------------------+---------------------+-------------------+
| P     | 85.788 %          | 5.690 %             | 75.756 %          |
| rec@5 |                   |                     |                   |
+-------+-------------------+---------------------+-------------------+

Note: Different machine models can lead to minor variations in the
accuracy of quantized model with adaquant.
