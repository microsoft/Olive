Diffusion Model Quantization using Quark
========================================

This topic provides examples of FP8 weight-activation quantization and INT8 weight-only quantization using Quark, along with instructions for exporting the quantized models. Supported models include SDXL, SDXL-Turbo, SD1.5, SDXL-Controlnet, and SD1.5-Controlnet. To incorporate additional diffusion models, adjust the pipeline when loading the model, as demonstrated in ``quantize_diffusers.py``.

Third-party Dependencies
------------------------

The example relies on ``torchvision``. Users need to install the version of ``torchvision`` that is compatible with their specific version of PyTorch.

.. code-block:: shell

   export DIFFUSERS_ROOT=$PWD
   git clone https://github.com/mlcommons/inference.git
   cd inference
   git checkout 87ba8cb8a6a4f6525f26255fa513d902b17ab060
   cd ./text_to_image/tools/
   sh ./download-coco-2014.sh --num-workers 5
   sh ./download-coco-2014-calibration.sh -n 5
   cd ${DIFFUSERS_ROOT}
   export PYTHONPATH="${DIFFUSERS_ROOT}/inference/text_to_image/:$PYTHONPATH"

Dataset Files
-------------

- The calibration dataset file will be downloaded to ``${DIFFUSERS_ROOT}/inference/text_to_image/coco2014/calibration/captions.tsv``.
- The test dataset file will be downloaded to ``${DIFFUSERS_ROOT}/inference/text_to_image/coco2014/captions/captions_source.tsv``.

Quantization & Export Scripts
-----------------------------

You can run the following Python scripts in the ``examples/torch/diffusers`` path.

Run Diffusion Model Without Quantization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Run SDXL:

.. code-block:: shell

   python quantize_diffusers.py --model_id stabilityai/stable-diffusion-xl-base-1.0 --skip_quantization

- Run SD1.5 Controlnet:

.. code-block:: shell

   python quantize_diffusers.py --model_id runwayml/stable-diffusion-v1-5 --controlnet_id lllyasviel/control_v11p_sd15_canny --input_image {your input image for guidence in controlnet} --skip_quantization

Calibration and Export
~~~~~~~~~~~~~~~~~~~~~~

Note: For Controlnet, only unet is quantized. For Diffusion Models, entire pipeline (unet, vae, text_encoder and text_encoder_2) can be quantized.

Quantize Controlnet and Export SafeTensors (unet-only)
------------------------------------------------------

.. code-block:: shell

   python quantize_diffusers.py --model_id {your diffusion model} --controlnet_id {your controlnet if used} --input_image {guidence image if controlnet is used} --quant_config_file_path {your quant config file path} --calib_prompts {your calibration dataset file path} --calib_size {number of calibration prompts, default 500} --dump_data_folder {file path to dump data folder} --export safetensor --saved_path {output path for your quantized model}


Quantize Diffusion and Export ONNX (entire pipeline)
----------------------------------------------------

.. code-block:: shell

   python quantize_diffusers.py --model_id {your diffusion model} --quant_config_file_path {your quant config file path} --calib_prompts {your calibration dataset file path} --calib_size {number of calibration prompts, default 500} --dump_data_folder {file path to dump data folder} --export onnx --onnx_path {output path for your exported onnx model}

Load SafeTensor and Test
~~~~~~~~~~~~~~~~~~~~~~~~

Load and Test
-------------

.. code-block:: shell

   python quantize_diffusers.py --model_id {your diffusion model} --controlnet_id {your controlnet if used} --input_image {guidence image if controlnet is used} --load --saved_path {the path for your quantized model} --test --test_prompts {your test dataset file path} --test_size {number of test prompts, default 5000}

Load SafeTensor and Run with a prompt
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Load and Run
------------

.. code-block:: shell

   python quantize_diffusers.py --model_id {your diffusion model} --controlnet_id {your controlnet if used} --input_image {guidence image if controlnet is used} --load --saved_path {the path for your quantized model} --prompt "A city at night with people walking around."

Benchmark
---------

**MI210** GPU, diffusers==0.21.2

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20

   * - Model Name
     - Quant Config
     - CLIP score
     - FID score
   * - SDXL base 1.0
     - FP16
     - 31.74845
     - 23.56758
   * -
     - W-FP8-A-FP8
     - 31.83954
     - 23.61475
   * -
     - W-INT8
     - 31.77445
     - 23.34854
   * - SD 1.5
     - FP16
     - 29.53386
     - 41.85444
   * -
     - W-FP8-A-FP8
     - 29.44639
     - 45.26559
   * -
     - W-INT8
     - 29.53238
     - 42.34745
