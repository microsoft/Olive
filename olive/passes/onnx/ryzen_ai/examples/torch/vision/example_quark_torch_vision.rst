Vision Model Quantization Using Quark FX Graph Mode
===================================================

What content on this page:

- What is PyTorch Fx graph and advantages.
- Overall brief feature & usage instruction.
- Some experiments Result.


PyTorch Fx Graph & Quark Quantization Tool
------------------------------------------

Advantage about the fx graph
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Unlike ``nn.Module``, the  `torch.fx.GraphModule <https://pytorch.org/docs/stable/fx.html#torch.fx.GraphModule>`_ contains detailed graph information that describes the network forward execution process. In graph, each operation (e.g torch/python function, nn.Module, torch.aten) will be represented as a node and the linking direction represent the computation flow.

Quark Fx quantization tool
^^^^^^^^^^^^^^^^^^^^^^^^^^

In Quark, we take advantage of the ``fx.GraphModule``, Once we get the fully described computation graph, we parse it and do the quantization in the way we demand.

**Utilize Graph information to perform fine-grained quantization.**

- In `eager-mode quantization <https://pytorch.org/docs/stable/quantization.html#eager-mode-quantization>`_ method, that uses traditional ``nn.Module`` as input/output. And do the direct replacement on model’s component (e.g. ``nn.Conv2d`` to ``QuantizedConv2d``). This method can not recognize and quantize the Python inner operation (e.g. ``x = x + 10``), meaning this quantization method can only quant a small part of the model. Seems little possible to deploy on the demand hardware.
- In Quark Fx model quantization, we use the ``torch.fx.GraphModule`` as the inner interpretation. The ``fx.GraphModule`` contain every operation relationship in the computation graph. Quark Fx tool utilize this characteristics to parse the computation graph and insert the Quantizer at the proper place. Meaning the model can be fully quantized. The quantized model are more friendly to AMD NPU etc. device.



Key Feature & Brief Usage instruction.
--------------------------------------

Key Feature
^^^^^^^^^^^

- **AMD hardware friendly**: Compatible with AMD's NPU-related hardware, these devices have runtime and latency requirements. The quantized models can be easily deployed on these devices with low-bit (e.g. INT8) & hardware (e.g. Pow of two quant) requirements. And for accelerate computation.
- **Easy-to-use**: Equal with Quark eager mode quantization tool. Users take their PyTorch ``nn.Module`` as input and the related dataset (e.g. data used for calibration/test/training) and take care less about ``fx.GraphModule``, all quantization will be finished by Quark.
- **PTQ & QAT**: As ``fx.GraphModule`` is Autograd safe, supports the training as typical ``nn.Module``, Quark Fx graph-based quantization tool supports both PTQ (Post Training Quantization) and QAT (Quantization Aware Training).
- **Multi quantization schema**: This tool is mainly used for hardware-related deployment, we also support part/fully quantization, with kinds of quantization schema and Quantizer supported. (e.g. float/Pow-of-two quantization).


Quantization Work Flow
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    '''
    float_model(Python)                                            example_inputs
        \                                                                 /
    —------------------------------------------------------------------------------—
    |          # Step 1. Get the torch.fx.GraphModule (Use PyTorch API)            |
    | exported_model = export_for_training(float_model, example_inputs).module()   |
    —------------------------------------------------------------------------------—
                                          |
                                   FX Graph in ATen
                                          |
    —------------------------------------------------------------------------------—
    |                          Quark Fx Quantization  (Inner Process)              |
    |             # Step 2.  model optimization before quant                       |
    |             # Step 3.  annotate the Graph node to convey quant demand        |
    |             # Step 4.  Insert Quantizer based on annotation information      |
    |             # Step 5.  Use calibration data to Perform PTQ (Optional)        |
    —------------------------------------------------------------------------------—
                                          |
                                   Quantized_Model
                                          |
                         train(Quantized_Model)  QAT (Optional)
                                          |
    —-----------------------—-------------------------------------------------------
    |                        Quark Fx Quantization  (Inner Process)                |
    |             # Step 6. Post optimization to align hardware requirements       |
    |             # Step 7. Explicitly call function to export to ONNX model       |
    —-------------------------------------------------------------------------------
                                           |
                          Compile & Deploy to AMD NPU device
    '''

Some Key Tech Feature
^^^^^^^^^^^^^^^^^^^^^

- **Quantization is realized by the Fakequantize**. In the forward pass, the tensor will be fake quantized in the QDQ manner, known as QDQ (Quantize-DeQuantize) model.
- **Observer**: Typically, each Fakequantizer contains an observer, which is used to record the FP32 tensor value and use specific algorithms to compute the quantization parameter (e.g. scale, zero point). The scale and zero point, quant min, and quant max are used for quantization. In Quark Fx tool, two additoon types of observers are supported in QAT.
  - **`LSQ <https://arxiv.org/abs/1902.08153>`_** adapts a float format scale that adjusts the scale during training.
  - **`TQT <https://arxiv.org/abs/1903.08066>`_** uses a pow-of-2 format scale and will adjust the scale during the training loss, which is more friendly for hardware deployment.


Brief using instruction
^^^^^^^^^^^^^^^^^^^^^^^

In this section, we give an overall method of using the Quark Fx quantization tool.

1. Prepare the PyTorch model and the related dataset.

   .. code-block:: python

      import torch
      class SimpleConv(torch.nn.Module):
          def __init__(self) -> None:
              self.conv = torch.nn.Conv2d(3, 16, 3, padding=1)
              ...
              self.relu = torch.nn.ReLU()
          def forward(self, x: torch.Tensor) -> torch.Tensor:
              a = self.conv(x)
              ...
              return self.relu(a)

      model = SimpleConv()
      # Assume the model is pre-trained in FP32 format
      model.load_state_dict(torch.load(PRE_TRAINED_WEIGHT))
      calib_loader # data user for calibration (PTQ)
      train_loader, val_loader # data user for train (QAT)

2. Prepare fx model and specify the desired quantization schema.

   .. code-block:: python

      # Prepare the fx model using PyTorch API
      example_inputs = (torch.rand(1, 3, 224, 224),)
      graph_model = torch.export.export_for_training(model.eval(), example_inputs).module()

      # Prepare the Quantization config to convey the quant demand
      # More details can be found in the example codes
      from quark.torch import ModelQuantizer, ...
      INT8_PER_TENSOR = QuantizationSpec(dtype=Dtype.int8, qscheme=QSchemeType.per_tensor,
          observer_cls=PerTensorMinMaxObserver, symmetric=True,scale_type=ScaleType.float,
          round_method=RoundType.half_even, is_dynamic=False)
      quant_config = QuantizationConfig( weight=INT8_PER_TENSOR,input_tensors=INT8_PER_TENSOR,
           output_tensors=INT8_PER_TENSOR, bias=INT8_PER_TENSOR)
      quant_config = Config(global_quant_config=quant_config, quant_mode=QuantizationMode.fx_graph_mode)

3. Perform quantization (PTQ/QAT)

   .. code-block:: python

      quantizer = ModelQuantizer(quant_config)
      # PTQ: use calib_loader to perform PTQ.
      quantized_model = quantizer.quantize_model(graph_model, calib_loader)
      # NOTE: if calib_loader is empty (e.g []), will not perform PTQ.
      # QAT: User can train the model just as the traditional PyTorch model
      train(quantized_model, train_loader)

4. Verify the accuracy, Export to onnx, and Deploy to AMD hardware. (Optional)

   Call ``quantizer.freeze`` will perform some AMD hardware specific optimization, which is used for better board deployment.

   .. code-block:: python

      validate(val_loader, quantized_model) # use the quantized_model to validate the accuracy
      from quark.torch import ModelExporter
      from quark.torch.export.config.config import ExporterConfig, JsonExporterConfig
      # Export to ONNX model
      freezeded_model = quantizer.freeze(quantized_model.eval())
      config = ExporterConfig(json_export_config=JsonExporterConfig())
      exporter = ModelExporter(config=config, export_dir=args.export_dir)
      # NOTE: using batch size 1 for better hardware deploy compile
      example_inputs = (torch.rand(1, 3, 224, 224),)
      exporter.export_onnx_model(freezed_model, example_inputs[0])

.. note::
   The above gives a brief workflow about the Quark Fx-Graph quantization, code can not be directly run.
   The runnable code can be found in the example folder.


Experiments:
------------

As this is a long-term project, more and detailed experiments and Python script will be added.

Image Classification Task
^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Model Name
     - Method
     - Result (Acc@1/Acc@5)
   * - ResNet-18
     - Original Float model
     - 69.76 / 89.08
   * - (Torchvision)
     - QAT: NON Overflow, pow-of-2 scale
     - 69.69 / 89.01
   * -
     - PTQ: float scale
     - 69.08 / 88.65
   * - MobileNet-V2
     - Original Float model
     - 71.87 / 90.29
   * - (Torchvision)
     - QAT: TQT, pow-of-2 scale
     - 71.49 / 90.09
   * -
     - PTQ: float scale
     - 65.74 / 86.62

Object Detection Task
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Model Name
     - Method
     - Result (mAP @0.50:0.95)
   * - `YOLO-NAS <https://github.com/Deci-AI/super-gradients>`_
     - Original Float model
     - 0.4759
   * -
     - PTQ: NON flow quantizer
     - 0.3244
   * -
     - QAT: NON flow quantizer
     - 0.3416

As we quantize the entire model (containing the detection head), the training model is different from the eval model. Obtaining a high quantization accuracy is relatively hard in the YOLO-NAS model.

.. note::
   The detailed materials (e.g. PTQ/QAT code, training parameters, etc.) can be found in the corresponding example folder.


Detailed Experiments script
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Below we share a list of recipies that about the vision task.

.. toctree::
   :maxdepth: 1

   example_quark_fx_image_classification.rst
   sample_yolo_nas_quant.rst
   sample_yolo_x_tiny_quant.rst
