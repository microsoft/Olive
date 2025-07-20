Image Classification Models FX Graph Quantization
=================================================

What content on this page:
--------------------------

- Highlight points about the FX mode quantization
- Main API specification
- Some experiments results

Highlight points about the FX mode quantization
-----------------------------------------------

- **Support AMD NPU hardware deployment**. As mentioned above, a fully quantized model can usually be adapted and deployed to a target device for accelerated computation. The Quark Fx quantization tool can not only parse and quantize the whole computation graph; this tool also considers the AMD NPU's hardware property and does the quantization optimization to make the quantized model easier to deploy to AMD NPU.

- **Support both PTQ & QAT**: This tool can support both PTQ and QAT. If PTQ not meet the precision demand, user can use QAT to further improve the model performance. In addition, we also support `TQT <https://arxiv.org/abs/1903.08066>`_ & `LSQ <https://arxiv.org/abs/1902.08153>`_ (two types of learnable Observer) to enrich our QAT library.

- **Support various quantization methods and datatype:** Typically INT8 & pow-of-2 scale is used for hardware computation acceleration. We also support other types of Quantizer with different datatype and scale formats. For more datatype and quantization schema, users can refer to Quark documents.

Main API Declaration
--------------------

For each model quantization, we need to specify the quantization configuration to convey the quantization requirements. Then, instantiate the Quantizer to perform the quantization. (More details can be found in the Starting Quark PyTorch Guide.)

Initiate the **Config** to convey the quantization demand. Users need to specify the following four types of tensors: ``input``, ``output``, ``weight``, and ``bias``.
(Usually, ``output`` & ``input`` are set to equal config.) The following is an example.


.. code-block:: python

    INT8_PER_TENSOR_POW2 = QuantizationSpec(dtype=Dtype.int8, \
                            qscheme=QSchemeType.per_tensor, \
                            observer_cls= PerTensorPowOf2MinMaxObserver, \
                            symmetric=True, scale_type=ScaleType.float,\
                            round_method=RoundType.half_even, is_dynamic=False)
    INT8_PER_TENSOR_FL = QuantizationSpec(dtype=Dtype.int8, \
                            qscheme=QSchemeType.per_tensor, \
                            observer_cls=PerTensorMinMaxObserver, \
                            symmetric=True, scale_type=ScaleType.float,\
                            round_method=RoundType.half_even, is_dynamic=False)
    quant_glb_config = QuantizationConfig(input_tensors=INT8_PER_TENSOR_FL, \
                            output_tensors=INT8_PER_TENSOR_FL, \
                            weight=INT8_PER_TENSOR_POW2, \
                            bias=INT8_PER_TENSOR_POW2)
    quant_config = Config(global_quant_config=quant_glb_config, \
                          quant_mode=QuantizationMode.fx_graph_mode)

Initialize the **quantizer** and perform quantization:

.. code-block:: python

    # We need fx graph as quantizer input
    example_inputs = (torch.rand(1, 3, 224, 224),)
    graph_model = torch.export.export_for_training(model.eval(), example_inputs).module()
    quantizer = ModelQuantizer(quant_config)
    quant_model = quantizer.quantize_model(graph_model, calib_dataloader)  # PTQ is automatically performed.

Initialize the Exporter to export to an ONNX model for hardware deployment (optional):

.. code-block:: python

    config = ExporterConfig(json_export_config=JsonExporterConfig())
    exporter = ModelExporter(config=config, export_dir=args.export_dir)

    # NOTE: For NPU compilation, it is recommended to use batch-size = 1 for better compliance
    example_inputs = (torch.rand(1, 3, 224, 224).to(device), )
    exporter.export_onnx_model(freezeded_model, example_inputs[0])

Experiment Results
------------------

We conducted several experiments on some classic models. The table below summarizes the key results:

.. list-table::
   :header-rows: 1

   * - Model Name
     - Method
     - Result (Acc@1/Acc@5)
   * - ResNet-18 (Torchvision)
     - Original Float Model
     - 69.76 / 89.08
   * - ResNet-18
     - QAT: NON Overflow Observer (Pow-of-2 Scale)
     - 69.69 / 89.01
   * - ResNet-18
     - PTQ: Float Scale
     - 69.08 / 88.65
   * - ResNet-18
     - PTQ: MSE Observer, (Pow-of-2 Scale)
     - 68.90 / 88.65
   * - MobileNet-V2 (Torchvision)
     - Original Float Model
     - 71.87 / 90.29
   * - MobileNet-V2
     - QAT: TQT Observer, (Pow-of-2 Scale)
     - 71.49 / 90.09
   * - MobileNet-V2
     - PTQ: Float Scale
     - 65.74 / 86.62
   * - MobileNet-V2
     - PTQ: MSE Observer, (Pow-of-2 Scale)
     - 61.56 / 83.456

The above experiment scripts can be found at: ``./examples/torch/vision/quantize.py``

For example:

- For ResNet-18, using a NON Overflow quantizer (Zero point = 0, scale: pow-of-2 format, NPU deployable):

  .. code-block:: shell

      python quantize.py --model_name=resnet18 \
                         --pretrained=./resnet18-f37072fd.pth  \
                         --qat \
                         --non_overflow \
                         --data_dir={DATA_PATH_to_Imagenet}

- For MobileNet-V2, using a TQT quantizer (Zero point = 0, scale: pow-of-2 format, NPU deployable):

  .. code-block:: shell

      python quantize.py --model_name=mobilenetv2 \
                         --pretrained=./mobilenet_v2-b0353104.pth  \
                         --qat \
                         --tqt \
                         --data_dir={DATA_PATH_to_Imagenet}

Using the above scripts, users can reproduce the results based on the hyperparameters in the code. Due to time limitations, the hyperparameters in the provided scripts may not yield optimal results. Trying and adjust parameters may result in higher accuracy.

In addition, all int8, symmetric, pow-of-2 scale (quant min: -128 quant max: 127, zero_point: 0, scale: pow-of-2) quantization models have been verified & deployed on AMD NPU devices.

.. note::
   Users can try different quantization settings to achieve better experimental results.
