YOLO-X Tiny FX Graph Quantization
=================================

In this example, we present an Object Detection Model Quantization workflow. We used YOLO-X Tiny as a demonstration to illustrate the effectiveness of FX-graph-based QAT and PTQ.

1. We conduct **QAT** (Quantization-Aware Training) experiments and show competitive results compared with **PTQ** (Post-Training Quantization).
2. The finally exported ONNX model can be used for NPU hardware compile and deployment.
3. The detailed code about `YOLO-X Tiny Model <https://github.com/Megvii-BaseDetection/YOLOX>`_ can be found in Megvii Research.

This repo contains the code for the training, evaluation, etc. In this quant example code, we adopt the original repo and the majority of the code to perform the quantization.

Highlight Overview
------------------

- **Quantization schema**: INT8 (quant range [-128, 127]), symmetric, power-of-2 scale (e.g., 1/(2\*\*4)) for weight, bias, activation.
- **Hardware friendly**: Step-by-step instructions to deploy in the AMD NPU.
- **Satisfied Quant results**: For the original FP32 model, the detection results get the 32.8mAP on COCO val dataset. Using the Quark FX quant tool, the PTQ model gets 25.2 mAP. After QAT(training), the final quantized model gets 30.3 30.3 mAP. This means that even after int8 and pow-of-2 format scale quantization, the quantized model can recover over 92% of the original floating-point model.


Important Information
---------------------

YOLO-X Tiny is an object detection model in computer vision tasks. Developed by Megvii. The original GitHub repo can be found here `YOLOX <https://github.com/Megvii-BaseDetection/YOLOX>`_. We use code from this repo to perform the quantization and only keep the demand code.

**Modify the YOLO-X model code** As we adopt the official PyTorch API to trace the orthodox PyTorch code to get the `torch.fx.GraphModule` format computation graph. We need to modify the original model code. As:
- In original repo: 1. In the YOLO-X `forward` process, both the `loss computation code <https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/yolox.py>`_ and the final `bounding-boxes decoding <https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/yolo_head.py>`_ code are included. However, for quantization, we only need to quantize the model itself; we should not quantize the loss computation process.
- As a result, we modify the code and let the neural network body as `base_model `, in `base_model` model, contain no loss computation and bounding-boxes. We only need to trace the `base_model` to get the `fx` model to insert quantizers to perform the quantization. Meanwhile, not influence the training procedure. After the modification, the modified code was saved in `yolo-x_tiny/models/`. The user can compare the code to find the difference.

**For better & easier quantization**, we only use one GPU to perform the quantization, which reduces a lot of complexity in the code. We have cleaned up the code and reduced the amount of code a lot. We have cleaned up the code and reduced the amount of code a lot.

**Quantization scope:** In Yolo-X, the model mainly contains two parts, the model body and the detection head. In the detection head, there are several constant tensors used for the final bounding box decode. In this example, we quantized the YOLO-X model body. All weight, bias, and activation tensors are quantized.. The detection head part is not quantized, meaning it keeps the FP32 computation. The following image shows that the detection head is not quantized.


Preparation & Workflow
----------------------

1. Prepare the `COCO Dataset <https://cocodataset.org/#download>`_ 2017 Dataset:
   - User can follow the instructions of the `YOLO-X <https://github.com/Megvii-BaseDetection/YOLOX/tree/main>`_ repo.

2. Download pretrained weights: `YOLO-X Tiny Weight <https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth>`_

3. As we have prepared the runnable code,  the following code block is just for workflow demonstration.

   1. Prepare the Quantization config:

      .. code-block:: python

         # NOTE Weight, bias, output and input set to int8, per-tensor, pow-of-2, symmetric quantization, which is more friendly for AMD NPU hardward.
         INT8_PER_WEIGHT_TENSOR_SPEC = QuantizationSpec(dtype=Dtype.int8,
                                       qscheme=QSchemeType.per_tensor,
                                       observer_cls=PerTensorPowOf2MinMSEObserver,
                                       symmetric=True,
                                       scale_type=ScaleType.float,
                                       round_method=RoundType.half_even,
                                       is_dynamic=False)
         quant_config = QuantizationConfig(weight=INT8_PER_WEIGHT_TENSOR_SPEC,
                                             input_tensors=INT8_PER_WEIGHT_TENSOR_SPEC,
                                             output_tensors=INT8_PER_WEIGHT_TENSOR_SPEC,
                                             bias=INT8_PER_WEIGHT_TENSOR_SPEC)

   2. Trace the PyTorch code and prepare the Fx graph model.

      .. code-block:: python

         model = exp.get_model()  # FP32 yolo-x tiny model
         dummy_input = torch.randn(1, 3, 416, 416)
         graph_model = torch.export.export_for_training(model.base_model, (dummy_input, )).module()
         graph_model = torch.fx.GraphModule(graph_model, graph_model.graph)
         model.base_model = graph_model # Replace the base_model to fx traced model

   3. Perform the PTQ (using calibration data).

      .. code-block:: python

         quant_config = Config(global_quant_config=quant_config, quant_mode=QuantizationMode.fx_graph_mode)
         quantizer = ModelQuantizer(quant_config)
         # NOTE. As we use MSEObserver, this is a time & computation-intensive operation, we only using one mini-batch to perform calibation.
         calib_data = [x[0] for x in list(itertools.islice(self.evaluator.dataloader, 1))]
         quantized_model = quantizer.quantize_model(graph_model, calib_data)

   4. Perform the QAT (using training data)

      .. code-block:: python

         # pseudocode
         train_loader = get_data_loader(batch_size)
         lr_scheduler = get_lr_scheduler(basic_lr_per_img * batch_size, max_iter)
         optimizer = get_optimizer(batch_size)
         for epoch in range(start_epoch, max_epoch):
             train_in_iter(quantized_model, train_loader, lr_scheduler, optimizer)

   5. evaluate the quantized model

      .. code-block:: python

         *_, summary = evaluator.evaluate(trainer.model)
         ''' the summary may as follows
         Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.**
         Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.**
         Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.**
         Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.**
         Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.**
         Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.**
         '''
   6. Export to ONNX model (User for compile and deployed on AMD NPU)

      .. code-block:: python

         # Freeze model and do post-quant optimization to meet hardware(NPU) compile requirements.
         freezeded_model = self.quantizer.freeze(self.model.base_model.eval())
         self.model.base_model = freezeded_model
         config = ExporterConfig(json_export_config=JsonExporterConfig())
         exporter = ModelExporter(config=config, export_dir=self.file_name)
         # NOTE for NPU compile, it is better using batch-size = 1 for better compliance
         example_inputs = (torch.rand(1, 3, 416, 416).to(self.device), )
         exporter.export_onnx_model(self.model, example_inputs[0])
         # For better visualization, user can use simplify tool
         from onnxsim import simplify
         quant_model = onnx.load("./***/quark_model.onnx")
         model_simp, check = simplify(quant_model)
         onnx.save_model(model_simp, "./sample_quark_model.onnx")


Quick Start
-----------

.. code-block:: shell

   python PTQ_QAT_exp.py -c=.{PRE_TRAINED_PATH}/yolox_tiny.pth --data_dir=/{DATA_PATH}/COCO/images


In addition, we also supply a `Jupyter` notebook file for better demonstration.


Quantization Results
--------------------

The results is get under the image resolution under 416 * 416. In addition, the hyperparameter such as `nmsthre` and `test_conf` will also influence the test results. We use the default of the YOLO-X  repo.

+---------------+-----------------+--------------+
| Model format  | mAP 0.50:0.95   | mAP 0.50     |
+===============+=================+==============+
| FP32          | 32.6            | 50.0         |
+---------------+-----------------+--------------+
| PTQ int8      | 25.5 (78.2%)    | 43.0 (86.0%) |
+---------------+-----------------+--------------+
| QAT int8      | 30.3 (92.9%)    | 48.3 (96.6%) |
+---------------+-----------------+--------------+
