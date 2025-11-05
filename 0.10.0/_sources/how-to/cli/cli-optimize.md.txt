# How To Use `optimize` Command

The `olive optimize` command optimizes a PyTorch/Hugging Face model so that it runs with quality and efficiency on the ONNX Runtime.

## {octicon}`zap` Quickstart

The Olive optimization command (`optimize`) can pull models from Hugging Face, Local disk, or the Azure AI Model Catalog. Following `optimize` command that will download the model, quantize models weights to use int4, convert the model to ONNX and optimize the ONNX graph.

```bash
olive optimize \
    --model_name_or_path microsoft/Phi-3.5-mini-instruct \
    --device cpu \
    --provider CPUExecutionProvider \
    --precision int4 \
```

## Optimize NPU models

You can use `olive optimize` command to optimize a model for NPUs.

```bash
olive optimize \
    --model_name_or_path microsoft/Phi-3.5-mini-instruct \
    --precision int4 \
    --act_precision int8 \
    --provider QNNExecutionProvider \
```

This command will quantize weights into int4 precision before converting the model into ONNX format. The model will be further processed to use int8 precision for activation and use static shapes.

## Customizing model optimization process

`olive optimize` primarily requests desired model precision and intended ExecutionProvider that will be used to run the optimized model. Based on these information, `olive optimize` command will generate model optimiation recipe as per user request and execute the recipe to produce to output model. Advanced users can use `--dry_run` option to save the `config.json` file on the disk. See comprehensive list of [options](../../reference/options.html) you can use to customize the model optimization process further by modifying the `config.json` file produced by the `olive optimize --dry_run ...` command.

## Additional details

See `olive optimize` [reference](../../reference/python_api.md#optimize) for the complete list of supported options by this command.
