# Experts distributor for MoE models (using ORT MoE implementation)
This folder contains a sample use case of Olive to distribute experts across multiple onnx graphs for use with distributed inferencing with MoE models.

## Prerequisites
### Clone the repository and install Olive

Refer to the instructions in the [examples README](../README.md) to clone the repository and install Olive.

**Important:** To run the example, you would need a custom onnxruntime build with support for cuda, nccl & mpi.

### Pip requirements
Install the necessary python packages:
```
python -m pip install -r requirements.txt
```

## Run the config to distribute the model
```bash
python -m olive.workflows.run --config examples/switch/config.json --setup
```

## Test the generated models
```bash
python examples/switch/inference.py --filename-pattern {base_model_name}{{:02d}}.onnx --world-size {gpu count}

```

## To compare the results of non-distributed model to the distributed one
```bash
python examples/switch/inference.py --filepath {base_model_name}.onnx --filename-pattern {base_model_name}_{{:02d}}.onnx --world-size {gpu count} --compare

```
