# Python Interface

Olive provides Python API to transform models. All functions are available directly from the `olive` package.

## `run(...)`

This is the most generic way to run an Olive workflow from a configuration file.

**Arguments:**
- `run_config` (str, Path, or dict): Path to config file or config dictionary.
- `list_required_packages` (bool): List packages required to run a workflow. Defaults to `False`.
- `package_config` (str, Path, dict, optional): Path to optional package config file.
- `tempdir` (str or Path, optional): Root directory for tempfile directories and files.

```python
from olive import run

# Run workflow from a configuration file
workflow_output = run("config.json")
```

The rest of the functions are specialized workflows for common tasks.

## `optimize(...)`

Optimize the input model with comprehensive pass scheduling.

**Arguments:**
- `model_name_or_path` (str): Path to the input model (file path or HuggingFace model name).
- `task` (str, optional): Task for which the huggingface model is used. Default task is text-generation-with-past.
- `trust_remote_code` (bool): Trust remote code when loading a huggingface model. Defaults to `False`.
- `adapter_path` (str, optional): Path to the adapters weights saved after peft fine-tuning. Local folder or huggingface id.
- `model_script` (str, optional): The script file containing the model definition. Required for the local PyTorch model.
- `script_dir` (str, optional): The directory containing the local PyTorch model script file.
- `output_path` (str): Output directory path. Defaults to `"optimized-model"`.
- `provider` (str): Execution provider ("CPUExecutionProvider", "CUDAExecutionProvider", "QNNExecutionProvider", "VitisAIExecutionProvider", "OpenVINOExecutionProvider"). Defaults to `"CPUExecutionProvider"`.
- `device` (str, optional): Target device ("cpu", "gpu", "npu").
- `precision` (str): Target precision ("int4", "int8", "int16", "int32", "uint4", "uint8", "uint16", "uint32", "fp4", "fp8", "fp16", "fp32", "nf4"). Defaults to `"fp32"`.
- `act_precision` (str, optional): Activation precision for quantization.
- `num_split` (int, optional): Number of splits for model splitting.
- `memory` (int, optional): Available device memory in MB.
- `exporter` (str, optional): Exporter to use ("model_builder", "dynamo_exporter", "torchscript_exporter", "optimum_exporter").
- `dim_param` (str, optional): Dynamic parameter names for dynamic to fixed shape conversion.
- `dim_value` (str, optional): Fixed dimension values for dynamic to fixed shape conversion.
- `use_qdq_format` (bool): Use QDQ format for quantization. Defaults to `False`.
- `surgeries` (list[str], optional): List of graph surgeries to apply.
- `block_size` (int, optional): Block size for quantization. Use -1 for per-channel quantization.
- `modality` (str): Model modality ("text"). Defaults to `"text"`.
- `enable_aot` (bool): Enable Ahead-of-Time (AOT) compilation. Defaults to `False`.
- `qnn_env_path` (str, optional): Path to QNN environment directory (required when using AOT with QNN).
- `log_level` (int): Logging level (0-4: DEBUG, INFO, WARNING, ERROR, CRITICAL). Defaults to `3`.
- `save_config_file` (bool): Generate and save the config file for the command. Defaults to `False`.

```python
from olive import optimize

workflow_output = optimize(model_name_or_path="path/to/model")
```

## `quantize(...)`

Quantize a PyTorch or ONNX model.

**Arguments:**
- `model_name_or_path` (str): Path to the input model (file path or HuggingFace model name).
- `task` (str, optional): Task for which the huggingface model is used. Default task is text-generation-with-past.
- `trust_remote_code` (bool): Trust remote code when loading a huggingface model. Defaults to `False`.
- `adapter_path` (str, optional): Path to the adapters weights saved after peft fine-tuning. Local folder or huggingface id.
- `model_script` (str, optional): The script file containing the model definition. Required for the local PyTorch model.
- `script_dir` (str, optional): The directory containing the local PyTorch model script file.
- `output_path` (str): Output directory path. Defaults to `"quantized-model"`.
- `algorithm` (str): Quantization algorithm ("awq", "gptq", "hqq", "rtn", "spinquant", "quarot"). Defaults to `"rtn"`.
- `precision` (str): Quantization precision (int4, int8, int16, int32, uint4, uint8, uint16, uint32, fp4, fp8, fp16, fp32, nf4, PrecisionBits.BITS4, PrecisionBits.BITS8, PrecisionBits.BITS16, PrecisionBits.BITS32). Defaults to `"int8"`.
- `act_precision` (str): Activation precision for static quantization (int4, int8, int16, int32, uint4, uint8, uint16, uint32, fp4, fp8, fp16, fp32, nf4). Defaults to `"int8"`.
- `implementation` (str, optional): Specific implementation of quantization algorithms to use.
- `use_qdq_encoding` (bool): Use QDQ encoding in ONNX model for quantized nodes. Defaults to `False`.
- `data_name` (str, optional): Dataset name for static quantization.
- `subset` (str, optional): The subset of the dataset to use.
- `split` (str, optional): The dataset split to use.
- `data_files` (str, optional): The dataset files (comma-separated if multiple).
- `text_field` (str, optional): The text field to use for fine-tuning.
- `text_template` (str, optional): Template to generate text field from (e.g., '### Question: {prompt} \n### Answer: {response}').
- `use_chat_template` (bool): Use chat template for text field. Defaults to `False`.
- `max_seq_len` (int, optional): Maximum sequence length for the data.
- `add_special_tokens` (bool, optional): Whether to add special tokens during preprocessing.
- `max_samples` (int, optional): Maximum samples to select from the dataset.
- `batch_size` (int, optional): Batch size.
- `input_cols` (list[str], optional): List of input column names.
- `account_name` (str, optional): Azure storage account name for shared cache.
- `container_name` (str, optional): Azure storage container name for shared cache.
- `log_level` (int): Logging level (0-4: DEBUG, INFO, WARNING, ERROR, CRITICAL). Defaults to `3`.
- `save_config_file` (bool): Generate and save the config file for the command. Defaults to `False`.

```python
from olive import quantize

workflow_output = quantize(model_name_or_path="path/to/model")
```

## `capture_onnx_graph(...)`

Capture ONNX graph for a PyTorch model.

**Arguments**:
- `model_name_or_path` (str): Path to the input model (file path or HuggingFace model name).
- `task` (str, optional): Task for which the huggingface model is used. Default task is text-generation-with-past.
- `trust_remote_code` (bool): Trust remote code when loading a huggingface model. Defaults to `False`.
- `adapter_path` (str, optional): Path to the adapters weights saved after peft fine-tuning. Local folder or huggingface id.
- `model_script` (str, optional): The script file containing the model definition. Required for the local PyTorch model.
- `script_dir` (str, optional): The directory containing the local PyTorch model script file.
- `output_path` (str): Output directory path. Defaults to `"captured-model"`.
- `conversion_device` (str): The device used to run the model to capture the ONNX graph ("cpu", "gpu"). Defaults to `"cpu"`.
- `use_ort_genai` (bool): Use OnnxRuntime generate() API to run the model. Defaults to `False`.
- `account_name` (str, optional): Azure storage account name for shared cache.
- `container_name` (str, optional): Azure storage container name for shared cache.
- `log_level` (int): Logging level (0-4: DEBUG, INFO, WARNING, ERROR, CRITICAL). Defaults to `3`.
- `save_config_file` (bool): Generate and save the config file for the command. Defaults to `False`.
- `use_dynamo_exporter` (bool): Use dynamo export API to export ONNX model. Defaults to `False`.
- `fixed_param_dict` (str, optional): Fix dynamic input shapes by providing dimension names and values (e.g., 'batch_size=1,max_length=128').
- `past_key_value_name` (str, optional): The arguments name to point to past key values (used with dynamo exporter).
- `torch_dtype` (str, optional): The dtype to cast the model to before capturing ONNX graph (e.g., 'float32', 'float16').
- `target_opset` (int): The target opset version for the ONNX model. Defaults to `17`.
- `use_model_builder` (bool): Use Model Builder to capture ONNX model. Defaults to `False`.
- `precision` (str): The precision of the ONNX model for Model Builder ("fp16", "fp32", "int4"). Defaults to `"fp32"`.
- `int4_block_size` (int): Block size for int4 quantization (16, 32, 64, 128, 256). Defaults to `32`.
- `int4_accuracy_level` (int, optional): Minimum accuracy level for activation of MatMul in int4 quantization.
- `exclude_embeds` (bool): Remove embedding layer from ONNX model. Defaults to `False`.
- `exclude_lm_head` (bool): Remove language modeling head from ONNX model. Defaults to `False`.
- `enable_cuda_graph` (bool): Enable CUDA graph capture for CUDA execution provider. Defaults to `False`.

```python
from olive import capture_onnx_graph

workflow_output = capture_onnx_graph(model_name_or_path="path/to/model")
```

## `finetune(...)`

Fine-tune a model using LoRA or QLoRA.

**Arguments:**
- `model_name_or_path` (str): Path to the input model (file path or HuggingFace model name).
- `task` (str, optional): Task for which the huggingface model is used. Default task is text-generation-with-past.
- `trust_remote_code` (bool): Trust remote code when loading a huggingface model. Defaults to `False`.
- `output_path` (str): Output directory path. Defaults to `"finetuned-adapter"`.
- `method` (str): Fine-tuning method ("lora", "qlora"). Defaults to `"lora"`.
- `lora_r` (int): LoRA R value. Defaults to `64`.
- `lora_alpha` (int): LoRA alpha value. Defaults to `16`.
- `target_modules` (str, optional): Target modules for LoRA (comma-separated).
- `torch_dtype` (str): PyTorch dtype for training ("bfloat16", "float16", "float32"). Defaults to `"bfloat16"`.
- `data_name` (str): Dataset name (required).
- `train_subset` (str, optional): The subset to use for training.
- `train_split` (str, optional): The split to use for training.
- `eval_subset` (str, optional): The subset to use for evaluation.
- `eval_split` (str, optional): The dataset split to evaluate on.
- `data_files` (str, optional): The dataset files (comma-separated if multiple).
- `text_field` (str, optional): The text field to use for fine-tuning.
- `text_template` (str, optional): Template to generate text field from (e.g., '### Question: {prompt} \n### Answer: {response}').
- `use_chat_template` (bool): Use chat template for text field. Defaults to `False`.
- `max_seq_len` (int, optional): Maximum sequence length for the data.
- `add_special_tokens` (bool, optional): Whether to add special tokens during preprocessing.
- `max_samples` (int, optional): Maximum samples to select from the dataset.
- `batch_size` (int, optional): Batch size.
- `input_cols` (list[str], optional): List of input column names.
- `account_name` (str, optional): Azure storage account name for shared cache.
- `container_name` (str, optional): Azure storage container name for shared cache.
- `log_level` (int): Logging level (0-4: DEBUG, INFO, WARNING, ERROR, CRITICAL). Defaults to `3`.
- `save_config_file` (bool): Generate and save the config file for the command. Defaults to `False`.
- `**training_kwargs`: HuggingFace training arguments.

```python
from olive import finetune

workflow_output = finetune(model_name_or_path="hf_model_name")
```

## `generate_adapter(...)`

Generate adapter for an ONNX model.

**Arguments**:
- `model_name_or_path` (str): Path to the input model (file path or HuggingFace model name).
- `output_path` (str): Output directory path. Defaults to `"generated-adapter"`.
- `adapter_type` (str): Type of adapters to extract ("lora", "dora", "loha"). Defaults to `"lora"`.
- `adapter_format` (str): Format to save weights in ("pt", "numpy", "safetensors", "onnx_adapter"). Defaults to `"onnx_adapter"`.
- `account_name` (str, optional): Azure storage account name for shared cache.
- `container_name` (str, optional): Azure storage container name for shared cache.
- `log_level` (int): Logging level (0-4: DEBUG, INFO, WARNING, ERROR, CRITICAL). Defaults to `3`.
- `save_config_file` (bool): Generate and save the config file for the command. Defaults to `False`.

```python
from olive import generate_adapter

workflow_output = generate_adapter(model_name_or_path="path/to/onnx/model")
```

## `tune_session_params`

Tune ONNX Runtime session parameters for optimal performance.

**Arguments**:
- `model_name_or_path` (str): Path to the input model (file path or HuggingFace model name).
- `output_path` (str): Output directory path. Defaults to `"tuned-params"`.
- `cpu_cores` (int, optional): CPU cores used for thread tuning.
- `io_bind` (bool): Enable IOBinding search for ONNX Runtime inference. Defaults to `False`.
- `enable_cuda_graph` (bool): Enable CUDA graph for CUDA execution provider. Defaults to `False`.
- `execution_mode_list` (list[int], optional): Parallelism list between operators.
- `opt_level_list` (list[int], optional): Optimization level list for ONNX Model.
- `trt_fp16_enable` (bool): Enable TensorRT FP16 mode. Defaults to `False`.
- `intra_thread_num_list` (list[int], optional): List of intra thread number for test.
- `inter_thread_num_list` (list[int], optional): List of inter thread number for test.
- `extra_session_config` (str, optional): Extra customized session options during tuning process (JSON string).
- `disable_force_evaluate_other_eps` (bool): Whether to disable force evaluation of all execution providers. Defaults to `False`.
- `enable_profiling` (bool): Enable profiling for ONNX Runtime inference. Defaults to `False`.
- `predict_with_kv_cache` (bool): Use key-value cache for ORT session parameter tuning. Defaults to `False`.
- `device` (str): Target device ("gpu", "cpu", "npu"). Defaults to `"cpu"`.
- `providers_list` (list[str], optional): List of execution providers to use for ONNX model (case sensitive).
- `memory` (int, optional): Memory limit for the accelerator in bytes.
- `account_name` (str, optional): Azure storage account name for shared cache.
- `container_name` (str, optional): Azure storage container name for shared cache.
- `log_level` (int): Logging level (0-4: DEBUG, INFO, WARNING, ERROR, CRITICAL). Defaults to `3`.
- `save_config_file` (bool): Generate and save the config file for the command. Defaults to `False`.

```python
from olive import tune_session_params

workflow_output = tune_session_params(model_name_or_path="path/to/onnx/model")
```

## Utility Functions

There are also utility functions that don't produce a `WorkflowOutput`:

### `convert_adapters(...)`

Convert lora adapter weights to a file that will be consumed by ONNX models generated by Olive ExtractedAdapters pass.

**Arguments:**
- `adapter_path` (str): Path to the adapters weights saved after peft fine-tuning. Can be a local folder or huggingface id.
- `adapter_format` (str): Format to save weights in ("pt", "numpy", "safetensors", "onnx_adapter"). Defaults to `"onnx_adapter"`.
- `output_path` (str): Path to save the exported weights. Will be saved in the specified format.
- `dtype` (str): Data type to save float adapter weights as ("float32", "float16"). If quantize_int4 is True, this is the data type of the quantization scales. Defaults to `"float32"`.
- `quantize_int4` (bool): Quantize the adapter weights to int4 using blockwise quantization. Defaults to `False`.
- `int4_block_size` (int): Block size for int4 quantization of adapter weights (16, 32, 64, 128, 256). Defaults to `32`.
- `int4_quantization_mode` (str): Quantization mode for int4 quantization ("symmetric", "asymmetric"). Defaults to `"symmetric"`.
- `log_level` (int): Logging level (0-4: DEBUG, INFO, WARNING, ERROR, CRITICAL). Defaults to `3`.

### `extract_adapters(...)`

Extract LoRA adapters from PyTorch model to separate files.

**Arguments:**
- `model_name_or_path` (str): Path to the PyTorch model. Can be a local folder or Hugging Face id.
- `format` (str): Format to save the LoRAs in ("pt", "numpy", "safetensors", "onnx_adapter").
- `output` (str): Output folder to save the LoRAs in the requested format.
- `dtype` (str): Data type to save LoRAs as ("float32", "float16"). Defaults to `"float32"`.
- `cache_dir` (str, optional): Cache dir to store temporary files in. Default is Hugging Face's default cache dir.
- `log_level` (int): Logging level (0-4: DEBUG, INFO, WARNING, ERROR, CRITICAL). Defaults to `3`.

### `generate_cost_model(...)`

Generate a cost model for a given model and save it as a csv file. This cost model is consumed by the CaptureSplitInfo pass. Only supports HfModel.

**Arguments:**
- `model_name_or_path` (str): Path to the input model (file path or HuggingFace model name).
- `task` (str, optional): Task for which the huggingface model is used. Default task is text-generation-with-past.
- `trust_remote_code` (bool): Trust remote code when loading a huggingface model. Defaults to `False`.
- `output_path` (str): Output directory path. Defaults to current directory.
- `weight_precision` (str): Weight precision ("fp32", "fp16", "fp8", "int32", "uint32", "int16", "uint16", "int8", "uint8", "int4", "uint4", "nf4", "fp4"). Defaults to `"fp32"`.

## Accessing API Results

Most API functions return a `WorkflowOutput` object. You can use it to access the optimized models and their metrics.

```python
from olive import run, WorkflowOutput

workflow_output: WorkflowOutput = run("config.json")

# Check if optimization produced any results
if workflow_output.has_output_model():
    # Get the best model overall
    best_model = workflow_output.get_best_candidate()
    print(f"Model path: {best_model.model_path}")
    print(f"Model type: {best_model.model_type}")
    print(f"Device: {best_model.from_device()}")
    print(f"Execution provider: {best_model.from_execution_provider()}")
    print(f"Metrics: {best_model.metrics_value}")

    # Get the best model for CPU
    best_cpu_model = workflow_output.get_best_candidate_by_device("CPU")

    # Get all models for GPU
    gpu_models = workflow_output.get_output_models_by_device("GPU")
```

## Output Class Hierarchy

Olive organizes optimization results in a hierarchical structure:

- `WorkflowOutput`: Top-level container for all results across devices
  - Contains multiple `DeviceOutput` instances (one per device)
    - Each `DeviceOutput` contains multiple `ModelOutput` instances

## Output Classes in Detail

### WorkflowOutput

The `WorkflowOutput` class organizes results from an entire optimization workflow, containing outputs across different hardware devices and execution providers.

```python
from olive import WorkflowOutput
```

#### Key Methods

- `get_input_model_metrics()` - Get the metrics for the input model
- `get_available_devices()` - Get a list of devices that the workflow ran on
- `has_output_model()` - Check if any optimized models are available
- `get_output_models_by_device(device)` - Get all optimized models for a specific device
- `get_output_model_by_id(model_id)` - Get a specific model by its ID
- `get_output_models()` - Get all optimized models sorted by metrics
- `get_best_candidate_by_device(device)` - Get the best model for a specific device
- `get_best_candidate()` - Get the best model across all devices
- `trace_back_run_history(model_id)` - Get the optimization history for a specific model

### DeviceOutput

The `DeviceOutput` class groups model outputs for a specific device, containing results for different execution providers on that device.

```python
from olive import DeviceOutput
```

#### Key Attributes

- `device` - The device type (e.g., "cpu", "gpu")

#### Key Methods

- `has_output_model()` - Check if any model outputs are available for this device
- `get_output_models()` - Get all model outputs for this device
- `get_best_candidate()` - Get the best model output for this device based on metrics
- `get_best_candidate_by_execution_provider(execution_provider)` - Get the best model for a specific execution provider

### ModelOutput

The `ModelOutput` class represents an individual optimized model, containing its path, metrics, and configuration.

```python
from olive import ModelOutput
```

#### Key Attributes

- `metrics` - Dictionary containing the model's performance metrics
- `metrics_value` - Simplified version of metrics with just the values
- `model_path` - Path to the optimized model file
- `model_id` - Unique identifier for this model
- `model_type` - Type of the model (e.g., "onnxmodel")
- `model_config` - Configuration details for the model

#### Key Methods

- `from_device()` - Get the device this model was optimized for
- `from_execution_provider()` - Get the execution provider this model was optimized for
- `from_pass()` - Get the Olive optimization pass that generated this model
- `get_parent_model_id()` - Get the ID of the parent model this was derived from
- `use_ort_extension()` - Check if the model uses the ONNXRuntime extension
- `get_inference_config()` - Get the model's inference configuration
