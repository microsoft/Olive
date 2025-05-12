# How To Configure Model

This document describes how to configure the different types of Models in Olive.

## Model Types
### HfModel
Hf Model is a model type that is used to load HuggingFace transformers model.
```json
{
    "type": "HfModel",
    "model_path": "model_name_or_path",
    "task": "text-generation-with-past",
    "adapter_path": "adapter_path",
    "load_kwargs": {
        "attention_implementation": "eager",
        "torch_dtype": "float16"
    }
}
```
- `model_path` can be a local path or a string name (such as a model hub id). Olive will automatically infer if it is a string name or local file.
- `task` is a string that specifies the task type. The default task is `text-generation-with-past` which is equivalent to a causal language model with key-value cache enabled.
- `adapter_path` is an optional field that is used to load HuggingFace Peft adapters, such as LoRA, to the base model. It can be a local path or a string name (such as a model hub id).
- `load_kwargs` is an optional field that is used to pass additional arguments to the `from_pretrained` methods used to load the model, config, and tokenizer.

If optimum is installed, Olive will use it to automatically obtain the model's input/output config and dummy inputs for conversion to ONNX. Else, the model's io_config must be provided. Refer to [options](../../reference/options.md#input-model-information) for more details.

### PytorchModel
Pytorch Model is a model type that is used to load PyTorch models. It can be used to load any PyTorch model.

#### Entire Model
If the `model_path` is a file with the full model (e.g. `model.pt`) and can be loaded directly using the `torch.load` method:
```json
{
    "type": "PytorchModel",
    "model_path": "model_path"
}
```

#### Custom Model
If the `model_path` is a file/folder/string with model artifaces (e.g. `model_dir`) and requires a custom loading function:
```json
{
    "type": "PytorchModel",
    "model_path": "model_dir",
    "model_script": "load_model.py",
    "model_loader": "load_model",
}
```
- `model_script` is the path to the script that contains the custom loading function.
- `model_loader` is the name of the function that loads the model. It should take the `model_path` as an argument and return the loaded PyTorch model.

#### IO Config and Dummy Inputs
- `io_config` is required in the model configuration for conversion to ONNX. It can be a dict similar to the one used in `HfModel` or a function defined in the `model_script`. The function should take the model handler as an argument and return the io_config.
- If the `io_config` has `input_shapes` and optionally `input_types`, it can be used to create dummy inputs for the model. Alternatively, the `dummy_inputs_func` field can be used to specify a function that generates dummy inputs. The function should take the model handler as an argument and return the dummy inputs.

```json
{
    "type": "PytorchModel",
    "model_path": "model_dir",
    "model_script": "load_model.py",
    "model_loader": "load_model",
    "io_config": {
        "input_names": ["input_ids", "attention_mask"],
        "output_names": ["logits"]
    },
    "dummy_inputs_func": "generate_dummy_inputs"
}
```

### ONNX Model
ONNX Model is a model type that is used to load ONNX models.

#### No external data
If there is just the ONNX model file and no external data files:

```json
{
    "type": "ONNXModel",
    "model_path": "path/to/model.onnx"
}
```

#### External data
If the model has external data file(s) in the same directory as the ONNX model file, specify the parent directory as the `model_path` and the ONNX model file name as `onnx_file_name`:

```json
{
    "type": "ONNXModel",
    "model_path": "path/to/parent_dir",
    "onnx_file_name": "model.onnx"
}
```

### OpenVINO Model
OpenVINO Model is a model type that is used to load OpenVINO models.
```json
{
    "type": "OpenVINOModel",
    "model_path": "model_dir"
}
```
- `model_path` is the path to the OpenVINO model directory. The directory should contain the `model.xml` and `model.bin` files.

### QNN Model
QNN Model is a model type that is used to load QNN models.
```json
{
    "type": "QNNModel",
    "model_file_format": "QNN.CPP",
    "model_path": "model_dir"
}
```
- `model_file_format` is the format of the QNN model file. It can be `QNN.CPP`, `QNN.LIB`, or `QNN.SERIALIZED.BIN`.
- `model_path` is the path to the model file for `QNN.CPP` and `QNN.SERIALIZED.BIN` formats, or the path to the model directory for `QNN.LIB` format.

### SNPE Model
SNPE Model is a model type that is used to load SNPE models.
```json
{
    "type": "SNPEModel",
    "model_path": "model.dlc"
}
```
- `model_path` is the path to the SNPE DLC file.

### TensorFlow Model
TensorFlow Model is a model type that is used to load TensorFlow models.
```json
{
    "type": "TensorFlowModel",
    "model_path": "model_dir"
}
```
- `model_path` is the path to the TensorFlow model directory.

### Composite Model
Composite Model is a model type that is used to load composite models. Composite models are models that are composed of multiple sub-models.

```json
{
    "type": "CompositeModel",
    "model_component_names": ["model1", "model2"],
    "model_components": [
        {
            "type": "ONNXModel",
            "model_path": "path/to/model1.onnx"
        },
        {
            "type": "ONNXModel",
            "model_path": "path/to/model2.onnx"
        }
    ]
}
```


## Configuring Model Path
The model path is specified in the model config under the ``model_path`` key. The model path can be a local file, a local folder, or a remote resource.

### Local Model Path
If the model path is a local path or a string name (such as a model hub id), it can be directly specified as a string. Olive will automatically
infer if it is a string name, local file or local directory.

```json
{
    "model_path": "my_model.pt"
}
```

You can also specify the resource type explicitly.

#### Local File
```json
{
    "model_path": {
        "type": "file",
        "path": "my_model.pt"
    }
}
```

#### Local Folder
```json
{
    "model_path": {
        "type": "folder",
        "path": "my_model_dir"
    }
}
```

#### String Name
```json
{
    "model_path": {
        "type": "string_name",
        "name": "my_model"
    }
}
```

### Remote Model Path
Olive supports remote model resources. Currently, it supports AzureML model, AzureML datastore and AzureML job output.

#### AzureML Model
Models registered in an Azure Machine Learning workspace.

```json
{
    "model_path": {
        "type": "azureml_model",
        "azureml_client": {
            "subscription_id": "my_subscription_id",
            "resource_group": "my_resource_group",
            "workspace_name": "my_workspace"
        },
        "name": "my_model",
        "version": 1
    }
}
```

#### AzureML Registry Model
Models curated in an Azure Machine Learning or models in your own registry. Azure ML curated model doesn't require an ``azureml_client`` config section, but you can still add this section for additional ``mlclient`` configuration.

```json
{
    "model_path": {
        "type": "azureml_registry_model",
        "name": "model_name",
        "registry_name": "registry_name",
        "version": 1
    }
}
```

#### AzureML Datastore
Model files or folders stored in an Azure Machine Learning datastore.

```json
{
    "model_path": {
        "type": "azureml_datastore",
        "azureml_client": {
            "subscription_id": "my_subscription_id",
            "resource_group": "my_resource_group",
            "workspace_name": "my_workspace"
        },
        "datastore_name": "my_datastore",
        "relative_path": "model_dir/my_model.pt" // Relative path to the resource from the datastore root
    }
}
```

#### AzureML Job Output
Model files or folders generated by an Azure Machine Learning job and saved in the job output.

```json
{
    "model_path": {
        "type": "azureml_job_output",
        "azureml_client": {
            "subscription_id": "my_subscription_id",
            "resource_group": "my_resource_group",
            "workspace_name": "my_workspace"
        },
        "job_id": "my_job_id", // id of the job
        "output_name": "my_output_name", // name of the job output
        "relative_path": "model_dir/my_model.pt" // Relative path to the resource from the job output root
    }
}
```

**Note**: If the workflow config file has ``azureml_client`` at the top level, ``azureml_client`` in the model path config can be omitted. The
workflow will automatically use the top level ``azureml_client`` if it is not specified in the model path config.

```json
{
    "azureml_client": {
        "subscription_id": "my_subscription_id",
        "resource_group": "my_resource_group",
        "workspace_name": "my_workspace"
    },
    "input_model": {
        "type": "PytorchModel",
        "model_path": {
            "type": "azureml_model",
            "name": "my_model",
            "version": 1
        }
    }
}
```
