# Huggingface Model Optimization

## Introduction
This document outlines the integrations between Olive and Huggingface. Discover how to use Huggingface resources within Olive.

## hf_config
If you want to optimize a Huggingface model, or evaluate a Huggingface model, you will need `hf_config` defined in your `input_model` section. Please refer to [this section](../overview/options.md#input-model-information) for detailed parameters of `hf_config`.

Here is how you can use `hf_config`:

### Model config loading
Olive can automatically retrieve model configurations from Huggingface hub:

- Olive retrieves model [configuration](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoConfig) from transformers for future usage.

- Olive simplifies the process by automatically fetching configurations such as IO config and dummy input required for the `OnnxConversion` pass from [OnnxConfig](https://huggingface.co/docs/transformers/main_classes/onnx#onnx-configurations). This means there's no need for you to manually specify the IO config and dummy input when using the `OnnxConversion` pass.

If you want to use your own `io_config` or `dummy_input`, you can still add them to the model config:
```json
"input_model":{
    "type": "PyTorchModel",
    "config": {
        "model_script": "user_script.py",
        "io_config": "get_io_config",
        "dummy_inputs_func": "get_dummy_inputs",
        "hf_config": {
            "model_name": "meta-llama/Llama-2-7b-hf",
            "task": "text-generation"
        }
    }
}
```

### Model loading
#### Load Huggingface model from Huggingface hub
Olive can automatically retrieve models from Huggingface hub. Here are the examples:

#### PyTorch model
Take `Intel/bert-base-uncased-mrpc` as an example, you can specify task name as `text-classification` to form the `hf_config` as follows:

```json
"input_model":{
    "type": "PyTorchModel",
    "config": {
        "hf_config": {
            "model_name": "Intel/bert-base-uncased-mrpc",
            "task": "text-classification"
        }
    }
}
```

#### Optimum model
Optimum model is a special case of PyTorch model. By specifying `OptimumModel` as `type`, the `model_path` should be the model's name. Then add the names of the model components to `model_components`. Olive will retrieve the components from Huggingface hub:

```json
"input_model":{
    "type": "OptimumModel",
    "config": {
        "model_path": "openlm-research/open_llama_3b",
        "model_components": ["decoder_model.onnx", "decoder_with_past_model.onnx"],
        "hf_config": {
            "model_class": "LlamaForCausalLM"
        }
    }
}
```

### Model loading from local
If you have the Huggingface model prepared in local, add `model_path` to the model config, and specify `model_name` and `task` in `hf_config` so that Olive can automatically fetch the model attributes:

Example:
```json
"input_model":{
    "type": "PyTorchModel",
    "config": {
        "model_path": "path_to_local_model",
        "hf_config": {
            "model_name": "Intel/bert-base-uncased-mrpc",
            "task": "text-classification"
        }
    }
}
```

### Model loading from local with custom components
You can use your own custom components functions for your model. You will need to define the details of your components in your script as functions.

Example:
```json
{
    "input_model": {
        "type": "PyTorchModel",
        "config": {
            "model_script": "user_script.py",
            "hf_config": {
                "model_class": "WhisperForConditionalGeneration",
                "model_name": "openai/whisper-medium",
                "components": [
                    {
                        "name": "encoder_decoder_init",
                        "io_config": "get_encdec_io_config",
                        "component_func": "get_encoder_decoder_init",
                        "dummy_inputs_func": "encoder_decoder_init_dummy_inputs"
                    },
                    {
                        "name": "decoder",
                        "io_config": "get_dec_io_config",
                        "component_func": "get_decoder",
                        "dummy_inputs_func": "decoder_dummy_inputs"
                    }
                ]
            }
        }
    },
}
```

#### Script example
```python
# my_script.py
def get_dec_io_config(model: OliveModelHandler):
    # return your io dict
    ...

def get_decoder(model: OliveModelHandler):
    # your component implementation
    ...

def dummy_inputs_func(model: OliveModelHandler):
    # return the dummy input for your component
    ...
```

### Model loading from Azure ML resources
Olive supports loading model from your Azure Machine Learning workspace. Find detailed configurations [here](./azureml_integration.md).

Example: [Llama-2-7b](https://ml.azure.com/models/Llama-2-7b/version/13/catalog/registry/azureml-meta) from Azure ML model catalog:
```json
"input_model":{
    "type": "PyTorchModel",
    "config": {
        "model_path": {
            "type": "azureml_registry_model",
            "config": {
                "name": "Llama-2-7b",
                "registry_name": "azureml-meta",
                "version": "13"
            }
        },
        "model_file_format": "PyTorch.MLflow",
        "hf_config": {
            "model_name": "meta-llama/Llama-2-7b-hf",
            "task": "text-generation"
        }
    }
}
```

Please note the model for `Llama-2-7b` in Azure ML model catalog is a mlflow model. So `"model_file_format": "PyTorch.MLflow"` is required here.

## Huggingface datasets
Olive supports automatically downloading and applying [Huggingface datasets](https://huggingface.co/datasets) to Passes and Evaluators.

Datasets can be added to `data_configs` section in the configuration file with `"type": "HuggingfaceContainer"`. More details about `data_configs` can be found [here](../tutorials/configure_data.rst).

You can reference the dataset by its name in the Pass config

Example: datasets in `data_configs`:
```json
"data_configs": [{
    "name": "oasst1_train",
    "type": "HuggingfaceContainer",
    "params_config": {
        "data_name": "timdettmers/openassistant-guanaco",
        "split": "train",
        "component_kwargs": {
            "pre_process_data": {
                "text_cols": ["text"],
                "corpus_strategy": "line-by-line",
                "source_max_len": 512,
                "pad_to_max_len": false
            }
        }
    }
}]
```

Pass config:
```json
"perf_tuning": {
    "type": "OrtPerfTuning",
    "config": {
        "data_config": "oasst1_train"
    }
}
```

## Huggingface metrics
Huggingface metrics in Olive are supported by [Huggingface evaluate](https://huggingface.co/docs/evaluate/index). You can refer to [Huggingface metrics page](https://huggingface.co/metrics) for a complete list of available metrics.

Example metric config
```json
{
    "name": "accuracy",
    "type": "accuracy",
    "backend": "huggingface_metrics",
    "data_config": "oasst1_train",
    "sub_types": [
        {"name": "accuracy", "priority": -1},
        {"name": "f1"}
    ]
}
```
Please refer to [metrics](../overview/options.md#metrics) for more details.

## Huggingface login
For certain gated models or datasets, you need to log in to your Huggingface account to access them. If the Huggingface resources you are using require a token, please add `hf_token: true` to the Olive system configuration. Olive will then automatically manage the Huggingface login process, allowing you to access these gated resources.

### Local system, docker system and Python environment system
For local system, docker system and Python environment system, please run command `huggingface-cli login` in your terminal to login your Huggingface account. Find more details about login [here](https://huggingface.co/docs/huggingface_hub/quick-start#login).

### AzureML system
Follow these steps to enable Huggingface login for AzureML system:
1. Get your Huggingface token string from Settings -> [Access Tokens](https://huggingface.co/settings/tokens).
1. Create or use an existing [Azure Key Vault](https://learn.microsoft.com/en-us/azure/key-vault/general/overview). Assume the key vault is named `my_keyvault_name`. Add a new secret named `hf-token`, and set the value as the token from the first step. It is important to note that Olive reserves `hf-token` secret name specifically for Huggingface login. Do not use this name in this keyvault for other purpose.
1. Make sure you have `azureml_client` section in your configuration file, and add a new attribute `keyvault_name` to it. For example:
    ```json
    "azureml_client": {
        "subscription_id": "<subscription_id>",
        "resource_group": "<resource_group>",
        "workspace_name": "<workspace_name>",
        "keyvault_name" : "my_keyvault_name"
    }
    ```
1. Configure the Managed Service Identity (MSI) for the host compute or target compute. Detailed instruction can be found [here](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-setup-authentication?view=azureml-api-2&tabs=sdk#configure-a-managed-identity). Then grant the host compute or target compute access to the key vault resource following this [guide](https://learn.microsoft.com/en-us/azure/key-vault/general/assign-access-policy?tabs=azure-portal)

With the above steps, Olive can automatically retrieve your Huggingface token from the `hf-token` secret in the `my_keyvault_name` key vault, and log in your Huggingface account in the AML job.

## E2E example
For the complete example, please refer to [Bert Optimization with PTQ on CPU](https://github.com/microsoft/Olive/tree/main/examples/bert#bert-optimization-with-ptq-on-cpu).
