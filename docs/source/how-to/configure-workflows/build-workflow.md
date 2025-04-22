# How to build an Olive workflow from scratch

Olive workflows are json files with following configuration.
- `workflow_id: str`: A unique identifier used to identify outputs in intermediate cache and output models directory. Defaulted to _default_workflow_.
- `azureml_client: AzureMLClientConfig`: AzureML client configuration to be used for all cloud based resources during run.
- `input_model: [ModelConfig](../configure-workflows/how-to-configure-model.md) `: Input model configuration.
- `system: Dict[str, [SystemConfig](../configure-workflows/systems.md)]`: List of named system configurations.
- `data_configs: List[[DataConfig](../configure-workflows/how-to-configure-data.md)]`: List of data configurations.
- `evaluators: Dict[str, [OliveEvaluatorConfig](../configure-workflows/metrics-configuration.md)]`: List of named evaluator configurations.
- `engine: RunEngineConfig`: Engine configuration.
- `passes: Dict[str, List[[RunPassConfig](../configure-workflows/pass-configuration.md)]]`: List of named passes to run.
- `auto_optimizer_config: AutoOptimizerConfig`: Configuration to auto-optimize the input model.
- `workflow_host: [SystemConfig](../configure-workflows/systems.md)`: System configuration to use host the workflow run.

```{Note}
All fields of the workflow are optional.
```

## AzureML Client Configuration

Developers can configure Olive to use Azure cloud resources, storage and compute. For detailed instructions on configuring and using Azure Machine Learning for your workflow, see [azure-ai](../../features/azure-ai/azure-ai.md)

## Input Model Configuration

Olive supports a number of different input model types including **HuggingFace**, **Pytorch**, **ONNX**, **OpenVINO**, **QNN**, **SNPE**, **TensorFlow**, and **Composite**. For detailed instructions on configuring your input model, see [how to configure input model](../configure-workflows/how-to-configure-model.md).

## System Configuration

**System** in Olive is a concept that encapsulates operating system, hardware configuration, device (cpu, gpu, npu, etc.), and execution provider needed to run and evaluate the input model during workflow run. Note that the system used to run the workflow (i.e. the **host** system) doesn't necessarily have to be the same system (i.e. the **target** system) where the model will finally be deployed and used for inferencing. Olive supports a number of system configurations including **local system**, **Python environment system**, **Docker system**, **AzureML system**, and **Isolated ORT system**. For detailed instructions on configuring your host and target system, see [how to configure system](../configure-workflows/systems.md).

## Data Configuration

Datasets, provided via the **DataConfig** are used for fine-tuning, quantization, and evaluating the models during workflow run. DataConfig supports loading, preprocessing, batching, and postprocessing of number of different data formats and providers. In addition, Olive also provides templates for many commonly used dataset like ones from HuggingFace. For detailed instructions on configuring data, see [how to configure data](../configure-workflows/how-to-configure-data.md).

## Evaluator Configuration

Evaluators are used to compute metrics for models (input and generated) during workflow run. Computed metrics are used to make runtime decisions to influence the workflow path during search for optimized model and the generated output. Olive includes evaluators for all supported input model types and widely used python evaluation packages like [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness). For detailed instructions on configuring evaluators and metrics, see [how to configure evaluator](../configure-workflows/metrics-configuration.md).

## Engine Configuration

Engine is the glue that binds different components of the workflow together to generate the output. Options like host system, target system, evaluator, etc can be configured through `EngineConfig`.

## Pass Configuration

Passes in Olive are the actionable items in the workflow. Input model flows through the list of passes to generate the output. Each implements a specific process/algorithm (like conversion, quantization, etc). The output of the previous pass is usually fed as an input to the next pass in the list. Note, however, that the walk through the list of passes is still not a linear one as output of the same pass can be used as input to multiple different passes. For detailed instructions on configuring passes, see [how to configure pass](../configure-workflows/pass-configuration.md). For a complete list of available passes, see [passes](../../reference/pass.rst).

## Auto Optimizer Configuration

Auto optimizer configures the different available passes based on the input model's configuration to fine-tune and optimize the model. Developers can also provide additional inputs through the `AutoOptimizerConfig` to influence the searched passes. For detailed instructions on configuring auto optimization process, see [how to configure pass](../../features/auto-opt.md).

For complete examples of workflows, developers can refer to the examples folders in Olive repository.
