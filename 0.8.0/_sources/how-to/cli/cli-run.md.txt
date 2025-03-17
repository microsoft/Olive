# Run Olive workflows

The Olive `run` command allows you to execute any of the 40+ optimizations available in Olive in a sequence you define in a YAML/JSON file called a *workflow*.

## {octicon}`zap` Quickstart

In this quickstart, you'll execute the following Olive workflow:

```{mermaid}
graph LR
    A[/"`Llama-3.2-1B-Instruct (from Hugging Face)`"/]
    C["`IncDynamicQuantization`"]
    A --> B[OnnxConversion]
    B --> C
    C --> E[OrtSessionParamsTuning]
    E --> F[/ZipFile/]
```

The *input* into the workflow is the Llama-3.2-1B-Instruct model from Hugging Face. The workflow has the following of *passes* (steps):

1. Convert the model into the ONNX format using the `OnnxConversion` pass.
1. Quantize using the `IncDynamicQuantization` pass (Intel® Neural Compressor Dynamic Quantization).
1. Optimize the ONNX Runtime inference settings using the `OrtSessionParamsTuning` pass.

The *output* of the workflow is a Zip file containing the ONNX model and ORT configuration settings.

### {octicon}`code-square` Define the workflow in a YAML file

First, define the 'quickstart workflow' in a YAML file. Alternatively, you can use a JSON file. For more details about the available options for the configuration file, please refer to this [reference](../../reference/options.md):

```yaml
# quickstart-workflow.yaml
input_model:
  type: HfModel
  model_path: meta-llama/Llama-3.2-1B-Instruct
systems:
  local_system:
    type: LocalSystem
    accelerators:
      - device: cpu
        execution_providers:
          - CPUExecutionProvider
data_configs:
  - name: transformer_token_dummy_data
    type: TransformersTokenDummyDataContainer
passes:
  conversion:
    type: OnnxConversion
    target_opset: 16
    save_as_external_data: true
    all_tensors_to_one_file: true
    save_metadata_for_token_generation: true
  quantize:
    type: IncDynamicQuantization
  session_params_tuning:
    type: OrtSessionParamsTuning
    data_config: transformer_token_dummy_data
    io_bind: true
packaging_config:
  - type: Zipfile
    name: OutputModel
log_severity_level: 0
host: local_system
target: local_system
cache_dir: cache
output_dir: null
```

### {octicon}`rocket` Run the workflow

The workflow is executed using the `run` command:

```bash
olive run --config quickstart-workflow.yaml
```

