## Prerequisites
* einops

## Usage
```bash
python -m olive.workflows.run --config phi2_optimize.json
```

## Limitations
From the time being, the official [phi2](https://huggingface.co/microsoft/phi-2) model could not be exported to ONNX by using the official model code. Therefore, we need patch the forward method to do preprocessing and postprocessing for the past_key_values arguments. When the official model could be exported to ONNX, we will remove this patch.

when https://github.com/huggingface/optimum/issues/1642 is fixed, we need turn on the post_process by changing `no_post_process` to False in the config file.
