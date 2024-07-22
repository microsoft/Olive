# Super Resolution Optimization with OnnxRuntime extension
This folder demonstrates an examples of using OnnxRuntime extension to optimize Super Resolution.
Visit [OnnxRuntime Extension](https://github.com/microsoft/onnxruntime-extensions) for installation and
 usage instructions.
Visit [Super Resolution with OnnxRuntime](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)
 for detailed information.

## Using OnnxRuntime extension with Olive
Olive includes a specific pass `AppendPrePostProcessingOps` to append pre- and post- processing operations to exported
 ONNX model.

```json
"passes": {
    "prepost": {
        "type": "AppendPrePostProcessingOps",
        "tool_command": "superresolution",
        "tool_command_args": {
            "output_format": "png"
        }
    }
}
```

## How to run
### Pip requirements
Install the necessary python packages:
```sh
python -m pip install -r requirements.txt
```

### Run sample using config
```sh
olive run --config config.json
```

or run simply with python code:
```python
from olive.workflows import run as olive_run
olive_run("config.json")
```

After running the above command, the model and corresponding config will be saved in the output directory.
