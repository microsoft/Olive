# OpenVINO Model Package (`OpenVINOModelPackage`)

`OpenVINOModelPackage` compiles an ONNX model for the selected OpenVINO device or
devices (NPU, GPU, or CPU) and creates an ONNX Runtime `.ortpackage`. The package
directory is the deliverable.

---

## Prerequisites

1. **Python** 3.12, the OpenVINO stack for the selected device.
2. **Olive** installed:
   ```bash
   pip install -e .
   ```
3. **OpenVINO EP wheels** — install compatible builds:
   ```bash
   pip install --force-reinstall --no-deps \
     onnxruntime_openvino \
     onnxruntime_ep_openvino \
     openvino
   ```
    Use ONNX Runtime 1.30 or newer with the OpenVINO EP plugin to compile packages.
    Loading packages through the stable `OrtModelPackageApi` requires ONNX Runtime 1.31 or newer.
    The ORT Python build must expose `onnxruntime.ModelPackage`.

---

## Target devices

`target_devices` accepts OpenVINO device IDs in uppercase hexadecimal. Leave it empty to
compile for every device the plugin exposes, of any type. For physical CPU, GPU, or NPU
devices, set it explicitly so only the device you want is compiled. The pass fails if a
requested device is unavailable or compilation fails.

Whether those ids resolve to *virtual* cross-compile targets or the *physical* local
device is decided by **`ep_registration_name`**:

| `ep_registration_name`            | Devices exposed                          |
| --------------------------------- | ---------------------------------------- |
| `"ovep_registration.virtual"` *(default)* | Virtual cross-compile devices |
| `"OpenVINO"` (any plain name)     | Physical OpenVINO devices on this machine |

Known virtual NPU device IDs:

| Device | ID |
| ------ | -- |
| Lunar Lake (LNL) | `643E` |
| Panther Lake (PTL) | `B03E` |
| Nova Lake (NVL) | `D71D` |


---

## Run it

Save a workflow config as `my_model_package_config.json`, set `input_model.model_path` and
`engine.output_dir` to your model and output location, then run:

```bash
python -m olive run --config my_model_package_config.json
```

**Example** `my_model_package_config.json` for a physical GPU. Set `device` to `cpu` or `npu`
for those devices, and `target_devices` to the ID of the device:

```json
{
  "input_model": { "type": "ONNXModel", "model_path": "C:/path/to/your_model.onnx" },
  "systems": {
    "local_system": {
      "type": "LocalSystem",
      "accelerators": [ { "device": "gpu", "execution_providers": [ "OpenVINOExecutionProvider" ] } ]
    }
  },
  "host": "local_system",
  "target": "local_system",
  "passes": {
    "ov_model_package": {
      "type": "OpenVINOModelPackage",
      "package_name": "my-model",
      "ep_registration_name": "OpenVINO",
      "target_devices": [ "<device id>" ]
    }
  },
  "engine": {
    "search_strategy": false,
    "evaluate_input_model": false,
    "no_artifacts": true,
    "output_dir": "C:/path/to/output"
  }
}
```

### Physical devices (GPU, CPU, local NPU)

A plain `ep_registration_name` such as `"OpenVINO"` exposes the physical devices on this
machine. List their IDs and types:

```bash
python -c "import onnxruntime as ort, onnxruntime_ep_openvino as ep; ort.register_execution_provider_library('OpenVINO', ep.get_library_path()); print([(f'{d.device.device_id:X}', str(d.device.type)) for d in ort.get_ep_devices() if d.ep_name == 'OpenVINOExecutionProvider'])"
```

### Virtual NPU targets

The default registration exposes virtual NPUs, so one machine can build a package for several
NPU generations. Refer the example above, set `device` to `npu`, remove
`ep_registration_name`, and list the IDs from the table above in `target_devices`, for example
`[ "643E", "B03E", "D71D" ]`.

---

## Options

Pass options live under `passes.<name>` in the Olive config. The example above shows
the required workflow structure. Common options are `package_name`, `target_devices`,
`session_options` for runtime settings, and `compile_session_options` for compile-time
EPContext settings such as embed mode. Compilation uses the AOT flow.

---

## `package_only` mode (package without compiling)

Assemble a package from EPContext models you already compiled — no device or compile step:

```json
"passes": {
  "ov_model_package": {
    "type": "OpenVINOModelPackage",
    "package_name": "my-model",
    "package_only": true,
    "session_options": { "ep.context_file_path": "model_ctx.onnx" },
    "shared_asset_model_path": "C:/path/to/original_model.onnx",
    "compiled_model_paths": {
      "643E": "C:/path/to/lnl/model_ctx.onnx",
      "B03E": "C:/path/to/ptl/model_ctx.onnx"
    }
  }
}
```

Each `compiled_model_paths` value may be an EPContext `.onnx` file or a directory
containing that file. Set `shared_asset_model_path` to the original ONNX model when
the packaged EPContext models were compiled from a different input path. Keep
`ep.context_file_path` in `session_options`; without it the package builds but fails to load.

---

This pass creates packages for ONNX Runtime's model-package loader. Its output is not a
replacement for packages generated by `olive generate-model-package` for ONNX Runtime GenAI.
Applications that load these packages should use ONNX Runtime's stable C/C++
`OrtModelPackageApi`, obtained from `OrtApi::GetModelPackageApi`. Package authoring
uses ORT's `onnxruntime.ModelPackage` Python API.
