# OpenVINO Model Package (`EPContextBinaryGenerator` + `ORTModelPackage`)

Creating an ONNX Runtime `.ortpackage` for OpenVINO takes two passes:

1. **`EPContextBinaryGenerator`** compiles an ONNX model into an EPContext model for each
   selected OpenVINO device (NPU, GPU, or CPU).
2. **`ORTModelPackage`** assembles EPContext models into an `.ortpackage` through ONNX Runtime's
   `onnxruntime.ModelPackage` authoring API. It is not specific to OpenVINO; the execution provider
   comes from the workflow's accelerator.

The package directory is the deliverable.

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

`EPContextBinaryGenerator.target_devices` accepts OpenVINO device IDs in uppercase hexadecimal and
produces a composite model with one EPContext model per device, each in its own `<device_id>`
directory. The pass fails if a requested device is unavailable or compilation fails.

Whether those ids resolve to *virtual* cross-compile targets or the *physical* local
device is decided by **`ep_registration_name`**:

| `ep_registration_name`            | Devices exposed                          |
| --------------------------------- | ---------------------------------------- |
| `"ovep_registration.virtual"`     | Virtual cross-compile devices            |
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

**Example** `my_model_package_config.json` for virtual NPU targets:

```json
{
  "input_model": { "type": "ONNXModel", "model_path": "C:/path/to/your_model.onnx" },
  "systems": {
    "local_system": {
      "type": "LocalSystem",
      "accelerators": [ { "device": "npu", "execution_providers": [ "OpenVINOExecutionProvider" ] } ]
    }
  },
  "host": "local_system",
  "target": "local_system",
  "passes": {
    "ep_context": {
      "type": "EPContextBinaryGenerator",
      "compile_flow": "aot",
      "ep_registration_name": "ovep_registration.virtual",
      "target_devices": [ "643E", "B03E", "D71D" ],
      "output_model_name": "model_ctx.onnx",
      "session_options": {
        "ep.context_enable": "1",
        "ep.context_embed_mode": "0",
        "ep.enable_weightless": "1"
      }
    },
    "ov_model_package": {
      "type": "ORTModelPackage",
      "package_name": "my-model",
      "session_options": { "ep.context_file_path": "model_ctx.onnx" }
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

`EPContextBinaryGenerator` options are passed to ONNX Runtime verbatim; Olive does not set any
EPContext keys. For `compile_flow: "aot"`, embed mode is taken from
`model_compiler_options.embed_compiled_data_into_model`, not from `ep.context_embed_mode`.

### Physical devices (GPU, CPU, local NPU)

A plain `ep_registration_name` such as `"OpenVINO"` exposes the physical devices on this
machine. List their IDs and types:

```bash
python -c "import onnxruntime as ort, onnxruntime_ep_openvino as ep; ort.register_execution_provider_library('OpenVINO', ep.get_library_path()); print([(f'{d.device.device_id:X}', str(d.device.type)) for d in ort.get_ep_devices() if d.ep_name == 'OpenVINOExecutionProvider'])"
```

Set the accelerator `device` to `gpu`, `cpu`, or `npu`, `ep_registration_name` to `"OpenVINO"`, and
`target_devices` to the ID of the device.

---

## `ORTModelPackage` options

| Option | Purpose |
| ------ | ------- |
| `package_name` | Package/component name. Defaults to the shared-asset model's file stem. |
| `session_options` | Runtime session options written into each variant's `executor_info.ort.session_options`. Keep `ep.context_file_path` set to the EPContext model file name; without it the package builds but fails to load. |
| `variant_name_template` | Variant naming. Fields: `package_name`, `device`, `variant`. Defaults to `{package_name}.{device}_{variant}`. |
| `create_shared_asset` | Stage the original model and its external-data files once as a content-addressed asset that weightless variants read their weights from. |
| `shared_asset_model_path` | Original ONNX model for the shared asset. Defaults to the `source_model_path` attribute set by `EPContextBinaryGenerator`. |

The `variant` field is the component name for a composite input model (the device ID when produced
by `EPContextBinaryGenerator`) and `default` for a single ONNX model.

---

## Packaging existing EPContext models

`ORTModelPackage` does not compile, so it can package EPContext models thats already available, even when
they are in different folders. Provide them as a `CompositeModel` input, with one `ONNXModel` component
per EPContext model. The component name becomes the variant. A single ONNX model input is also accepted
and becomes one `default` variant.

```json
{
  "input_model": {
    "type": "CompositeModel",
    "model_component_names": [ "643E", "B03E" ],
    "model_components": [
      { "type": "ONNXModel", "model_path": "C:/path/to/lnl", "onnx_file_name": "model_ctx.onnx" },
      { "type": "ONNXModel", "model_path": "C:/path/to/ptl", "onnx_file_name": "model_ctx.onnx" }
    ]
  },
  "systems": {
    "local_system": {
      "type": "LocalSystem",
      "accelerators": [ { "device": "npu", "execution_providers": [ "OpenVINOExecutionProvider" ] } ]
    }
  },
  "host": "local_system",
  "target": "local_system",
  "passes": {
    "ov_model_package": {
      "type": "ORTModelPackage",
      "package_name": "my-model",
      "shared_asset_model_path": "C:/path/to/original_model.onnx",
      "session_options": { "ep.context_file_path": "model_ctx.onnx" }
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

Where each file comes from:

| Files | How they are passed |
| ----- | ------------------- |
| EPContext models | One component per model in `model_components`. Each can be in its own folder. |
| EPContext `.bin` files | Not passed separately. They are copied from each EPContext model's folder: files referenced by the model's EPContext nodes (`ep_cache_context`, as a relative path inside that folder) and `<model stem>*.bin` files next to it. Keep them beside, or below, the EPContext model. |
| Original model and its weights | `shared_asset_model_path` points to the original `.onnx`. The external-data (weight) files recorded in that model are copied from its folder, so keep them where the model expects them. |

Set `create_shared_asset` to `false` if the EPContext models embed their weights. If they still read
external data, provide the weights folder through `session_options`
(`session.model_external_initializers_file_folder_path`). Paths that are absolute or contain `..` are rejected.

---

This pass creates packages for ONNX Runtime's model-package loader. Its output is not a
replacement for packages generated by `olive generate-model-package` for ONNX Runtime GenAI.
Applications that load these packages should use ONNX Runtime's stable C/C++
`OrtModelPackageApi`, obtained from `OrtApi::GetModelPackageApi`. Package authoring
uses ORT's `onnxruntime.ModelPackage` Python API.
