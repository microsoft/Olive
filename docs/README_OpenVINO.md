# OpenVINO Model Package (`OpenVINOModelPackage`)

An Olive pass that compiles an ONNX model into per-device **OpenVINO NPU EPContext**
variants and assembles a self-contained **ONNX Runtime model package**
(`.ortpackage`). ONNX Runtime selects the correct variant for the available NPU at
load time.

The original model and its external-data weights are stored **once** as a
content-addressed shared asset; every compiled (weightless) variant reads its weights
from that shared asset, so the weights are never duplicated per device.

---

## Output layout

```
<output_dir>/
├── model_config.json                     # Olive model descriptor (outside the package)
└── <package_name>.ortpackage/
    ├── manifest.json
    ├── <package_name>/
    │   ├── <package_name>.npu_643E/       # one variant per target device
    │   │   ├── model_ctx.onnx             # compiled EPContext model
    │   │   └── model_ctx_OpenVINOExecutionProvider.bin
    │   └── <package_name>.npu_B03E/
    │       ├── model_ctx.onnx
    │       └── model_ctx_OpenVINOExecutionProvider.bin
    └── shared_assets/
        └── sha256-<hex>/                  # original model + weights (shared once)
            ├── <model>.onnx
            └── <weights>.data
```

The `.ortpackage/` directory is the deliverable. `model_config.json` is Olive
bookkeeping written next to it (not inside it).

---

## Prerequisites

1. **Python** 3.12, on a machine with an Intel NPU / the OpenVINO NPU stack.
2. **Olive** installed (editable install from this repo):
   ```bash
   pip install -e .
   ```
3. **OpenVINO EP wheels** — install the provided builds:
   ```bash
   pip install --force-reinstall --no-deps \
     onnxruntime_openvino \
     onnxruntime_ep_openvino
   ```
   These provide `onnxruntime` (≥ 1.26, with the `ModelCompiler` compile API) and the
   `onnxruntime_ep_openvino` plugin package (`get_library_path()` / `get_ep_name()`).

Verify the plugin exposes virtual devices:
```bash
python -c "import onnxruntime as ort, onnxruntime_ep_openvino as ep; \
ort.register_execution_provider_library('ovep_registration.virtual', ep.get_library_path()); \
print([ (f'{d.device.device_id:X}', dict(d.ep_metadata).get('npu_platform')) \
        for d in ort.get_ep_devices() if d.ep_name=='OpenVINOExecutionProvider' \
        and dict(d.device.metadata).get('is_virtual')=='1' ])"
```

---

## Target devices

`target_devices` takes NPU **device ids** (uppercase hex). Leave it empty/unset to
auto-discover and compile for **every** device the plugin exposes.

Whether those ids resolve to *virtual* cross-compile targets or the *physical* local
device is decided by **`ep_registration_name`** (a single knob):

| `ep_registration_name`            | Devices exposed                          |
| --------------------------------- | ---------------------------------------- |
| `"ovep_registration.virtual"` *(default)* | Virtual cross-compile NPUs (all platforms below) |
| `"OpenVINO"` (any plain name)     | The physical NPU/GPU/CPU on this machine |

Known virtual NPU ids (from the OpenVINO plugin's device table):

| Device id | Platform | Codename          |
| --------- | -------- | ----------------- |
| `643E`    | 4000     | Lunar Lake (LNL)  |
| `B03E`    | 5010     | Panther Lake (PTL)|
| `FD3E`    | 5020     | Wildcat Lake      |

---

## Run it

[`model_package_config_template.json`](../olive/passes/openvino/examples/model_package_config_template.json)
is a **template** — its `input_model.model_path` and `engine.output_dir` are placeholder paths. Copy
it and set those to your model and output location first, then run:

```bash
python -m olive run --config my_model_package_config.json
```

---

## Configuration reference

Pass options live under `passes.<name>` in the Olive config.

| Option                         | Default                                                     | Description |
| ------------------------------ | ----------------------------------------------------------- | ----------- |
| `package_name`                 | input model stem                                            | Package/component name; also the `.ortpackage` folder name. |
| `package_version`              | `"1.0.0"`                                                   | `manifest.package_version`. |
| `package_suffix`               | `".ortpackage"`                                             | Package root directory suffix. |
| `target_devices`               | *(auto-discover all)*                                       | List of device ids (virtual or physical), e.g. `["643E","B03E"]`. |
| `device`                       | accelerator device type                                     | Device class recorded on each variant and used in the variant name. |
| `variant_name_template`        | `"{package_name}.{device}_{device_id}"`                     | Variant naming; fields: `package_name`, `device`, `device_id`. |
| `output_ctx_name`              | `"model_ctx.onnx"`                                          | File name of the compiled EPContext model in each variant dir. |
| `compile_flow`                 | `"aot"`                                                     | `aot` = ORT `ModelCompiler`; `jit` = `InferenceSession`. |
| `compile_options`              | `null`                                                      | ORT session config entries applied at compile time (e.g. `ep.context_embed_mode`, `ep.enable_weightless_ep_context_nodes`). |
| `session_options`              | `null`                                                      | Runtime session options written verbatim into each variant's `executor_info.ort.session_options` (e.g. `ep.context_file_path`). |
| `provider_options`             | `null`                                                      | Options passed verbatim to the OpenVINO EP for every compile (e.g. `load_config`). |
| `create_shared_asset`          | `true`                                                      | Stage the original model + external-data as a content-addressed shared asset. |
| `shared_asset_model_path`      | input model path                                            | Model used for shared-asset staging (set in `package_only` mode). |
| `external_initializers_option` | `"session.model_external_initializers_file_folder_path"`    | Session-option key set to the shared-asset URI so weightless variants find their weights. |
| `executor_namespace`           | `"ort"`                                                     | `executor_info` namespace key. |
| `ep_registration_name`         | `"ovep_registration.virtual"`                               | EP library registration name. The `.virtual` suffix exposes virtual cross-compile devices; a plain name (e.g. `"OpenVINO"`) exposes the physical device. |
| `package_only`                 | `false`                                                     | Skip compilation and package existing EPContext model(s). |
| `compiled_model_paths`         | `null`                                                      | `{device_id: path}` of pre-compiled EPContext models for `package_only` mode. |

### Passing OpenVINO NPU properties

Provider option `load_config` is a **JSON string**, nested by device name. NPU property
keys come from the OpenVINO NPU properties header (e.g. `NPU_TILES`, `NPU_COMPILER_TYPE`):

```json
"provider_options": {
  "load_config": "{\"NPU\":{\"NPU_TILES\":3,\"NPU_COMPILER_TYPE\":\"PLUGIN\"}}"
}
```

---

## `package_only` mode (package without compiling)

Assemble a package from EPContext models you already compiled — no NPU/compile step:

```json
"passes": {
  "ov_model_package": {
    "type": "OpenVINOModelPackage",
    "package_name": "my-model",
    "package_only": true,
    "shared_asset_model_path": "C:/path/to/original_model.onnx",
    "compiled_model_paths": {
      "643E": "C:/path/to/lnl/model_ctx.onnx",
      "B03E": "C:/path/to/ptl/model_ctx.onnx"
    }
  }
}
```

Each `compiled_model_paths` value may be the ctx `.onnx` file or a directory containing
`output_ctx_name`; the pass copies the ctx model and its sidecar `.bin` into the variant.

---

## Package format vs `olive generate-model-package`

This pass and the `olive generate-model-package` CLI both emit `.ortpackage` directories, but they
use **different, intentionally distinct manifest schemas** for different consumers:

| | `OpenVINOModelPackage` (this pass) | `olive generate-model-package` CLI |
| --- | --- | --- |
| `schema_version` | `"1.0"` (string) | `1` (integer) |
| `components` | object/map, variants embedded inline in `manifest.json` (`executor_info`) | list of names; per-component `models/<component>/metadata.json` + `genai_config_overlay.json` |
| Consumer | ONNX Runtime **`OrtModelPackageApi`** (experimental `_SinceV28`) variant selection | ONNX Runtime GenAI variant selection |

This pass targets the ORT `OrtModelPackageApi` model-package loader (the format produced by the
reference `model_package_tool` C++ sample and validated by its `load` command), which is why it
uses the string `schema_version` and inline `executor_info`. Do not mix the two formats: pick the
pass whose output matches your loader. The manifest shape is covered by
`test_package_only_manifest_shape` in `test/passes/openvino/test_openvino_model_package.py`.

---

## Notes

- **`no_artifacts: true`** in the `engine` block keeps the package clean (suppresses
  Olive's `footprint.json` / `output_footprint.json` / `run_history.txt`).
- **`allow_partial_package`** (default `false`): the pass fails if any requested target device
  fails to compile. Set it `true` to skip failed devices and package the ones that succeeded.
- **`compatibility_string`**: the per-variant `compatibility_string` in `manifest.json`
  is read from the compiled model's `ep_compatibility_info.OpenVINOExecutionProvider`
  metadata. With the current OpenVINO NPU stack, weightless compilation emits an empty
  `ov_compat_string` (a known OpenVINO limitation); non-weightless compilation emits a
  populated per-device string.
