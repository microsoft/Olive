# Design: Multi-Component Model Optimization in Olive


## 1. Problem Statement

Olive needs to optimize **multi-component models** (different components → different optimizations) **and** produce **multiple target-specific outputs** from one config. Motivating cases:

- **Multi-component models:**
  - **VLM / multimodal HF models** — quantize the language decoder (e.g. GPTQ int4) while keeping the vision tower / projector at higher precision.
  - **Diffusion models** (SD / SDXL / SD3 / FLUX) — optimize text encoders, the diffusion backbone (UNet/transformer), and VAE differently.
  - **Future multi-component families** — without Olive learning every architecture's naming convention.
- **Multi-device / multi-EP builds** — produce several target-specific outputs from a single config, e.g. an OpenVINO **GPU** build and an OpenVINO **NPU** build of the same model, each with its own conversion/quantization/encapsulation pipeline and `host`/`target`.


## 2. Approach

### The `builds` Schema

A build is a named CLI orchestration unit:

```python
class BuildNode:
    components: list[str] | None   # component names; omitted ⇒ full model
    pipeline: list[str]            # ordered pass names from the top-level `passes`
    output_dir: str
    host: SystemConfig | str | None
    target: SystemConfig | str | None
    evaluator: OliveEvaluatorConfig | str | None
    search_strategy: SearchStrategyConfig | bool | None
```

Semantics:

- **`components`** selects named components from the resolved `CompositeModel`. Omitted ⇒ run on the full model. A single name unwraps to that component; multiple names produce a sub-composite.
- **`pipeline`** lists pass names from the shared top-level `passes` dict, composed per build. Different builds reuse the same pass definitions in different orders/subsets.
- **`host`/`target`/`evaluator`/`search_strategy`** override engine defaults per build.

`builds` is expanded before `RunConfig` validation. Each named build becomes a complete, ordinary Olive run
configuration with its selected passes, input-model scope, engine overrides, output directory, and a namespaced
`workflow_id`. The expanded configurations then use the existing single-workflow execution path without adding
multi-build behavior to `Engine` or `RunConfig`.

From one `input_model`, several independent builds produce several outputs:
- **Per-component builds** — each build optimizes a different `components` subset (one model in → one output per component).
- **Per-target builds** — builds omit `components` and differ by `host`/`target`/`pipeline`, one output per device/EP.

Build-to-build dependencies are intentionally not part of the initial design.

---

## 3. Config Examples

### 3.1 Basic shape — independent sibling builds

Shared `passes`; each build picks a component and composes its own `pipeline`. Each build writes one optimized folder. The same shape is used whether the user optimizes ONNX components after export (Flow A) or PyTorch components before export (Flow B).

```jsonc
{
  "input_model": { "type": "DiffusersModel", "model_path": "stabilityai/stable-diffusion-xl-base-1.0" },
  "systems": {
    "local_system": { "type": "LocalSystem", "accelerators": [ { "device": "gpu", "execution_providers": ["CUDAExecutionProvider"] } ] }
  },
  "data_configs": [
    { "name": "quantize_data_config", "user_script": "user_script.py",
      "load_dataset_config": { "type": "local_dataset" },
      "dataloader_config": { "type": "quantize_data_loader", "data_num": 100 } }
  ],
  "passes": {
    "convert":       { "type": "OnnxConversion", "target_opset": 17 },
    "optimize_clip": { "type": "OrtTransformersOptimization", "model_type": "clip", "float16": true },
    "optimize_vae":  { "type": "OrtTransformersOptimization", "model_type": "vae",  "float16": true },
    "optimize_unet": { "type": "OrtTransformersOptimization", "model_type": "unet", "float16": true },
    "quantization":  { "type": "OnnxStaticQuantization", "data_config": "quantize_data_config" }
  },
  "builds": {
    "text_encoder": { "components": ["text_encoder"], "pipeline": ["convert", "optimize_clip", "quantization"], "output_dir": "out/text_encoder", "evaluator": "common_evaluator" },
    "vae_encoder":  { "components": ["vae_encoder"],  "pipeline": ["convert", "optimize_vae",  "quantization"], "output_dir": "out/vae_encoder",  "evaluator": "common_evaluator" },
    "vae_decoder":  { "components": ["vae_decoder"],  "pipeline": ["convert", "optimize_vae",  "quantization"], "output_dir": "out/vae_decoder",  "evaluator": "common_evaluator" },
    "unet":         { "components": ["unet"],         "pipeline": ["convert", "optimize_unet", "quantization"], "output_dir": "out/unet",         "evaluator": "common_evaluator" }
  }
}
```

Each build writes one optimized component under its `output_dir`.


### 3.2 Component optimization — two flows


#### Flow A — export to ONNX model first, then per-component optimization

Export with `MobiusBuilder`, which takes an `HfModel` and produces a directory containing the ONNX components.
The export and per-component optimization are two explicit steps, avoiding build dependency and DAG semantics.

Step 1 — export. Each component lands in its own subfolder:

```powershell
olive capture-onnx-graph --model_name_or_path <vlm> --use_mobius_builder --output_path exported_pkg
# exported_pkg/decoder/model.onnx, exported_pkg/vision_encoder/model.onnx, exported_pkg/embedding/model.onnx
```

Step 2 — point `input_model` at that directory. Olive loads it as a `CompositeModel`, taking each **subfolder name as the component name**. Plain sibling builds, no `input` dependency:

```jsonc
{
  "input_model": { "type": "CompositeModel", "model_path": "exported_pkg" },
  "data_configs": [
    { "name": "calib", "user_script": "user_script.py", "load_dataset_config": { "type": "local_dataset" } }
  ],
  "passes": {
    "transformer_opt": { "type": "OrtTransformersOptimization", "float16": true },
    "quantization":    { "type": "OnnxStaticQuantization", "data_config": "calib" }
  },
  "builds": {
    "decoder":        { "components": ["decoder"],        "pipeline": ["transformer_opt", "quantization"], "output_dir": "out/decoder" },
    "vision_encoder": { "components": ["vision_encoder"], "pipeline": ["transformer_opt"],                 "output_dir": "out/vision_encoder" }
  }
}
```

Each subfolder is a standard local ONNX model Olive already loads. The only new piece is aggregating a directory of per-component subfolders into a `CompositeModel` whose component names come from the folder names.

- Pros:
  - Clear config file, no DAG.
  - User doesn't need to call a different CLI to get the components name.
- Cons:
  - 2 steps.
  - User needs to read output model folder to get components name.

#### Flow B — optimize first, then export (recommended)

For PyTorch-stage optimization (e.g. GPTQ on the decoder) **before** export.

##### How components are obtained

###### Option A — Query Mobius (preferred)

Olive calls Mobius at runtime to inspect the model:

```python
components = mobius.inspect_components(model_path_or_id, task=None, trust_remote_code=False)
```

- **Pros:**
  - always in sync with Mobius's own architecture support;
  - no per-model maintenance in Olive;
  - covers any model Mobius can export, including new ones; single source of truth shared with the exporter.
- **Cons:**
  - hard runtime dependency on `mobius-ai` even for the optimization step;
  - coupled to Mobius versions (names/fields may shift)


###### Option B — Olive-maintained YAML registry

Olive ships a YAML file enumerating the components of common models, keyed by `model_type` / architecture. Two component description styles appear, matching the two families:

- **HF/VLM components** only need a **submodule path** to slice the component out of one model. `name` (for `builds.components`) plus `source.path` (where the submodule lives) is enough; `kind` is optional (only used for pass↔kind validation):

```yaml
# olive/model/component_registry.yaml
llava:
  components:
    - { name: decoder,        kind: decoder,        source: { path: "model.language_model" } }
    - { name: vision_encoder, kind: vision_encoder, source: { path: "model.vision_tower" } }
    - { name: embedding,      kind: embedding,      source: { path: "model.language_model.embed_tokens" } }
```

- **Diffusion components** reuse existing Diffusion model components yaml file:

```yaml
stable-diffusion:                      # SD 1.5 family (identified by model_index.json)
  type: DiffusersModel
  components:
    - name: text_encoder
      kind: text_encoder
      loader: { component: text_encoder }              # DiffusersModel.get_component("text_encoder")
      io_config:
        input_names: [input_ids]
        output_names: [last_hidden_state, pooler_output]
        dynamic_axes: { input_ids: { 0: batch, 1: sequence } }
      dummy_inputs: text_encoder                       # generate_diffusers_dummy_inputs(...)
    - name: vae_encoder
      kind: vae_encoder
      loader: { component: vae, patch: get_vae_encoder }   # olive.model.utils.diffusers_utils.get_vae_encoder
      io_config:
        input_names: [sample, return_dict]
        output_names: [latent_sample]
        dynamic_axes: { sample: { 0: batch, 1: channels, 2: height, 3: width } }
      dummy_inputs: vae_encoder
    - name: vae_decoder
      kind: vae_decoder
      loader: { component: vae, patch: get_vae_decoder }   # olive.model.utils.diffusers_utils.get_vae_decoder
      io_config:
        input_names: [latent_sample, return_dict]
        output_names: [sample]
        dynamic_axes: { latent_sample: { 0: batch, 1: channels, 2: height, 3: width } }
      dummy_inputs: vae_decoder
    - name: unet
      kind: diffusion_backbone
      loader: { component: unet }
      io_config:
        input_names: [sample, timestep, encoder_hidden_states, return_dict]
        output_names: [out_sample]
        dynamic_axes:
          sample: { 0: batch, 1: channels, 2: height, 3: width }
          timestep: { 0: batch }
          encoder_hidden_states: { 0: batch, 1: sequence }
      dummy_inputs: unet
    # SDXL adds text_encoder_2 (kind: text_encoder) and extra UNet inputs (text_embeds, time_ids);
    # SD3 / FLUX replace `unet` with `transformer` (kind: diffusion_backbone).
```


- **Pros:**
  - no runtime Mobius dependency for the optimization step;
  - works offline;
  - human-readable, reviewable, and overridable by users (drop-in extra entries);
  - stable across Mobius versions;
  - users can add an unsupported model without code changes.
- **Cons:**
  - must be **maintained by Olive** as new architectures appear (the same per-architecture maintenance Mobius already does);
  - risk of drifting out of sync with Mobius's actual export expectations (e.g. `export_key`s, weight prefixes);
  - duplicates knowledge that also lives in Mobius.

**(a) Optimize each component**. Only the components the user wants to optimize need a build.

```jsonc
{
  "input_model": { "type": "HfModel", "model_path": "<vlm>" },
  "data_configs": [ { "name": "decoder_calib", "user_script": "user_script.py", "load_dataset_config": { "type": "local_dataset" } } ],
  "passes": {
    "decoder_quant": { "type": "Gptq", "bits": 4, "group_size": 128, "data_config": "decoder_calib" }
  },
  "builds": {
    "decoder": { "components": ["decoder"], "pipeline": ["decoder_quant"], "output_dir": "out/decoder" }
  }
}
```

**(b) Converge the optimized component(s) into one complete HF model directory.** The recommended form is **in-place**: the optimization runs on the full model and quantizes only the selected submodule, so step (a)'s `output_dir` is already a complete HF directory with the decoder quantized.

> **`builds.components` means different things for the two families:**
> - **Diffusion:** slice this component out and optimize it independently → independent ONNX artifact.
> - **VLM:** locate and optimize this submodule inside the full model, output the full model → one complete HF directory.

**(c) Export with the existing `capture-onnx-graph` CLI + Mobius builder.** `--use_mobius_builder`, takes `--model_name_or_path` as one complete HF model directory  and lets Mobius re-identify and export the multi-component package.

```powershell
olive capture-onnx-graph `
  --model_name_or_path local_folder `
  --use_mobius_builder `
  --output_path out\pkg
```

**(c) requires a quant format bridge.** Olive saves `quant_method="olive"` with **uint8** packing; Mobius's `preprocess_gptq_weights` expects `quant_method="gptq"`/`"awq"` with **int32** packing. A conversion (or a Mobius `"olive"` branch) is required for Mobius to load the quantized weights.


### 3.3 Per-target builds — multi-device / multi-EP from one config

The **same** `builds` schema produces several target-specific outputs without any `components`. Each build differs only by `host`/`target` and its `pipeline`; shared `passes` are composed per target. This is the OpenVINO GPU + NPU case (Qwen2.5-Coder).

```jsonc
{
  "input_model": { "type": "HfModel", "model_path": "Qwen/Qwen2.5-Coder-7B-Instruct" },
  "systems": {
    "ov_gpu": { "type": "LocalSystem", "accelerators": [ { "device": "gpu", "execution_providers": ["OpenVINOExecutionProvider"] } ] },
    "ov_npu": { "type": "LocalSystem", "accelerators": [ { "device": "npu", "execution_providers": ["OpenVINOExecutionProvider"] } ] }
  },
  "passes": {
    "optimum_convert_gpu": { "type": "OpenVINOOptimumConversion", "extra_args": { "device": "gpu", "task": "text-generation-with-past" }, "ov_quant_config": { "weight_format": "int4", "group_size": 128, "ratio": 0.8 } },
    "optimum_convert_npu": { "type": "OpenVINOOptimumConversion", "extra_args": { "device": "npu" }, "ov_quant_config": { "weight_format": "int4", "group_size": 128, "dataset": "wikitext2", "ratio": 1, "sym": true, "backup_precision": "int8_asym" } },
    "io_update":           { "type": "OpenVINOIoUpdate", "static": false, "reuse_cache": true },
    "encapsulation_gpu":   { "type": "OpenVINOEncapsulation", "target_device": "gpu", "ov_version": "2025.1", "reuse_cache": true },
    "encapsulation_npu":   { "type": "OpenVINOEncapsulation", "target_device": "npu", "ov_version": "2025.2", "reuse_cache": true, "genai_config_override": { "model": { "context_length": 4224 } } }
  },
  "builds": {
    "gpu": { "host": "ov_gpu", "target": "ov_gpu", "search_strategy": false, "pipeline": ["optimum_convert_gpu", "io_update", "encapsulation_gpu"], "output_dir": "gpu_output" },
    "npu": { "host": "ov_npu", "target": "ov_npu", "search_strategy": false, "pipeline": ["optimum_convert_npu", "io_update", "encapsulation_npu"], "output_dir": "npu_output" }
  }
}
```

---

## 4. Low Level Details

This section covers the entry-layer orchestration behavior.

- The raw config is expanded before any build executes.
- Every expanded config is validated with the existing `RunConfig`; if any build is invalid, none run.
- All builds run concurrently through the existing workflow entry point. Results retain config order; runtime failures
  are reported with the failing build names after the concurrently running builds finish.
- Parallel builds must use non-overlapping artifact and cache directories. A build's artifacts also cannot overlap
  another build's cache.
- Builds that use the same Docker image prepare that image before execution, run their containers concurrently, and
  clean the shared image once after all users finish.
- Olive cache lookups and Python-environment subprocesses receive a per-build cache context without racing on the
  process-global `OLIVE_CACHE_DIR`.
- Local builds share one Python process, so pass implementations that temporarily change process-wide state must
  synchronize that critical section. OpenVINO Optimum conversion serializes its temporary-directory override.
- Component discovery is part of entry expansion, so resolver dependencies such as `mobius-ai` must be available in
  the calling environment before a Docker or other remote host is dispatched.
- Each build receives `workflow_id = "{base_workflow_id}_{build_name}"`, which also namespaces its local cache.
- `--list_required_packages` unions the dependencies calculated from all expanded configurations.
- Results are returned as `dict[str, WorkflowOutput]` keyed by build name.
- `olive run --output_path` and `--test` are rejected for multi-build configs; each build owns its output directory,
  and discrepancy testing currently expects one workflow output.
- Build DAGs, build-to-build inputs, and parallel scheduling are out of scope for the initial implementation.

## 5. Open Questions


- Should the YAML registry (Option B) be hand-authored, generated from Mobius, or both (generated then user-overridable)?
- Should component resolution run for every HfModel/DiffusersModel, or only when a build references `components`?
- After per-component optimization, what is the cleanest way to assemble the optimized weights into a single model that `capture-onnx-graph --use_mobius_builder` can consume (merged checkpoint folder vs. in-place weight swap)?
- For diffusion, is per-component sibling output sufficient, or is a final "collect into one package" export also wanted?
