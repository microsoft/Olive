# How to convert a Phi model with a quick `--test` smoke check

If you are converting a large language model, it is often useful to validate the Olive command, environment, and conversion recipe on a much smaller model before spending time on the full checkpoint.

The `--test` option does that for Hugging Face models. Olive keeps the same model architecture, reduces it to a random 2-layer test model, saves it to the folder you provide, and reuses that folder on later runs.

This example uses [`microsoft/Phi-3.5-mini-instruct`](https://huggingface.co/microsoft/Phi-3.5-mini-instruct), but the same pattern works for other supported Hugging Face LLMs.

## Step 1: run a fast smoke test

Start with a lightweight conversion pass that uses `--test` to create and reuse a reduced Phi model.

```bash
olive optimize \
    --model_name_or_path microsoft/Phi-3.5-mini-instruct \
    --device cpu \
    --provider CPUExecutionProvider \
    --precision int4 \
    --test out/phi-test-model \
    --output_path out/phi-smoke
```

What this does:

- `--test out/phi-test-model` creates a reduced random Phi model and saves it in `out/phi-test-model`
- later runs reuse the same saved test model instead of recreating it
- `--output_path out/phi-smoke` stores the converted ONNX artifacts from the smoke test

This is a quick way to confirm that:

- Olive can load the source model
- the selected optimization recipe is valid for your setup
- the conversion path completes before you run the full model

If you only want to inspect the generated workflow first, add `--dry_run`:

```bash
olive optimize \
    --model_name_or_path microsoft/Phi-3.5-mini-instruct \
    --device cpu \
    --provider CPUExecutionProvider \
    --precision int4 \
    --test out/phi-test-model \
    --dry_run \
    --output_path out/phi-smoke
```

The generated `config.json` will include both `test_model_config` and `test_model_path`, so the same reduced model can be reused later.

## Step 2: run the full conversion

Once the smoke test succeeds, rerun the conversion on the full Phi checkpoint by removing `--test`.

```bash
olive optimize \
    --model_name_or_path microsoft/Phi-3.5-mini-instruct \
    --device cpu \
    --provider CPUExecutionProvider \
    --precision int4 \
    --output_path out/phi-full
```

At this point you know the Olive command and the conversion recipe already worked on the lightweight test model, so you can focus on the full-model run instead of debugging both at once.

## Why keep the test model folder?

The saved test model is useful beyond the first smoke test:

- you can rerun the reduced conversion quickly while iterating on options
- you can reuse the same HF test model later when comparing the Hugging Face model against the exported ONNX model
- you avoid recreating a new random test checkpoint every time

## Related docs

- [How to use the `olive optimize` command to optimize a Pytorch model](cli-optimize)
- [How to write a new workflow from scratch](../configure-workflows/build-workflow)
- [CLI reference](../../reference/cli)
