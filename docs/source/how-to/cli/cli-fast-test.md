# How to convert a Qwen model with a quick `--test` smoke check

If you are converting a large language model, it is often useful to validate the Olive command, environment, and conversion recipe on a much smaller model before spending time on the full checkpoint.

The `--test` option does that for Hugging Face models. Olive keeps the same model architecture, reduces it to a random 2-layer test model, saves it to the folder you provide, and reuses that folder on later runs.

This example uses [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B), but the same pattern works for other supported Hugging Face LLMs.

## Step 1: generate the workflow config

Start by generating the config that Olive will run for the Qwen conversion.

```bash
olive optimize \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --device cpu \
    --provider CPUExecutionProvider \
    --precision int4 \
    --output_path out/qwen \
    --dry_run
```

This creates `out/qwen/config.json` without launching the full conversion yet.

## Step 2: run a fast smoke test with `olive run --test`

Use the generated config with `olive run` and pass `--test` so Olive swaps in a reduced random Qwen model.

```bash
olive run \
    --config out/qwen/config.json \
    --test out/qwen-test-model \
    --output_path out/qwen-test-run
```

What this does:

- `--test out/qwen-test-model` creates a reduced random Qwen model and saves it in `out/qwen-test-model`
- later runs reuse the same saved test model instead of recreating it
- `--output_path out/qwen-test-run` gives the smoke test its own output folder, so the generated ONNX artifacts are easy to find
- Olive marks that output folder as a test-only run and refuses to reuse a non-test conversion folder for `--test`

After the smoke test finishes, look under `out/qwen-test-run` for the exported ONNX model and related files.

This is a quick way to confirm that:

- Olive can load the source model
- the selected optimization recipe is valid for your setup
- the conversion path completes before you run the full model

If you omit the folder and just pass `--test`, `olive run` will save the reduced model under `<output_path>/test_model`.

### Optional: choose which `--test` metrics to run

By default, `--test` evaluates both:

- `mae`: maximum absolute error between the ONNX and reference model outputs
- `speedup`: ONNX-vs-PyTorch latency measurement

You can select a subset with `--test_metrics`. For example, to run only speedup checks:

```bash
olive run \
    --config out/qwen/config.json \
    --test out/qwen-test-model \
    --test_metrics speedup \
    --output_path out/qwen-test-run
```

## Step 3: run the full conversion

Once the smoke test succeeds, rerun the conversion on the full Qwen checkpoint by removing `--test`.

```bash
olive run \
    --config out/qwen/config.json \
    --output_path out/qwen-full
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
