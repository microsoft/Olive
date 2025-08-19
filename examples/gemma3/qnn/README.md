# Gemma-3-4B Model Optimization

This repository demonstrates the optimization of the [Google Gemma-3-4B](https://huggingface.co/google/gemma-3-4b-it) model using **post-training quantization (PTQ)** techniques. The optimization process utilizes an environment based heavily upon the [PTQ tutorial for Phi-3.5](https://github.com/CodeLinaro/Olive/blob/main/examples/phi3_5/README.md)

## Automated Setup (Linux Only)

Requirements:
* Python 3.10
* uv - Used throughout the setup scripts, please follow the [publically available installation instructions](https://docs.astral.sh/uv/getting-started/installation/#installation-methods)

This repository contains an automated setup script for Linux that can be used to help automate many of the steps listed in the Phi-3.5 tutorial above:

```bash
source env_setup.sh
```

## Optimization Process

Since Gemma-3-4B is a multi-modal model composed of both vision and text components, the strategy for optimizing it through Olive is to operate on the constituent models before configuring them to work in concert at the onnxruntime-genai stage.

Thus, the following commands should be used to separately produce context binaries for the text and vision portions of the model, respectively.

```bash
olive run --config gemma3-4b-text-qnn-config.json
```

```bash
olive run --config gemma3-4b-vision-qnn-config.json
```
