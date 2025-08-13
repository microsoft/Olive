# Gemma-3-4B Model Optimization

This repository demonstrates the optimization of the [Google Gemma-3-4B](https://huggingface.co/google/gemma-3-4b-it) model using **post-training quantization (PTQ)** techniques. The optimization process utilizes an environment based heavily upon the [PTQ tutorial for Phi-3.5](https://github.com/CodeLinaro/Olive/blob/main/examples/phi3_5/README.md)

## Automated Setup (Linux Only)

Requirements:
* Python 3.10
* uv

This repository contains an automated setup script for Linux that can be used to help automate many of the steps listed in the tutorial above:

```bash
source env_setup.sh
```

## Optimization Process

Run the following command in your Olive environment after completing the above setup steps:

```bash
olive run --config gemma3-4b-qnn-config.json
```
