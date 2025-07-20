# Stable Diffusion Quantization

This folder includes examples running quantization for stable diffusion models using Quark with Brevitas integration. This is adapted from the original [Brevitas Stable Diffusion examples](https://github.com/Xilinx/brevitas/tree/dev/src/brevitas_examples/stable_diffusion).

## Requirements

`dev` branch of Brevitas is required for this example.

```bash
# First, git clone Brevitas
git clone https://github.com/Xilinx/brevitas

# Checkout dev branch
cd brevitas
git checkout dev

# Build from source
pip install -e .[export]
```

## Example

This example applies PTQ quantization on SDXL model. It applies activation equalization and activation calibration.

```bash
python main_quark.py --resolution 1024 --batch 1 --model stabilityai/stable-diffusion-xl-base-1.0 --conv-weight-bit-width 8 --linear-weight-bit-width 8 --dtype float16 --weight-quant-type sym  --calibration-steps 8 --guidance-scale 8. --use-negative-prompts --activation-eq
```

<!--
## License
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
-->
