# Generate NPU model

Helper scripts to generate LLM suitable for running on RyzenAI NPU using WOQ (INT4) Model.

### Setup

In anaconda prompt/powershell, run the following commands

```sh
# create conda env
conda create -n npu-model-gen python==3.10
# activate env
conda activate npu-model-gen
# install other dependencies
pip install onnx==1.16.2 colorama
python -m pip install nanobind
```

### Generate NPU model using generator script

In the newly setup conda environment, run below sample command

```sh
python generate.py --input_model "path/to/your/model.onnx" --output_model "desired/path/for/output/model.onnx" --custom_ops --fuse_SSMLP --packed_const
```
This script first adds Cast nodes to input onnx model, pre-packs the constants and fuses the SSMLP pattern.

### All options available during NPU model generation:
- --custom_ops (generates AMD Custom ops eg: AMD GQA, SLRN, SSLRN)
- --fuse (fuses both SSMLP and GQO, requires the use of --custom_ops)
- --fuse_SSMLP (fuses only SSMLP, requires use of --custom_ops)
- --fuse_GQO (fuses only GQO, requires use of --custom_ops)
- --packed_const (generates model with packed constant replacing other MatMulNBits constants)
