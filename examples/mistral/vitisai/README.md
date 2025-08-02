# Model Optimization and Quantization for AMD NPU
This folder contains sample Olive configuration to optimize Mistral models for AMD NPU.

## ✅ Supported Models and Configs

| Model Name (Hugging Face)                         | Config File Name                  |
|:--------------------------------------------------|:----------------------------------|
| `mistralai/Mistral-7B-Instruct-v0.2`              | `Mistral-7B-Instruct-v0.2_quark_vitisai_llm.json`  |
| `mistralai/Mistral-7B-Instruct-v0.3`              | `Mistral-7B-Instruct-v0.3_quark_vitisai_llm.json`  |

## **Run the Quantization Config**

### **Quark quantization**

For LLMs - follow the below commands to generate the optimized model for VitisAI Execution Provider.

**Note:** We’ve tested it on Linux with ROCm and on Linux with CUDA. It is also supported on Windows with CPU, though quantization may be slower. Support for Windows with CUDA/ROCm is planned for a future release.

For more details about quark, see the [Quark Documentation](https://quark.docs.amd.com/latest/)

#### Create a Python 3.10 conda environment and run the below commands
```bash
conda create -n olive python=3.10
conda activate olive
```

```bash
cd Olive
pip install -e .
pip install -r requirements.txt
```

#### Install VitisAI LLM dependencies

```bash
cd examples/mistral/vitisai
pip install --force-reinstall -r requirements_vitisai_llm.txt

# Note: If you're running model generation on a Windows system, please uncomment the following line in requirements_vitisai_llm.txt:
# --extra-index-url=https://pypi.amd.com/simple
# model-generate==1.5.1
```
Make sure to install the correct version of PyTorch before running quantization. If using AMD GPUs, update PyTorch to use ROCm-compatible PyTorch build. For example see the below commands

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1

python -c "import torch; print(torch.cuda.is_available())" # Must return `True`
```
#### Generate optimized LLM model for VitisAI NPU
Follow the above setup instructions, then run the below command to generate the optimized LLM model for VitisAI EP

```bash
# Mistral-7B-Instruct-v0.2
olive run --config Mistral-7B-Instruct-v0.2_quark_vitisai_llm.json

# Mistral-7B-Instruct-v0.3
olive run --config Mistral-7B-Instruct-v0.3_quark_vitisai_llm.json
```

✅ Optimized model saved in: `models/Mistral-7B-Instruct-v0.2-vai/`
> **Note:** Output model is saved in `output_dir` mentioned in the json files.
