# Model Optimization and Quantization for AMD NPU
This folder contains sample Olive configuration to optimize Qwen models for AMD NPU.

## ✅ Supported Models and Configs

| Model Name (Hugging Face)          | Config File Name                |
| :--------------------------------- | :------------------------------ |
| `Qwen/Qwen1.5-7B-Chat`             | `qwen1_5-7B_chat_quark_vitisai_llm.json` |
| `Qwen/Qwen2.5-0.5B-Instruct`       | `qwen2_5_0_5B_inst_quark_vitisai_llm.json` |
| `Qwen/Qwen2.5-1.5B`                | `qwen2_5_1_5B_quark_vitisai_llm.json` |
| `Qwen/Qwen2.5-7B-Instruct`         | `qwen2_5_7B_inst_quark_vitisai_llm.json` |
| `Qwen/Qwen2.5-Coder-0.5B-Instruct` | `qwen2_5_coder_0_5B_inst_quark_vitisai_llm.json` |
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | `qwen2_5_coder_1_5B_inst_quark_vitisai_llm.json` |
| `Qwen/Qwen2.5-Coder-7B-Instruct`   | `qwen2_5_coder_7B_inst_quark_vitisai_llm.json` |
| `Qwen/Qwen2-7B-Instruct`           | `qwen2_7B_inst_quark_vitisai_llm.json` |

> **Note:** Before running, update the `model_path` in the config file to match the Hugging Face model name listed above.

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
cd examples/qwen2_5/vitisai
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
# Qwen1.5-7B-Chat
olive run --config qwen1_5-7B_chat_quark_vitisai_llm.json

# Qwen2.5-0.5B-Instruct
olive run --config qwen2_5_0_5B_inst_quark_vitisai_llm.json

# Qwen2.5-1.5B
olive run --config qwen2_5_1_5B_quark_vitisai_llm.json

# Qwen2.5-7B-Instruct
olive run --config qwen2_5_7B_inst_quark_vitisai_llm.json

# Qwen2.5-Coder-0.5B-Instruct
olive run --config qwen2_5_coder_0_5B_inst_quark_vitisai_llm.json

# Qwen2.5-Coder-1.5B-Instruct
olive run --config qwen2_5_coder_1_5B_inst_quark_vitisai_llm.json

# Qwen2.5-Coder-7B-Instruct
olive run --config qwen2_5_coder_7B_inst_quark_vitisai_llm.json

# Qwen2-7B-Instruct
olive run --config qwen2_7B_inst_quark_vitisai_llm.json
```

✅ Optimized model saved in: `models/qwen2_5_0_5b_inst-vai/`

> **Note:** Output model is saved in `output_dir` mentioned in the json files.
