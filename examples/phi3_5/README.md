# Phi-3.5 Model Optimization

This repository demonstrates the optimization of the [Microsoft Phi-3.5 Mini Instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) model using **post-training quantization (PTQ)** techniques. The optimization process is divided into two main workflows:

1. [**QDQ Model with 4-bit Weights & 16-bit Activations**](#qdq-model-with-4-bit-weights--16-bit-activations)
    - [Optimization Process](#optimization-process)
    - [Handling Dynamic and Static Input Shapes](#handling-dynamic-and-static-input-shapes)
    - [Usage](#usage)

2. [**PTQ + AOT Compilation for Qualcomm NPUs using QNN EP**](#ptq--aot-compilation-for-qualcomm-npus-using-qnn-ep)
    - [Resource Optimization Strategy](#resource-optimization-strategy)
    - [Compilation for Qualcomm NPU Deployment](#compilation-for-qualcomm-npu-deployment)
    - [Usage](#usage-1)
    - [Inference](#inference)

## **QDQ Model with 4-bit Weights & 16-bit Activations**

This workflow produces an ONNX QDQ model that is agnostic to the target hardware and accelerator, making it suitable for general inference.

### **Optimization Process**

The model is optimized using **weight-only quantization** and **activation quantization** for efficient deployment. The process includes:

1. **Weight Rotation ([QuaRot](https://arxiv.org/abs/2404.00456))**
   - Reduces outliers from weights and hidden states to enhance quantization efficiency.

2. **4-bit Per-Channel Symmetric Quantization ([GPTQ](https://arxiv.org/abs/2210.17323))**
   - Reduces transformer layer size while preserving accuracy.

3. **ONNX Graph Capture**
   - Exports the model to ONNX for further optimization.

4. **4-bit Block-wise Quantization**
   - Applies weight-only quantization to the **embedding layer** and **language modeling head**.

5. **16-bit Activation Quantization**
   - Uses 16-bit activations to balance precision and efficiency.

The final output is a **QDQ model** with **4-bit weights** and **16-bit activations**. This model also leverages [GroupQueryAttention (GQA)](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.GroupQueryAttention) for efficient long-context processing and long-sequence generation.

### **Handling Dynamic and Static Input Shapes**

NPUs require **precompiled graphs**, meaning the model must use **static input shapes**. However, **text generation** involves two distinct processing stages:

- **Prefill (Prompt Processing)**: Processes multiple tokens simultaneously.
- **Token Generation (Iteration)**: Processes one token at a time.

To support both efficiently, we create **two model instances**:
1. **Prefill model**: Optimized for batch processing.
2. **Token generation model**: Optimized for one-token-at-a-time inference.

### **Usage**

#### **Quantization Python Environment Setup**
Quantization is resource-intensive and requires GPU acceleration. In an [x64 Python environment with Olive installed](../README.md#important), install the required packages:

```bash
# Install common dependencies
pip install -r requirements.txt

# Install ONNX Runtime GPU packages
pip install "onnxruntime-gpu>=1.21.0" "onnxruntime-genai-cuda>=0.6.0"

# AutoGPTQ: Install from source (stable package may be slow for weight packing)
# Disable CUDA extension build (not required)
# Linux
export BUILD_CUDA_EXT=0
# Windows
# set BUILD_CUDA_EXT=0

# Install AutoGPTQ from source
pip install --no-build-isolation git+https://github.com/PanQiWei/AutoGPTQ.git
```

#### **Run the Quantization Config**

```bash
olive run --config qdq_config.json
```

✅ Optimized model saved in: `models/phi3_5_qdq/`


## **PTQ + AOT Compilation for Qualcomm NPUs using QNN EP**

This process extends the [**QDQ Model with 4-bit Weights & 16-bit Activations**](#qdq-model-with-4-bit-weights--16-bit-activations) by compiling it specifically for **Qualcomm NPUs** using the **QNN Execution Provider**.

### **Resource Optimization Strategy**

To maximize efficiency while supporting dynamic input handling:

- **Embedding Layer & Language Model Head** → Executed on CPU (handles dynamic input).
- **Transformer Layers** → Executed on NPU (requires static input shapes).
- **Weight Sharing** → Prefill & token generation models reuse weights to minimize memory usage.

> ⚠️ **Note:** GQA is an ONNX Runtime *contrib operator* and must be executed on the CPU. The model graph is partitioned into **CPU (GQA nodes)** and **NPU (other nodes)** for execution.

### **Compilation for Qualcomm NPU Deployment**

Once optimized, the model is compiled for Qualcomm NPUs using **ONNX Runtime QNNExecutionProvider**. The steps include:

1. **Split the Quantized Model** → Divide into three parts:
   - **Embedding Layer**
   - **Transformer Layers**
   - **Language Model Head**
2. **Set Static Input Shapes**:
   - **(1, 64)** for prefill (batch size, sequence length).
   - **(1, 1)** for token generation.
3. **Compile using QNNExecutionProvider**:
   - Leverages **weight sharing** across the prefill and token generation models.

### **Usage**
This workflow is configured using the [`qnn_config.json`](qnn_config.json) file. It contains all of the quantization and compilation steps. It is a standalone config and is run independently of the `qdq_config.json` workflow. It requires two separate Python environments described below.

#### Quantization Python Environment Setup
Follow the steps outlined in the [previous section](#quantization-python-environment-setup) to set up the environment.

> ⚠️ Only set up the environment and install the packages. Do not run the `olive run` command at this point.

#### AOT Compilation Python Environment Setup
Model compilation using QNN Execution Provider requires a Python environment with onnxruntime-qnn installed. In a separate Python environment with Olive installed, install the required packages:

```bash
# Install ONNX Runtime QNN
pip install -r https://raw.githubusercontent.com/microsoft/onnxruntime/refs/heads/main/requirements.txt
pip install -U --pre --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple onnxruntime-qnn --no-deps
```

Replace `/path/to/qnn/env/bin` in [qnn_config.json](qnn_config.json) with the path to the directory containing your QNN environment's Python executable. This path can be found by running the following command in the environment:

```bash
# Linux
command -v python
# Windows
# where python
```

This command will return the path to the Python executable. Set the parent directory of the executable as the `/path/to/qnn/env/bin` in the config file.

#### **Run the Quantization + Compilation Config**
Activate the **Quantization Python Environment** and run the workflow:

```bash
olive run --config qnn_config.json
```

Olive will run the AOT compilation step in the **AOT Compilation Python Environment** specified in the config file using a subprocess. All other steps will run in the **Quantization Python Environment** natively.

✅ Optimized model saved in: `models/phi3_5_qnn/`

> ⚠️ If optimization fails during context binary generation, rerun the command. The process will resume from the last completed step.

### **Inference**

The optimized model can be used for inference using ONNX Runtime QNNExecutionProvider and ONNX Runtime GenAI. **Inference must be run on a Windows Copilot+ PC with a Qualcomm NPU.**

#### **Install Dependencies**
1. Visual Studio Build Tools with the following individual components
   1. Compilers, build tools, and runtimes
      * C++ Universal Windows Platform support for v143 build tools (ARM64/ARM64EC)
      * MSVC v143 - VS 2022 C++ ARM64/ARM64EC build tools (Latest)
   2. SDKs, libraries, and frameworks
      * C++ ATL for latest v143 build tools (ARM64/ARM64EC)
      * C++ MFC for latest v143 build tools (ARM64/ARM64EC)

2. Python ARM64 (version 3.11 or 3.12)
3. NPU driver version 30.0.140.1000 or higher

#### **Install Required Python Packages**

Open ARM64 Native Tools Command Prompt for VS2022 and run the following commands:

```bash
pip install -r https://raw.githubusercontent.com/microsoft/onnxruntime/refs/heads/main/requirements.txt
pip install -U --pre --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple onnxruntime-qnn --no-deps
pip install "onnxruntime-genai>=0.7.0rc2"
```

#### **Run Console-Based Chat Interface**

Open ARM64 Native Tools Command Prompt for VS2022 and execute the provided [`app.py`](app.py) script:

```bash
python app.py
```
