# Phi-3.5 Model Optimization

This repository demonstrates the optimization of the [Microsoft Phi-3.5 Mini Instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) model using **post-training quantization (PTQ)** techniques. The optimized model is deployed on **Qualcomm NPUs** (e.g., Snapdragon X Series) using the **ONNX Runtime QNNExecutionProvider**.

### **Workflows**
1. **QDQ Model with 4-bit Weights & 16-bit Activations**
   - Produces an ONNX QDQ model optimized for general inference.
   - Configured via [qdq_config.json](qdq_config.json).

2. **PTQ + AOT Compilation for Qualcomm NPUs using QNN EP**
   - Optimized for deployment on Qualcomm NPUs.
   - Configured via [qnn_config.json](qnn_config.json).


## **Table of Contents**
1. [Optimization Process](#optimization-process)
2. [Handling Dynamic and Static Input Shapes](#handling-dynamic-and-static-input-shapes)
3. [Resource Optimization Strategy](#resource-optimization-strategy)
4. [Compilation for Qualcomm NPU Deployment](#compilation-for-qualcomm-npu-deployment)
5. [Requirements](#requirements)
6. [Usage](#usage)
7. [Inference](#inference)


## **Optimization Process**

The model is optimized using **weight-only quantization** and **activation quantization** for efficient deployment. The process includes:

1. **Weight Rotation ([QuaRot](https://arxiv.org/abs/2404.00456))**
   - Prepares model weights for better quantization.

2. **4-bit Per-Channel Symmetric Quantization ([GPTQ](https://arxiv.org/abs/2210.17323))**
   - Reduces transformer layer size while preserving accuracy.

3. **ONNX Graph Capture**
   - Exports the model to ONNX for further optimization.

4. **4-bit Block-wise Quantization**
   - Applies weight-only quantization to the **embedding layer** and **language modeling head**.

5. **16-bit Activation Quantization**
   - Uses 16-bit activations to balance precision and efficiency.

The final output is a **QDQ model** with **4-bit weights** and **16-bit activations**.


## **Handling Dynamic and Static Input Shapes**

NPUs require **precompiled graphs**, meaning the model must use **static input shapes**. However, **text generation** involves two distinct processing stages:

- **Prefill (Prompt Processing)**: Processes multiple tokens simultaneously.
- **Token Generation (Iteration)**: Processes one token at a time.

To support both efficiently, we create **two model instances**:
1. **Prefill model**: Optimized for batch processing.
2. **Token generation model**: Optimized for one-token-at-a-time inference.


## **Resource Optimization Strategy**

To maximize efficiency while supporting dynamic input handling:

- **Embedding Layer & Language Model Head** → Executed on CPU (handles dynamic input).
- **Transformer Layers** → Executed on NPU (requires static input shapes).
- **Weight Sharing** → Prefill & token generation models reuse weights to minimize memory usage.

Additionally, we use **[GroupQueryAttention (GQA)](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.GroupQueryAttention)** for **efficient long-context processing and long generation**.
> ⚠️ **Note:** GQA is an ONNX Runtime *contrib operator* and must be run on the CPU. The model graph is partitioned into **CPU (GQA nodes)** and **NPU (other nodes)** for execution.


## **Compilation for Qualcomm NPU Deployment**

Once optimized, the model is compiled for Qualcomm NPUs using **ONNX Runtime QNNExecutionProvider**.

### **Compilation Steps**
1. **Split the Quantized Model** → Divide into three parts:
   - **Embedding Layer**
   - **Transformer Layers**
   - **Language Model Head**
2. **Set Static Input Shapes**:
   - **(1, 64)** for prefill (batch size, sequence length).
   - **(1, 1)** for token generation.
3. **Compile using QNNExecutionProvider**:
   - Leverages **weight sharing** across the prefill and token generation models.


## **Requirements**

### **1st x64 Python Environment (QDQ/PTQ Optimization)**
> Requires GPU resources for quantization.

```bash
# Install common dependencies
pip install -r requirements.txt

# Install ONNX Runtime GPU packages
pip install "onnxruntime-gpu>=1.21.0" "onnxruntime-genai-cuda>=0.6.0"

# AutoGPTQ: Install from source (stable package may be slow for weight packing)
# set the environment variable to disable the CUDA extension build, not required since we are not doing inference
# Linux
export BUILD_CUDA_EXT=0
# Windows
# set BUILD_CUDA_EXT=0
# Install AutoGPTQ from source
pip install --no-build-isolation git+https://github.com/PanQiWei/AutoGPTQ.git
```

### **2nd x64 Python Environment (QNN AOT Compilation)**
> Requires **ONNX Runtime QNN nightly build** for AOT compilation.

```bash
# Install common dependencies
pip install -r requirements.txt

# Install ONNX Runtime QNN
pip install -r https://raw.githubusercontent.com/microsoft/onnxruntime/refs/heads/main/requirements.txt
pip install -U --pre --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple onnxruntime-qnn --no-deps
```

**Note:** Both environments require Olive to be installed. Refer to the [main examples README](../README.md#important) for instructions on how to set up python environments and install Olive.


## **Usage**

### **1️⃣ PTQ with QDQ Format**
To generate a **QDQ model**, run the following command in the **1st Python environment**:

```bash
olive run --config qdq_config.json
```

✅ Optimized model saved in: `models/phi3_5_qdq/`


### **2️⃣ PTQ + AOT Compilation for Qualcomm NPU**
To configure AOT compilation, update the path to the **2nd Python environment** in `qnn_config.json`.

```bash
# Linux
command -v python
# Windows
# where python
```
Use the parent directory of output path to update `/path/to/qnn/env/bin` in the config file.

Then, run the following command in the **1st Python environment**:

```bash
olive run --config qnn_config.json
```

✅ Optimized model saved in: `models/phi3_5_qnn/`

> **⚠️ If optimization fails during context binary generation, simply rerun the command.** The process will resume from the last completed step.


## **Inference on Qualcomm NPU**

> **Must be run on a Windows Copilot+ PC with a Qualcomm NPU.**

### **Install Required Packages (arm64 Python)**
```bash
pip install -r https://raw.githubusercontent.com/microsoft/onnxruntime/refs/heads/main/requirements.txt
pip install -U --pre --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple onnxruntime-qnn --no-deps
pip install "onnxruntime-genai>=0.7.0rc2"
```

### **Run Console-Based Chat Interface**
Execute the provided [`app.py`](app.py) script:
```bash
python app.py
```
