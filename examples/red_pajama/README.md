# Red Pajama Optimization with Optimum <!-- omit in toc -->

This sample shows how to export and optimize [Red Pajama](https://huggingface.co/togethercomputer/RedPajama-INCITE-Base-3B-v1) language model to run inferencing with ONNX Runtime.

Red Pajama comprises multiple PyTorch models (including both encoder and decoder) tied together into a *pipeline*. This Olive sample exports each such embedded PyTorch model to ONNX via the [Optimum](https://huggingface.co/docs/optimum/onnxruntime/overview) library, and then run the exported ONNX models through the `OrtTransformersOptimization` pass. The transformer optimization pass performs several time-consuming graph transformations that make the models more efficient for inference at runtime. Finally, all the models are merged into a single optimized model via the `OptimumMerging` pass that will be able to use cached results from previous sequences and consume less memory.

**Contents**:
- [Setup](#setup)
- [Conversion to ONNX and Latency Optimization](#conversion-to-onnx-and-latency-optimization)

# Setup

Olive is currently under pre-release, with constant updates and improvements to the functions and usage. This sample code will be frequently updated as Olive evolves, so it is important to install Olive from source when checking out this code from the main branch. See the [README for examples](https://github.com/microsoft/Olive/blob/main/examples/README.md#important) for detailed instructions on how to do this.

**Alternatively**, you may install a stable release that we have validated. For example:
```
# Install stable release of the Olive tool
pip install olive-ai[gpu]==0.2.1

# Clone Olive repo to access sample code
git clone https://github.com/microsoft/olive --branch v0.2.1
```
Once you've installed Olive, install the requirements for this sample matching the version of the library you are using:
```
cd olive/examples/red_pajama
pip install -r requirements.txt
```

# Conversion to ONNX and Latency Optimization

The easiest way to optimize is by running the olive pipeline for the example:
```
python -m olive.workflows.run --config config.json
```
The Red Pajama model is very large, and the optimization process is resource intensive. To run the process on low-end machines, you can increase your paging file size accordingly. Be warned, with the overhead of paging, the conversion process might take significantly longer to complete.

Once the script successfully completes, the optimized ONNX pipeline will be stored under `models/red_pajama`.

Recurrent runs will delete the output models, but will *not* delete the Olive cache. Subsequent runs will complete faster since it will simply be copying previously optimized models; you may use the `--clean_cache` option to start from scratch (not typically used unless you are modifying the scripts, for example).
