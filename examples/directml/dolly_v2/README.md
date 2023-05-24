# Dolly V2 Optimization with DirectML <!-- omit in toc -->

This sample shows how to optimize [Dolly V2](https://huggingface.co/databricks/dolly-v2-7b) to run with ONNX Runtime and DirectML.

Dolly V2 comprises multiple PyTorch models tied together into a *pipeline*. This Olive sample will convert each PyTorch model to ONNX via the [Optimum](https://huggingface.co/docs/optimum/onnxruntime/overview) library, and then run the converted ONNX models through the `OrtTransformersOptimization` pass. The transformer optimization pass performs several time-consuming graph transformations that make the models more efficient for inference at runtime. Finally, the models are merged into a single optimized model via the `OptimumMerging` pass that will be able to use cached results from previous sequences and use less memory. Output models are only guaranteed to be compatible with onnxruntime-directml 1.15.0 or newer.

**Contents**:
- [Setup](#setup)
- [Conversion to ONNX and Latency Optimization](#conversion-to-onnx-and-latency-optimization)

# Setup

Olive is currently under pre-release, with constant updates and improvements to the functions and usage. This sample code will be frequently updated as Olive evolves, so it is important to install Olive from source when checking out this code from the main branch. See the [README for examples](https://github.com/microsoft/Olive/blob/main/examples/README.md#important) for detailed instructions on how to do this.

**Alternatively**, you may install a stable release that we have validated. For example:

```
# Install stable release of the Olive tool
pip install olive-ai[directml]==0.2.1

# Clone Olive repo to access sample code
git clone https://github.com/microsoft/olive --branch v0.2.1
```

Once you've installed Olive, install the requirements for this sample matching the version of the library you are using:
```
cd olive/examples/directml/dolly_v2
pip install -r requirements.txt
```

# Conversion to ONNX and Latency Optimization

The easiest way to optimize the pipeline is with the `dolly_v2.py` helper script:

```
python dolly_v2.py --optimize
```

The Dolly V2 model is very large, and the optimization process is resource intensive. The optimization process can easily take more than 128GB of memory. You can still optimize the model on a machine with less memory, but you'd have to increase your paging file size accordingly and the conversion process will take significantly longer to complete (many hours).

Once the script successfully completes, the optimized ONNX pipeline will be stored under `models/optimized/databricks/dolly-v2-7b`.

Re-running the script with `--optimize` will delete the output models, but it will *not* delete the Olive cache. Subsequent runs will complete much faster since it will simply be copying previously optimized models; you may use the `--clean_cache` option to start from scratch (not typically used unless you are modifying the scripts, for example).
