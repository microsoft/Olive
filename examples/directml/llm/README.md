# LLM Optimization with DirectML <!-- omit in toc -->

This sample shows how to optimize many LLMs to run with ONNX Runtime and DirectML.

In order to make LLMs run fast on ONNX Runtime and DirectML, the implementation of those LLMs is slightly different from what can be found in HuggingFace's transformers repository or other LLM implementations. When creating a session for the DirectML execution provider, it's important to override a few dimensions like the following:

```python
llm_session_options = onnxruntime.SessionOptions()
llm_session_options.add_free_dimension_override_by_name("batch_size", 1)
llm_session_options.add_free_dimension_override_by_name("max_seq_len", 2048)
llm_session_options.add_free_dimension_override_by_name("seq_len_increment", 1)
llm_session = onnxruntime.InferenceSession(
    "decoder_model_merged.onnx",
    sess_options=llm_session_options,
    providers=["DmlExecutionProvider"],
)
```

`batch_size` and `max_seq_len` can be changed depending on your needs, but `seq_len_increment` should be overridden to `1` in most cases.

Also, the model is comprised of 2 subgraphs joined by an `If` node: `decoder_with_past` and `decoder`. This allows us to use a model with a variable sequence length for the first token, but then use the model with a fixed sequence length of 1 for subsequent inferences. Again, this is necessary to allow DirectML to re-use compiled kernels as much as possible.

**Contents**:
- [Setup](#setup)
- [Conversion to ONNX and Latency Optimization](#conversion-to-onnx-and-latency-optimization)

# Setup

Olive is currently under pre-release, with constant updates and improvements to the functions and usage. This sample code will be frequently updated as Olive evolves, so it is important to install Olive from source when checking out this code from the main branch. See the [README for examples](https://github.com/microsoft/Olive/blob/main/examples/README.md#important) for detailed instructions on how to do this.

1. Install Olive

```
pip install -e .
```

2. Install the requirements

```
cd olive/examples/directml/llm
pip install -r requirements.txt
```

3. (Only for LLaMA 2) Request access to the LLaMA 2 weights at the HuggingFace's [llama-2-7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) or [llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b-hf) repositories.


# Conversion to ONNX and Latency Optimization

The easiest way to optimize the pipeline is with the `llm.py` helper script:

1. To convert LLaMA 2:

```
python llm.py --model_type=llama-2-7b-chat
```

2. To convert Mistral:

```
python llm.py --model_type=mistral-7b-chat
```

The first time this script is invoked can take some time since it will need to download the model weights from HuggingFace. The LLMs can be very large and the optimization process is resource intensive (it can easily take more than 200GB of ram). You can still optimize the model on a machine with less memory, but you'd have to increase your paging file size accordingly. **If the optimization process aborts without an error message or tells you that python stopped working, it is likely due to an OOM error and you should increase your paging file size.**

Once the script successfully completes, the optimized ONNX pipeline will be stored under `models/optimized/<model_name>`.

Re-running the script with `--optimize` will delete the output models, but it will *not* delete the Olive cache. Subsequent runs will complete much faster since it will simply be copying previously optimized models. If an error happens during the conversion or optimization steps, make sure to delete the `footprints` and `cache` directory and try again.

If you only want to run the inference sample (possible after the model has been optimized), run the `run_llm_io_binding.py` helper script:

```
python run_llm_io_binding.py --model_dir=<path/to/the/model/directory> --prompt=<any_prompt_you_choose>
```
