# This program will run the ONNX version of the LLM.
import argparse
import os
import time

import numpy as np
import onnxruntime
from transformers import AutoTokenizer


def run_llm_io_binding(
    model_dir: str,
    prompt: str,
    max_seq_len: int = 2048,
    max_gen_len: int = 50,
    device: str = "dml",
    device_id: int = 0,
    ignore_eos: bool = False,
) -> str:
    onnxruntime.set_default_logger_severity(3)

    execution_provider = {
        "dml": "DmlExecutionProvider",
        "cuda": "CUDAExecutionProvider",
    }[device]

    # Create the ONNX session
    providers = [
        (
            execution_provider,
            {
                "device_id": device_id,
            },
        )
    ]

    llm_session_options = onnxruntime.SessionOptions()
    llm_session_options.add_free_dimension_override_by_name("batch_size", 1)
    llm_session_options.add_free_dimension_override_by_name("max_seq_len", max_seq_len)
    llm_session_options.add_free_dimension_override_by_name("seq_len_increment", 1)

    # add all nodes of the model as outputs

    # import onnx
    # model = onnx.load(os.path.join(model_dir, "decoder_model_merged.onnx"))
    # model.graph.output.extend([onnx.ValueInfoProto(name="/model/layers.0/self_attn/Concat_4_output_0")])
    # outputs = [x.name for x in model.graph.node[0].attribute[0].g.node]
    # for node in model.graph.node[0].attribute[0].g.node:
    #     print (node.name)
    #     for output in node.output:
    #         if output == "/model/layers.0/self_attn/Concat_4_output_0":
    #             model.graph.node[0].attribute[0].g.output.extend([onnx.ValueInfoProto(name=output)])
    #
    # outputs = [x.name for x in model.graph.node[0].attribute[1].g.node]
    # print (outputs)
    # for node in model.graph.node[0].attribute[1].g.node:
    #     print (node.name)
    #     for output in node.output:
    #         if output == "/model/layers.0/self_attn/Concat_4_output_0":
    #             model.graph.node[0].attribute[1].g.output.extend([onnx.ValueInfoProto(name=output)])
    #             model.graph.output.extend([onnx.ValueInfoProto(name=output)])

    # onnx.save(model, os.path.join(model_dir, "decoder_model_modified.onnx"))

    llm_session = onnxruntime.InferenceSession(
            os.path.join(model_dir, "decoder_model_merged.onnx"),
            sess_options=llm_session_options,
            providers=providers,
        )
    data_type = np.float16
    num_layers = 0
    for inputs_meta in llm_session._inputs_meta:
        if inputs_meta.name.startswith("cache.") and inputs_meta.name.endswith(".key"):
            num_layers += 1
            num_key_value_heads = inputs_meta.shape[1]
            head_dim = inputs_meta.shape[3]

    # Initialize the tokenizer and produce the initial tokens.
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # if tokenizer.chat_template is None:
    #     tokenizer.chat_template = "{% for message in messages %}{{message['content']}}{% endfor %}"

    print (model_dir)
    # tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    print (tokenizer.eos_token_id)

    inputs = tokenizer('''def print_prime(n):
        """
        Print all primes between 1 and n
        """''', return_tensors="pt", return_attention_mask=False)["input_ids"]

    tokens = inputs.numpy()
    tokens = onnxruntime.OrtValue.ortvalue_from_numpy(tokens, device)
    tokens_increment = onnxruntime.OrtValue.ortvalue_from_shape_and_type((1, 1), np.int64, device)

    past_seq_len = 0
    seq_len = tokens.shape()[1]

    # Create the LLM model's I/O binding
    llm_io_binding = llm_session.io_binding()

    # Create the K and V caches.
    cache_shape = (1, num_key_value_heads, max_seq_len, head_dim)
    initial_cache = np.zeros(cache_shape, dtype=data_type)
    k_caches = []
    v_caches = []

    for _ in range(num_layers):
        k_caches.append(onnxruntime.OrtValue.ortvalue_from_numpy(initial_cache, device))
        v_caches.append(onnxruntime.OrtValue.ortvalue_from_numpy(initial_cache, device))

    llm_io_binding.bind_cpu_input("use_cache_branch", np.zeros([1], dtype=np.bool_))
    llm_io_binding.bind_output("logits", device)
    llm_io_binding.bind_ortvalue_input("tokens", tokens)
    llm_io_binding.bind_ortvalue_input("tokens_increment", tokens_increment)

    before_time = time.perf_counter()

    # Iteratively generate tokens.
    output_tokens = []
    for idx in range(max_gen_len):
        if idx == 0:
            position_ids = np.arange(seq_len, dtype=np.int64).reshape((1, seq_len))
            llm_io_binding.bind_cpu_input("position_ids", position_ids)
        else:
            position_ids_increment = np.array(seq_len - 1, dtype=np.int64, ndmin=2)
            llm_io_binding.bind_cpu_input("position_ids_increment", position_ids_increment)

        seqlens_k = np.array(past_seq_len, dtype=np.int32, ndmin=1)
        llm_io_binding.bind_cpu_input("seqlens_k", seqlens_k)


        for layer_idx in range(num_layers):
            llm_io_binding.bind_ortvalue_input(f"cache.{layer_idx}.key", k_caches[layer_idx])
            llm_io_binding.bind_ortvalue_input(f"cache.{layer_idx}.value", v_caches[layer_idx])
            llm_io_binding.bind_ortvalue_output(f"cache_out.{layer_idx}.key", k_caches[layer_idx])
            llm_io_binding.bind_ortvalue_output(f"cache_out.{layer_idx}.value", v_caches[layer_idx])

        # Run the LLM
        llm_session.run_with_iobinding(llm_io_binding)

        # Decide the next token using your preferred sampling strategy.
        logits = llm_io_binding.get_outputs()[0].numpy()[:, -1, :]

        # print (logits)
        if(np.isnan(logits).any()):
            print (idx)
            print (logits)
            exit()

        # for layer_idx in range(num_layers):
        #     print (f"cache.{layer_idx}.key")
        #     print (k_caches[layer_idx].numpy().shape)
        #     print (k_caches[layer_idx].numpy()[:, :, seq_len-1, :])
        #     print (f"cache.{layer_idx}.value")
        #     print (v_caches[layer_idx].numpy().shape)
        #     print (v_caches[layer_idx].numpy()[:, :, seq_len-1, :])

        next_token = np.argmax(logits, axis=-1, keepdims=True)
        output_tokens.append(next_token.item())

        # Set the token for the next iteration
        llm_io_binding.bind_cpu_input("tokens_increment", next_token)

        # Stop if/when we get an ENDOFTEXT token before reaching maximum sequence length
        if not ignore_eos and output_tokens[-1] == tokenizer.eos_token_id:
            break

        if idx == 0:
            llm_io_binding.bind_cpu_input("use_cache_branch", np.ones([1], dtype=np.bool_))
            llm_io_binding.bind_output("logits", device)

        past_seq_len = seq_len
        seq_len += 1

    after_time = time.perf_counter()
    duration = after_time - before_time
    tokens_per_second = idx / duration

    # Only print the tokens/s when ignore_eos is provided for benchmarking purposes
    # if ignore_eos:
    print(f"Execution took {duration:0.4f} seconds (generated {tokens_per_second:0.2f} tokens per second)")

    output_str = tokenizer.decode(output_tokens, skip_special_tokens=True)
    print(output_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="What is the lightest element?")
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--max_gen_len", type=int, default=50)
    parser.add_argument("--ignore_eos", action="store_true")
    parser.add_argument("--device", type=str, choices=["dml", "cuda"], default="dml")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the folder containing the decoder_model_merged.onnx file",
    )

    args = parser.parse_args()
    run_llm_io_binding(
        args.model_dir,
        args.prompt,
        args.max_seq_len,
        args.max_gen_len,
        args.device,
        args.device_id,
        args.ignore_eos,
    )
