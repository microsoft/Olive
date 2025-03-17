# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# copied from https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/models/phi2/inference_example.py  # noqa: E501
import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoTokenizer

pt_to_np = {
    "torch.int32": np.int32,
    "torch.int64": np.int64,
    "torch.float32": np.float32,
    "torch.float16": np.float16,
}


# flake8: noqa: T201


# TODO(jambayk): Use ORTGenerator from example utils
class ORTGenerator:
    def __init__(self, decoder_path):
        self.onnx_decoder_path = decoder_path
        self.num_heads = 32
        self.head_size = 80
        self.num_layers = 32
        self.max_sequence_length = 2048

        self.use_fp16 = False
        # -1 to use CPU
        self.device_id = -1
        self.device = None
        self.use_buffer_share = False
        self.packed_kv = False
        self.use_step = False
        self.kv_torch_dtype = None
        self.sess = None
        self.tokenizer = None

    def get_initial_inputs_and_outputs(self, encodings_dict):

        input_ids = torch.tensor(encodings_dict["input_ids"], device=self.device, dtype=torch.int32)
        attention_mask = torch.tensor(encodings_dict["attention_mask"], device=self.device, dtype=torch.int32)
        step = torch.tensor([0], device=self.device, dtype=torch.int64)

        inputs = {
            "input_ids": input_ids.contiguous(),
            "attention_mask": attention_mask.contiguous(),
        }

        if self.use_step:
            inputs["step"] = step.contiguous()

        batch_size, sequence_length = input_ids.shape

        past_seq_length = self.max_sequence_length if self.use_buffer_share else 0
        past_shape = (
            (2, batch_size, self.num_heads, past_seq_length, self.head_size)
            if self.packed_kv
            else (batch_size, self.num_heads, past_seq_length, self.head_size)
        )
        for i in range(self.num_layers):
            past = torch.zeros(past_shape, device=self.device, dtype=self.kv_torch_dtype)
            past_key_name = f"past_key_{i}"
            past_value_name = f"past_value_{i}"
            (
                inputs.update({past_key_name: past.contiguous(), past_value_name: past.clone().contiguous()})
                if not self.packed_kv
                else inputs.update({f"past_{i}": past.contiguous()})
            )

        logits = torch.zeros(batch_size, sequence_length, 51200, device=self.device, dtype=self.kv_torch_dtype)
        outputs = {"logits": logits.contiguous()}

        if not self.use_buffer_share:
            present_shape = (
                (2, batch_size, self.num_heads, sequence_length, self.head_size)
                if self.packed_kv
                else (batch_size, self.num_heads, sequence_length, self.head_size)
            )
            for i in range(self.num_layers):
                present_key_name = f"present_key_{i}"
                present_value_name = f"present_value_{i}"
                present = torch.zeros(present_shape, device=self.device, dtype=self.kv_torch_dtype)
                (
                    outputs.update({present_key_name: present.contiguous(), present_value_name: present.contiguous()})
                    if not self.packed_kv
                    else outputs.update({f"present_{i}": present.contiguous()})
                )

        return inputs, outputs

    def apply_io_binding(self, model: ort.InferenceSession, inputs: dict, outputs: dict):
        io_binding = model.io_binding()
        device = None

        for k, v in inputs.items():
            io_binding.bind_input(
                name=k,
                device_type=v.device.type,
                device_id=0 if v.device.type == "cpu" else v.device.index,
                element_type=pt_to_np[repr(v.dtype)],
                shape=tuple(v.shape),
                buffer_ptr=v.data_ptr(),
            )
            device = v.device

        for output in model.get_outputs():
            name = output.name
            if self.use_buffer_share and "present" in name:
                v = inputs[name.replace("present", "past")]
                io_binding.bind_output(
                    name=name,
                    device_type=v.device.type,
                    device_id=0 if v.device.type == "cpu" else v.device.index,
                    element_type=(np.float16 if self.use_fp16 else np.float32),
                    shape=tuple(v.shape),
                    buffer_ptr=v.data_ptr(),
                )
            else:
                v = outputs[name]
                io_binding.bind_output(
                    name=name,
                    device_type=device.type,
                    device_id=0 if device.type == "cpu" else device.index,
                    element_type=(np.float16 if self.use_fp16 else np.float32),
                    shape=tuple(v.shape),
                    buffer_ptr=v.data_ptr(),
                )

        return io_binding

    def create_session(
        self,
        device_id,
        use_fp16=True,
        use_buffer_share=True,
        packed_kv=False,
        use_step=False,
        delay_ort_session_init=False,
    ):
        sess_options = ort.SessionOptions()
        self.kv_torch_dtype = torch.float16 if use_fp16 else torch.float32
        ep = "CUDAExecutionProvider" if device_id >= 0 else "CPUExecutionProvider"
        if not delay_ort_session_init:
            self.sess = ort.InferenceSession(self.onnx_decoder_path, sess_options=sess_options, providers=[ep])

        self.device_id = device_id
        self.device = torch.device("cuda:" + str(device_id) if device_id >= 0 else "cpu")
        self.use_fp16 = use_fp16
        self.use_buffer_share = use_buffer_share
        self.packed_kv = packed_kv
        self.use_step = use_step

        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/phi-2",
            trust_remote_code=True,
            padding_side="left",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt, max_length):
        encodings_dict = self.tokenizer.batch_encode_plus(prompt, padding=True)

        inputs, outputs = self.get_initial_inputs_and_outputs(encodings_dict)

        all_token_ids = inputs["input_ids"].clone()
        batch_size, sequence_length = all_token_ids.shape

        current_length = sequence_length
        has_eos = torch.zeros(batch_size, device=self.device, dtype=torch.bool)

        while current_length < max_length:
            io_binding = self.apply_io_binding(self.sess, inputs, outputs)

            io_binding.synchronize_inputs()
            self.sess.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()

            # Sample with argmax (greedy search)
            next_token_logits = outputs["logits"][:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1)

            # Check if we previously reached EOS token id or if generated token id is EOS token id
            has_eos = has_eos | next_tokens == self.tokenizer.eos_token_id

            # Determine which new tokens to add to list of all token ids
            # Add EOS token ids for batch entries that ended early
            # (ragged batching scenario where some batch entries ended early and some haven't)
            tokens_to_add = next_tokens.masked_fill(has_eos, self.tokenizer.eos_token_id).reshape([batch_size, 1])
            all_token_ids = torch.cat([all_token_ids, tokens_to_add], dim=-1)

            # Return early if all batch entries have reached EOS token id
            if torch.all(has_eos):
                break

            # Update inputs for next inference run
            current_length += 1
            # inputs["input_ids"] = torch.cat([inputs["input_ids"], tokens_to_add], 1).to(torch.int32)
            inputs["input_ids"] = tokens_to_add.to(torch.int32)
            if self.use_step:
                inputs["step"] = torch.tensor([current_length - 1], device=self.device, dtype=torch.int64)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], (~has_eos).reshape(batch_size, 1)], 1).to(
                torch.int32
            )

            # Set logits to zeros for next inference run and re-use memory buffer
            if outputs["logits"].shape[1] != 1:
                outputs["logits"] = outputs["logits"][:, :1, :].contiguous()
            outputs["logits"].zero_()

            if not self.use_buffer_share:
                for i in range(self.num_layers):
                    past_key_name = f"past_key_{i}"
                    past_value_name = f"past_value_{i}"
                    present_key_name = f"present_key_{i}"
                    present_value_name = f"present_value_{i}"
                    if not self.packed_kv:
                        inputs[past_key_name] = outputs[present_key_name]
                        inputs[past_value_name] = outputs[present_value_name]
                    else:
                        inputs[f"past_{i}"] = outputs[f"present_{i}"]

                new_sequence_length = inputs["attention_mask"].shape[1]
                present_shape = (
                    (2, batch_size, self.num_heads, new_sequence_length, self.head_size)
                    if self.packed_kv
                    else (batch_size, self.num_heads, new_sequence_length, self.head_size)
                )
                for i in range(self.num_layers):
                    present_key_name = f"present_key_{i}"
                    present_value_name = f"present_value_{i}"
                    present = torch.zeros(present_shape, device=self.device, dtype=self.kv_torch_dtype)
                    (
                        outputs.update(
                            {present_key_name: present.contiguous(), present_value_name: present.clone().contiguous()}
                        )
                        if not self.packed_kv
                        else outputs.update({f"present_{i}": present.contiguous()})
                    )

        return self.tokenizer.batch_decode(all_token_ids, skip_special_tokens=True)

    def optimum_generate(self, prompt, max_length):
        from pathlib import Path

        from optimum.onnxruntime import ORTModelForCausalLM
        from optimum.utils.save_utils import maybe_save_preprocessors
        from transformers import AutoConfig

        output_model_path = Path(self.onnx_decoder_path)
        model_id = "microsoft/phi-2"
        maybe_save_preprocessors(model_id, output_model_path.parent, trust_remote_code=True)
        AutoConfig.from_pretrained(model_id).save_pretrained(output_model_path.parent)

        model = ORTModelForCausalLM.from_pretrained(
            output_model_path.parent,
            provider="CUDAExecutionProvider" if self.device_id >= 0 else "CPUExecutionProvider",
            use_io_binding=self.device_id >= 0,
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        outputs = model.generate(**inputs, max_length=max_length)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


def genai_run(prompts, model_path, max_length=200):
    import time

    import onnxruntime_genai as og

    print("Loading model...")
    app_started_timestamp = time.time()
    model = og.Model(model_path)
    model_loaded_timestamp = time.time()
    print("Model loaded in {:.2f} seconds".format(model_loaded_timestamp - app_started_timestamp))
    tokenizer = og.Tokenizer(model)

    input_tokens = tokenizer.encode_batch(prompts)

    print("Creating generator ...")
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=max_length)
    generator = og.Generator(model, params)
    generator.append_tokens(input_tokens)

    print("Generating tokens ...")
    start_time = time.time()
    while not generator.is_done():
        generator.generate_next_token()
    run_time = time.time() - start_time

    print("Decoding generated tokens ...")
    output_token_count = 0

    for i, prompt in enumerate(prompts):
        print(f"Prompt #{i+1:02d}: {prompt}")
        print(tokenizer.decode(generator.get_sequence(i)))

    output_token_count = sum(len(generator.get_sequence(i)) for i in range(len(prompts)))
    print(f"Tokens: {output_token_count}, Time: {run_time:.2f}, Tokens per second: {output_token_count / run_time:.2f}")


def run(
    prompt,
    onnx_model_path,
    use_buffer_share,
    device_id,
    packed_kv=False,
    use_fp16=True,
    use_step=False,
    use_optimum=False,
    max_length=200,
):
    generator = ORTGenerator(onnx_model_path)
    generator.create_session(device_id, use_fp16, use_buffer_share, packed_kv, use_step, use_optimum)
    if not use_optimum:
        texts = generator.generate(prompt, max_length=max_length)
    else:
        texts = generator.optimum_generate(prompt, max_length=max_length)

    for i, text in enumerate(texts):
        print(f"Prompt: {prompt[i]}")
        yield text
