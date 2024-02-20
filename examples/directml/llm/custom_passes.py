# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
from pathlib import Path

import onnx
from onnx import ModelProto


def create_cache_model(model_path: str):
    model = onnx.load(model_path)

    for _, graph_input in enumerate(model.graph.input):
        if graph_input.name in {"tokens", "position_ids"}:
            if graph_input.type.tensor_type.shape.dim[1].dim_param != "seq_len":
                raise ValueError(f"Expected {graph_input.name} second axis to be dynamic and named `seq_len`.")
            graph_input.type.tensor_type.shape.dim[1].dim_param = "seq_len_increment"

            # Update the name in the model's graph
            new_input_name = f"{graph_input.name}_increment"
            for node in model.graph.node:
                for i, input_name in enumerate(node.input):
                    if input_name == graph_input.name:
                        node.input[i] = new_input_name

            # Finally, rename the input itself
            graph_input.name = new_input_name

    model_with_past_name = f"{Path(model_path).stem}_with_past.onnx"
    save_path = Path(model_path).with_name(model_with_past_name).as_posix()

    external_file_name = os.path.basename(save_path) + "_data"
    external_path = os.path.join(os.path.dirname(save_path), external_file_name)

    if Path(save_path).is_file():
        os.remove(save_path)
    if Path(external_path).is_file():
        os.remove(external_path)

    onnx.save(
        model,
        save_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_file_name,
    )

    return save_path


def merge_models(model_path_1: str, model_path_2: str):
    save_path = Path(model_path_1).with_name("decoder_model_merged.onnx").as_posix()

    # TODO(trajep): Remove this when the bug in Optimum is fixed. Optimum calls ByteSize() to see whether
    # it should be using the merged model directly or use the path instead in the model checker,
    # but unfortunately ByteSize() doesn't seem to be working correctly with external weights.
    # https://github.com/huggingface/optimum/issues/1044
    def new_byte_size_func(_):
        return 2147483648

    prev_byte_size_func = ModelProto.ByteSize
    try:
        ModelProto.ByteSize = new_byte_size_func

        from optimum.onnx import merge_decoders

        merge_decoders(model_path_1, model_path_2, save_path=save_path)
    finally:
        ModelProto.ByteSize = prev_byte_size_func

    return save_path
