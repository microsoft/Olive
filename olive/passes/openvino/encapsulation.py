# -------------------------------------------------------------------------
# Copyright (c) Intel Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
from pathlib import Path
from typing import ClassVar, Union

import onnx.helper as helper
from onnx import TensorProto, save

from olive.common.utils import hardlink_copy_dir, hardlink_copy_file
from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.model import ONNXModelHandler, OpenVINOModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam


class OpenVINOEncapsulation(Pass):
    """Encapsulates OpenVINO models with onnx context nodes."""

    openvino_to_onnx_dtype: ClassVar[dict] = {
        "f32": TensorProto.FLOAT,
        "float32": TensorProto.FLOAT,
        "f64": TensorProto.DOUBLE,
        "float64": TensorProto.DOUBLE,
        "f16": TensorProto.FLOAT16,
        "bf16": TensorProto.BFLOAT16,
        "i8": TensorProto.INT8,
        "int8_t": TensorProto.INT8,
        "i16": TensorProto.INT16,
        "int16_t": TensorProto.INT16,
        "i32": TensorProto.INT32,
        "int32_t": TensorProto.INT32,
        "i64": TensorProto.INT64,
        "int64_t": TensorProto.INT64,
        "u8": TensorProto.UINT8,
        "uint8_t": TensorProto.UINT8,
        "u16": TensorProto.UINT16,
        "uint16_t": TensorProto.UINT16,
        "u32": TensorProto.UINT32,
        "uint32_t": TensorProto.UINT32,
        "u64": TensorProto.UINT64,
        "uint64_t": TensorProto.UINT64,
        "bool": TensorProto.BOOL,
        "boolean": TensorProto.BOOL,
        # Add more if needed
    }

    # Add any required data members to the class
    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "target_device": PassConfigParam(
                type_=Device,
                default_value=accelerator_spec.accelerator_type.CPU,
                required=False,
                description=("Device the encapsulated model should run on. Available devices are cpu, gpu, npu."),
            ),
            "ov_version": PassConfigParam(
                type_=str,
                default_value=None,
                required=False,
                description=(
                    "Name of the OpenVINO version to override in model SDK version."
                    "Requires a minimum version of OpenVINO 2025.1"
                ),
            ),
            "opset_imports": PassConfigParam(
                type_=list,
                default_value=[
                    ["com.microsoft.nchwc", 1],
                    ["", 11],
                    ["ai.onnx.ml", 5],
                    ["com.ms.internal.nhwc", 11],
                    ["ai.onnx.training", 1],
                    ["ai.onnx.preview.training", 1],
                    ["com.microsoft.experimental", 1],
                    ["com.microsoft", 1],
                    ["org.pytorch.aten", 1],
                ],
                required=False,
                description="Opset name and version to be added in the generated context model",
            ),
            "keep_ov_dynamic_dims": PassConfigParam(
                type_=bool,
                default_value=False,
                required=False,
                description=(
                    "Keep OpenVINO dynamic dimensions in the generated ONNX model. "
                    "This is useful for models that require dynamic dimensions to be preserved."
                ),
            ),
            "reuse_cache": PassConfigParam(
                type_=bool,
                default_value=False,
                required=False,
                description=("Reuse cache of previous passes to reduce storage footprint."),
            ),
        }

    def _run_for_config(
        self,
        model: Union[OpenVINOModelHandler],
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> ONNXModelHandler:
        try:
            import openvino as ov
        except ImportError:
            raise ImportError("Please install olive-ai[openvino] to use OpenVINO model") from None

        model_name = model.model_config["model_name"]

        if config.reuse_cache:
            output_model_path = model.model_path

        if config.ov_version:
            ov_version = config.ov_version
        else:
            ov_version = ov.get_version()

        core = ov.Core()
        model_name_path = Path(model.model_path) / (f"{model_name}.xml")
        weight_name_path = Path(model.model_path) / (f"{model_name}.bin")

        loaded_model = core.read_model(model=model_name_path, weights=weight_name_path)

        context_name = f"{model_name}.xml"

        # Get/Fix input names & ov shapes.
        input_info = {}
        for i, inp in enumerate(loaded_model.inputs):
            name = "input_" + str(i)
            if inp:
                try:
                    name = inp.get_any_name()
                except Exception:
                    raise ValueError("Incorrect IO names, please use OpenVINO reshape pass before this pass") from None
            input_info[name] = (inp.get_partial_shape(), inp.get_element_type())

        # Get/Fix input names & ov shapes.
        output_info = {}
        for i, out in enumerate(loaded_model.outputs):
            name = "output_" + str(i)
            if out:
                try:
                    name = out.get_any_name()
                except Exception:
                    raise ValueError("Incorrect IO names, please use OpenVINO reshape pass before this pass") from None
            output_info[name] = (out.get_partial_shape(), out.get_element_type())

        # Transform to onnx input shapes
        inputs = []
        outputs = []
        for i, (name, (shape, datatype)) in enumerate(input_info.items()):
            shape_list = extract_shape_list(shape, config, prefix=f"input_{i}_")

            # Normalize the datatype string & map to ONNX data type
            normalized_dtype = str(datatype).split("'")[1]  # Extract 'int64_t' from "<Type: 'int64_t'>"
            onnx_dtype = self.openvino_to_onnx_dtype.get(normalized_dtype)

            inputs.append(helper.make_tensor_value_info(name, onnx_dtype, shape_list))

        # Transform to onnx output shapes
        for i, (name, (shape, datatype)) in enumerate(output_info.items()):
            shape_list = extract_shape_list(shape, config, prefix=f"output_{i}_")

            # Normalize the datatype string & map to ONNX data type and extract 'int64_t' from "<Type: 'int64_t'>"
            normalized_dtype = str(datatype).split("'")[1]
            onnx_dtype = self.openvino_to_onnx_dtype.get(normalized_dtype)

            outputs.append(helper.make_tensor_value_info(name, onnx_dtype, shape_list))

        # Create context node (simulates a custom EP context schema operation)
        context_node = helper.make_node(
            "EPContext",
            inputs=[name for name, _ in input_info.items()],
            outputs=[name for name, _ in output_info.items()],
            name="ContextNode",
            domain="com.microsoft",
        )

        # Properties of the context node, currently only support context node that points to the payload content
        context_node.attribute.extend([helper.make_attribute("embed_mode", 0)])
        context_node.attribute.extend([helper.make_attribute("ep_cache_context", context_name)])
        context_node.attribute.extend([helper.make_attribute("ep_sdk_version", ov_version)])
        context_node.attribute.extend([helper.make_attribute("main_context", 1)])
        context_node.attribute.extend([helper.make_attribute("max_size", 0)])
        context_node.attribute.extend([helper.make_attribute("source", "OpenVINOExecutionProvider")])
        context_node.attribute.extend([helper.make_attribute("DeviceClass", config.target_device.upper())])

        # Create the ONNX Graph
        graph_def = helper.make_graph(nodes=[context_node], name="EP_Context_Model", inputs=inputs, outputs=outputs)
        op_imports = [helper.make_opsetid(i[0], i[1]) for i in config.opset_imports]

        # Define the model with an Execution Provider (EP) Context
        model_def = helper.make_model(graph_def, opset_imports=op_imports)

        # Save the model
        context_model_output = f"{model_name}.onnx"
        context_model_output_dir = Path(output_model_path) / (context_model_output)

        if not os.path.exists(output_model_path):
            os.makedirs(output_model_path)

        save(model_def, context_model_output_dir)

        if not config.reuse_cache:
            model_name_path_dst = Path(output_model_path) / (f"{model_name}.xml")
            weight_name_path_dst = Path(output_model_path) / (f"{model_name}.bin")
            hardlink_copy_file(model_name_path, model_name_path_dst, follow_symlinks=True)
            hardlink_copy_file(weight_name_path, weight_name_path_dst, follow_symlinks=True)

            # copy JSON and text files for genai models
            all_genai_files = [name for name in Path(model.model_path).iterdir() if name.suffix in [".json", ".txt"]]
            for genai_file in all_genai_files:
                src_pth = Path(model.model_path) / genai_file
                dest_path = Path(output_model_path)
                hardlink_copy_file(src_pth, dest_path, follow_symlinks=True)

            # copy tokenizer folder if it exists
            src_tokenizer = Path(model.model_path) / "openvino_tokenizer"
            if src_tokenizer.exists() and src_tokenizer.is_dir():
                dest_tokenizer = Path(output_model_path) / "openvino_tokenizer"
                hardlink_copy_dir(src_tokenizer, dest_tokenizer, symlinks=True)

            # copy detokenizer folder if it exists
            src_detokenizer = Path(model.model_path) / "openvino_detokenizer"
            if src_detokenizer.exists() and src_detokenizer.is_dir():
                dest_detokenizer = Path(output_model_path) / "openvino_detokenizer"
                hardlink_copy_dir(src_detokenizer, dest_detokenizer, symlinks=True)

        # generate the genai_config.json file for GenAI models
        create_genai_config(context_model_output, output_model_path)

        return ONNXModelHandler(model_path=output_model_path)


def extract_shape_list(shape, config, prefix: str = "input_0_") -> list:
    """Extract the shape list from the OpenVINO model."""
    shape_list = []
    for j, dim in enumerate(shape):
        if not dim.is_dynamic:
            shape_list.append(dim.get_length())
        else:
            if not config.keep_ov_dynamic_dims:
                shape_list.append(prefix + f"{j}")
            else:
                shape_list.append(-1)
    return shape_list


def create_genai_config(model_name: str, output_path: str) -> None:
    """Generate the genai_config.json from the model config files.

    This is only for Generative AI models for which the config.json and generation_config.json files exist
    Arguments:
    @param model_name: name of model ONNX file that is generated
    @param output_path: path to the output directory where the genai_config.json file will be created
    @return: None
    """
    ip_conf_pth = Path(output_path) / "config.json"

    # do not create genai_config.json if config.json does not exist
    if not ip_conf_pth.exists():
        return

    ip_gen_pth = Path(output_path) / "generation_config.json"

    # do not create genai_config.json if generation_config.json does not exist
    if not ip_gen_pth.exists():
        return

    # Step 1: Create your data structure
    genai_config = {
        "model": {
            "bos_token_id": -1,
            "context_length": -1,
            "decoder": {
                "session_options": {
                    "log_id": "onnxruntime-genai",
                    "graph_optimization_level": "ORT_DISABLE_ALL",
                    "provider_options": [{"OpenVINO": {"device_type": "NPU", "enable_causallm": "True"}}],
                },
                "filename": "openvino_model.onnx",
                "head_size": -1,
                "hidden_size": -1,
                "inputs": {},
                "outputs": {},
                "num_attention_heads": -1,
                "num_hidden_layers": -1,
                "num_key_value_heads": -1,
            },
            "eos_token_id": -1,
            "type": "",
            "vocab_size": -1,
        },
        "search": {
            "diversity_penalty": 0.0,
            "do_sample": False,
            "early_stopping": True,
            "length_penalty": 1.0,
            "max_length": -1,
            "min_length": 0,
            "no_repeat_ngram_size": 0,
            "num_beams": 1,
            "num_return_sequences": 1,
            "past_present_share_buffer": False,
            "repetition_penalty": 1.0,
            "temperature": 1.0,
            "top_k": 1,
            "top_p": 1.0,
        },
    }

    import json

    with open(ip_conf_pth) as f:
        src_config = json.load(f)

    with open(ip_gen_pth) as f:
        src_gen_config = json.load(f)

    try:
        import onnx
    except ImportError:
        raise ImportError(
            "Please install onnx to create genai_config.json for ONNX OpenVINO IR Encapsulated model"
        ) from None

    model_path = Path(output_path) / model_name
    model = onnx.load(model_path)

    # Get input and output tensor names
    inputs = [inp.name for inp in model.graph.input]
    outputs = [out.name for out in model.graph.output]

    genai_config["model"]["bos_token_id"] = src_config.get("bos_token_id", -1)
    genai_config["model"]["context_length"] = src_config.get("max_position_embeddings", -1)
    genai_config["model"]["decoder"]["filename"] = model_name
    genai_config["model"]["decoder"]["head_size"] = src_config.get("hidden_size", -1) // src_config.get(
        "num_attention_heads", -1
    )
    genai_config["model"]["decoder"]["hidden_size"] = src_config.get("hidden_size", -1)

    for name in inputs:
        if name != "beam_idx":
            genai_config["model"]["decoder"]["inputs"].update({name: name})

    for name in outputs:
        genai_config["model"]["decoder"]["outputs"].update({name: name})

    genai_config["model"]["decoder"]["num_attention_heads"] = src_config.get("num_attention_heads", -1)
    genai_config["model"]["decoder"]["num_hidden_layers"] = src_config.get("num_hidden_layers", -1)
    genai_config["model"]["decoder"]["num_key_value_heads"] = src_config.get("num_key_value_heads", -1)

    genai_config["model"]["eos_token_id"] = src_gen_config.get("eos_token_id", -1)
    pad_token_id = src_gen_config.get("pad_token_id", -1)
    if pad_token_id != -1:
        genai_config["model"]["pad_token_id"] = pad_token_id
    else:
        genai_config["model"]["pad_token_id"] = genai_config["model"]["eos_token_id"]
    genai_config["model"]["type"] = src_config.get("model_type", "")
    genai_config["model"]["vocab_size"] = src_config.get("vocab_size", -1)

    genai_config["search"]["max_length"] = src_config.get("max_position_embeddings", -1)

    # Step 2: Write to JSON file
    output_genai_config = Path(output_path) / "genai_config.json"
    with open(output_genai_config, "w") as f:
        json.dump(genai_config, f, indent=4)
