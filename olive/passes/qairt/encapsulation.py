# -------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

import logging
import os
from pathlib import Path
from typing import Union

import onnx.helper as helper
from onnx import TensorProto, checker, save

from olive.common.utils import hardlink_copy_file
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler, QairtModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class QairtEncapsulation(Pass):
    """Encapsulates a QAIRT DLC model with an onnx protobuf."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "backend": PassConfigParam(
                type_=str,
                default_value="CPU",
                description="Target accelerator backend. Accepted values are 'CPU' and 'HTP'.",
            ),
            "log_level": PassConfigParam(
                type_=str,
                default_value=None,
                description="Log level to be used within underlying QAIRT components."
                "Valid values: DEBUG, INFO, WARN, ERROR.",
            ),
            "run_checker": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Runs the onnx checker on the model before it is encapsulated.",
            ),
            "opset_imports": PassConfigParam(
                type_=list,
                default_value=[
                    ["com.microsoft", 1],
                ],
                required=False,
                description="Opset name and version to be added in the generated context model",
            ),
        }

    def _run_for_config(
        self,
        model: Union[QairtModelHandler],
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> ONNXModelHandler:
        try:
            import qairt
            import qairt.gen_ai_api as qairt_genai
        except ImportError as exc:
            raise ImportError(
                "Failed to import QAIRT GenAIBuilder API - please install olive-ai[qairt] to use QAIRT passes."
                "If already installed, please run `qairt-vm -i` for help troubleshooting issues."
            ) from exc

        container: qairt_genai.LLMContainer = qairt_genai.LLMContainer.load(model.model_path)

        # Input/Ouptut metadata
        container.inputs = [("input_ids", TensorProto.INT32, ["batch_size", "sequence_length"])]
        container.outputs = [("logits", TensorProto.FLOAT, ["batch_size", 1, "vocab_size"])]

        input_info = {inp[0]: (inp[1], inp[2]) for inp in container.inputs}

        output_info = {out[0]: (out[1], out[2]) for out in container.outputs}

        # Input/Output tensor helpers
        inputs = []
        for name, datatype, shape in container.inputs:
            inputs.append(helper.make_tensor_value_info(name, datatype, shape))

        outputs = []
        for name, datatype, shape in container.outputs:
            outputs.append(helper.make_tensor_value_info(name, datatype, shape))

        container.export(output_model_path, export_format=qairt.ExportFormat.LM_EXECUTOR)

        # Find the .dlc file in the output directory
        output_path_obj = Path(output_model_path)
        dlc_files = list(output_path_obj.glob("*.dlc"))

        if not dlc_files:
            raise FileNotFoundError(
                f"No .dlc file found in {output_model_path} after export. "
                "Expected at least one .dlc file to be generated."
            )

        if len(dlc_files) > 1:
            logger.warning(
                "Multiple .dlc files found in %s: %s. Using the first one: %s",
                output_model_path,
                [f.name for f in dlc_files],
                dlc_files[0].name,
            )

        dlc_filename = dlc_files[0].name
        logger.info("Found DLC file: %s", dlc_filename)

        context_node = helper.make_node(
            "EPContext",
            inputs=[name for name, _ in input_info.items()],
            outputs=[name for name, _ in output_info.items()],
            name="ContextNode",
            domain="com.microsoft",
        )

        context_node.attribute.extend([helper.make_attribute("ep_context_type", "dlc")])
        context_node.attribute.extend([helper.make_attribute("ep_dlc_context", dlc_filename)])
        context_node.attribute.extend([helper.make_attribute("source", "QAIRTExport")])

        # Create the ONNX Graph
        graph_def = helper.make_graph(nodes=[context_node], name="EP_Context_Model", inputs=inputs, outputs=outputs)
        op_imports = [helper.make_opsetid(i[0], i[1]) for i in config.opset_imports]

        # Define the model with an Execution Provider (EP) Context
        model_def = helper.make_model(graph_def, opset_imports=op_imports)
        model_def.ir_version = 10

        if config.run_checker:
            checker.check_model(model_def)

        # Save the model
        model_name = "model"
        context_model_output = f"{model_name}.onnx"
        context_model_output_dir = Path(output_model_path) / (context_model_output)

        if not os.path.exists(output_model_path):
            os.makedirs(output_model_path)

        save(model_def, context_model_output_dir)

        # onnxruntime-genai requires certain source model files to be passed through
        passthrough_files = [
            "chat_template.jinja",
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ]
        for file in passthrough_files:
            config_path = Path(model.model_path) / file
            dest_path = Path(output_model_path)
            try:
                hardlink_copy_file(config_path, dest_path, follow_symlinks=True)
            except (OSError, FileNotFoundError):
                # Not every model has all the files listed above
                pass

        create_genai_config(context_model_output, output_model_path, config)

        return ONNXModelHandler(model_path=output_model_path)


def create_genai_config(model_name: str, output_path: str, config: type[BasePassConfig]) -> None:
    """Generate the genai_config.json from the model config files.

    This is only for Generative AI models for which the config.json and generation_config.json files exist
    Args:
        model_name: name of model ONNX file that is generated
        output_path: path to the output directory where the genai_config.json file will be created
        config: pass configuration containing backend and other settings
        return: None
    """
    source_config_path = Path(output_path) / "config.json"

    if not source_config_path.exists():
        raise ValueError("Cannot create gen_ai_config.json if source model config doesn't exist.")

    generation_config_path = Path(output_path) / "generation_config.json"

    if not generation_config_path.exists():
        raise ValueError("Cannot create gen_ai_config.json if generation config doesn't exist")

    genai_config = {
        "model": {
            "bos_token_id": -1,
            "context_length": -1,
            "decoder": {
                "session_options": {
                    "log_id": "onnxruntime-genai",
                    "graph_optimization_level": "ORT_DISABLE_ALL",
                    "provider_options": [{"QNN": {"backend_type": config.backend}, "genai_model": "True"}],
                },
                "filename": "qairt_model.onnx",
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
            "past_present_share_buffer": True,
            "repetition_penalty": 1.0,
            "temperature": 1.0,
            "top_k": 1,
            "top_p": 1.0,
        },
    }

    import json

    with open(source_config_path) as f:
        src_config = json.load(f)

    with open(generation_config_path) as f:
        gen_config = json.load(f)

    try:
        import onnx
    except ImportError:
        raise ImportError("Please install onnx to create genai_config.json for ONNX QAIRT Encapsulated model") from None

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
        genai_config["model"]["decoder"]["inputs"].update({name: name})

    for name in outputs:
        genai_config["model"]["decoder"]["outputs"].update({name: name})

    genai_config["model"]["decoder"]["num_attention_heads"] = src_config.get("num_attention_heads", -1)
    genai_config["model"]["decoder"]["num_hidden_layers"] = src_config.get("num_hidden_layers", -1)
    genai_config["model"]["decoder"]["num_key_value_heads"] = src_config.get("num_key_value_heads", -1)

    genai_config["model"]["eos_token_id"] = gen_config.get("eos_token_id", -1)
    genai_config["model"]["pad_token_id"] = (
        gen_config["pad_token_id"]
        if hasattr(gen_config, "pad_token_id") and gen_config["pad_token_id"] is not None
        else gen_config["eos_token_id"][0]
        if isinstance(gen_config["eos_token_id"], list)
        else gen_config["eos_token_id"]
    )
    genai_config["model"]["type"] = src_config.get("model_type", "")
    genai_config["model"]["vocab_size"] = src_config.get("vocab_size", -1)

    genai_config["search"]["max_length"] = src_config.get("max_position_embeddings", -1)

    output_genai_config = Path(output_path) / "genai_config.json"
    with open(output_genai_config, "w") as f:
        json.dump(genai_config, f, indent=4)
