# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from typing import Any, Dict

from onnx import ModelProto, TensorProto, helper
from onnxruntime import __version__ as OrtVersion
from onnxruntime.transformers.convert_generation import get_shared_initializers
from packaging import version

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import CompositeOnnxModel, OliveModel, ONNXModel
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.pass_config import PassConfigParam

logger = logging.getLogger(__name__)


class InsertBeamSearch(Pass):
    """Insert Beam Search Op."""

    _accepts_composite_model = True

    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {
            "no_repeat_ngram_size": PassConfigParam(
                type_=int,
                default_value=0,
                description=" If set to int > 0, all ngrams of that size can only occur once.",
            ),
            "use_forced_decoder_ids": PassConfigParam(
                type_=bool,
                default_value=False,
                description=(
                    "Use decoder_input_ids as an extra graph input to the beam search op. Only supported in ORT >="
                    " 1.16.0"
                ),
            ),
        }
        config.update(get_external_data_config())
        return config

    def chain_model(
        self, model_A: ModelProto, model_A_name: str, model_B: ModelProto, model_B_name: str, model_config, options
    ):
        # version check
        version_1_16 = version.parse(OrtVersion) >= version.parse("1.16.0")

        # Chain two models (model_A and model_B) by inserting beam search op in between.
        model_A.graph.name = f"{model_A_name} subgraph"
        model_B.graph.name = f"{model_B_name} subgraph"

        beam_inputs = [
            "input_features",
            "max_length",
            "min_length",
            "num_beams",
            "num_return_sequences",
            "length_penalty",
            "repetition_penalty",
            "",
            "",
            "" if version_1_16 else "attention_mask",
        ]
        if version_1_16:
            beam_inputs.extend(["decoder_input_ids" if options["use_forced_decoder_ids"] else "", ""])

        beam_outputs = ["sequences"]

        node = helper.make_node("BeamSearch", inputs=beam_inputs, outputs=beam_outputs, name="BeamSearch_node")
        node.domain = "com.microsoft"
        node.attribute.extend(
            [
                helper.make_attribute("eos_token_id", model_config["eos_token_id"]),
                helper.make_attribute("pad_token_id", model_config["pad_token_id"]),
                helper.make_attribute("decoder_start_token_id", model_config["decoder_start_token_id"]),
                helper.make_attribute("no_repeat_ngram_size", options["no_repeat_ngram_size"]),
                helper.make_attribute("early_stopping", True),
                helper.make_attribute("model_type", 2),
            ]
        )

        # beam graph inputs
        input_features = helper.make_tensor_value_info(
            "input_features", TensorProto.FLOAT, ["batch_size", "feature_size", "sequence_length"]
        )
        max_length = helper.make_tensor_value_info("max_length", TensorProto.INT32, [1])
        min_length = helper.make_tensor_value_info("min_length", TensorProto.INT32, [1])
        num_beams = helper.make_tensor_value_info("num_beams", TensorProto.INT32, [1])
        num_return_sequences = helper.make_tensor_value_info("num_return_sequences", TensorProto.INT32, [1])
        length_penalty = helper.make_tensor_value_info("length_penalty", TensorProto.FLOAT, [1])
        repetition_penalty = helper.make_tensor_value_info("repetition_penalty", TensorProto.FLOAT, [1])

        graph_inputs = [
            input_features,
            max_length,
            min_length,
            num_beams,
            num_return_sequences,
            length_penalty,
            repetition_penalty,
        ]
        if not version_1_16:
            attention_mask = helper.make_tensor_value_info(
                "attention_mask", TensorProto.INT32, ["batch_size", "feature_size", "sequence_length"]
            )
            graph_inputs.append(attention_mask)
        if version_1_16 and options["use_forced_decoder_ids"]:
            decoder_input_ids = helper.make_tensor_value_info(
                "decoder_input_ids", TensorProto.INT32, ["batch_size", "initial_sequence_length"]
            )
            graph_inputs.append(decoder_input_ids)

        # graph outputs
        sequences = helper.make_tensor_value_info(
            "sequences", TensorProto.INT32, ["batch_size", "num_return_sequences", "max_length"]
        )
        graph_outputs = [sequences]

        # Initializers/opsets
        # Delete shared data between decoder/encoder and move to larger graph initializers
        initializers = get_shared_initializers(model_A, model_B)
        node.attribute.extend(
            [
                helper.make_attribute("decoder", model_B.graph),
                helper.make_attribute("encoder", model_A.graph),
            ]
        )
        opset_import = [
            helper.make_opsetid(domain="com.microsoft", version=1),
            helper.make_opsetid(domain="", version=17),
        ]

        beam_graph = helper.make_graph([node], "beam-search-test", graph_inputs, graph_outputs, initializers)
        assert model_A.ir_version == model_B.ir_version
        logger.debug(f"Using IR version {model_A.ir_version} for chained model")

        # Set IR version of chained model to IR version of subgraphs in order to generate a working E2E model
        beam_model = helper.make_model_gen_version(
            beam_graph,
            producer_name="Olive",
            opset_imports=opset_import,
            ir_version=model_A.ir_version,
        )

        return beam_model

    def add_attention_mask(self, model: ModelProto):
        mask = helper.make_tensor_value_info(
            "encoder_attention_mask", TensorProto.INT32, shape=["batch", "feature_size", "sequence"]
        )
        model.graph.input.insert(1, mask)

    def _run_for_config(
        self, model: OliveModel, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> ONNXModel:
        if isinstance(model, ONNXModel):
            return model

        if not isinstance(model, CompositeOnnxModel):
            raise ValueError

        # FIXME : Right now we are assuming that the composite model only has two components and beam search op
        # will be inserted in between them to chain the components. We should add a config option to identify
        # the two components to chain together when there are more than 2 components in the composite model.

        # version check
        version_1_16 = version.parse(OrtVersion) >= version.parse("1.16.0")

        if not version_1_16 and config["use_forced_decoder_ids"]:
            logger.warning(
                "use_forced_decoder_ids is not supported in ONNX Runtime versions < 1.16.0. Will be ignored."
            )

        # Load encoder/decoder and insert necessary (but unused) graph inputs expected by BeamSearch op
        model_A = model.get_model_component(0)
        model_A_name = model.get_model_component_name(0)
        model_B = model.get_model_component(1)
        model_B_name = model.get_model_component_name(1)
        model_proto_A = model_A.load_model()
        model_proto_B = model_B.load_model()
        if not version_1_16:
            self.add_attention_mask(model_proto_A)
            self.add_attention_mask(model_proto_B)

        combined_model = self.chain_model(
            model_proto_A, model_A_name, model_proto_B, model_B_name, model.model_attributes, config
        )

        # save the model to the output path and return the model
        output_model_path = ONNXModel.resolve_path(output_model_path)
        return model_proto_to_olive_model(combined_model, output_model_path, config, True)
