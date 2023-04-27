# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import Any, Dict

from onnx import TensorProto, helper
from onnxruntime.transformers.convert_generation import get_shared_initializers

from olive.model import CompositeOnnxModel, OliveModel, ONNXModel
from olive.passes import Pass
from olive.passes.onnx.common import model_proto_to_olive_model
from olive.passes.pass_config import PassConfigParam


class InsertBeamSearchPass(Pass):
    """Insert Beam Search Op."""

    @staticmethod
    def _default_config() -> Dict[str, PassConfigParam]:
        config = {
            "no_repeat_ngram_size": PassConfigParam(
                type_=int,
                default_value=3,
                description=" If set to int > 0, all ngrams of that size can only occur once.",
            ),
        }
        return config

    def chain_model(self, model_A, model_A_name, model_B, model_B_name, model_config, options):
        # Chain two models (model_A and model_B) by inserting beam search op in between.
        model_A.graph.name = model_A_name + " subgraph"
        model_B.graph.name = model_B_name + " subgraph"

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
            "attention_mask",
        ]
        beam_outputs = ["sequences"]

        node = helper.make_node("BeamSearch", inputs=beam_inputs, outputs=beam_outputs, name="BeamSearch_node")
        node.domain = "com.microsoft"
        node.attribute.extend(
            [
                helper.make_attribute("eos_token_id", model_config.eos_token_id),
                helper.make_attribute("pad_token_id", model_config.pad_token_id),
                helper.make_attribute("decoder_start_token_id", model_config.decoder_start_token_id),
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
        attention_mask = helper.make_tensor_value_info(
            "attention_mask", TensorProto.INT32, ["batch_size", "feature_size", "sequence_length"]
        )

        graph_inputs = [
            input_features,
            max_length,
            min_length,
            num_beams,
            num_return_sequences,
            length_penalty,
            repetition_penalty,
            attention_mask,
        ]

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
                helper.make_attribute(model_B_name, model_B.graph),
                helper.make_attribute(model_A_name, model_A.graph),
            ]
        )
        opset_import = [
            helper.make_opsetid(domain="com.microsoft", version=1),
            helper.make_opsetid(domain="", version=17),
        ]

        beam_graph = helper.make_graph([node], "beam-search-test", graph_inputs, graph_outputs, initializers)
        beam_model = helper.make_model(beam_graph, producer_name="pytorch", opset_imports=opset_import)

        return beam_model

    def add_attention_mask(self, model, mask_name):
        mask = helper.make_tensor_value_info(mask_name, TensorProto.INT32, shape=["batch", "feature_size", "sequence"])
        model.graph.input.insert(1, mask)

    def _run_for_config(self, model: OliveModel, config: Dict[str, Any], output_model_path: str) -> ONNXModel:
        if isinstance(model, ONNXModel):
            return model

        if not isinstance(model, CompositeOnnxModel):
            raise ValueError

        # FIXME : Right now we are assuming that the composite model only has two components and beam search op
        # will be inserted in between them to chain the components. We should add a config option to identify
        # the two components to chain together when there are more than 2 components in the composite model.

        # Load encoder/decoder and insert necessary (but unused) graph inputs expected by BeamSearch op
        model_A = model.get_model_component(0)
        model_B = model.get_model_component(1)
        self.add_attention_mask(model_A, model_A.name + "_attention_mask")
        self.add_attention_mask(model_B, model_B.name + "_attention_mask")

        combined_model = self.chain_model(model_A, model_A.name, model_B, model_B.name, model.get_model_config(), config)

        # save the model to the output path and return the model
        output_model_path = ONNXModel.resolve_path(output_model_path)
        return model_proto_to_olive_model(combined_model.model, output_model_path, config, model.name)
