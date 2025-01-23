# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from typing import Dict, Type

from onnx import ModelProto, TensorProto, helper
from packaging import version

from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.model import CompositeModelHandler, OliveModelHandler, ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)

# ruff: noqa: N806


class InsertBeamSearch(Pass):
    """Insert Beam Search Op. Only used for whisper models.

    Uses WhisperBeamSearch contrib op if ORT version >= 1.17.1, else uses BeamSearch contrib op.
    """

    _accepts_composite_model = True

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        use_gpu = accelerator_spec.accelerator_type == Device.GPU
        config = {
            "no_repeat_ngram_size": PassConfigParam(
                type_=int,
                default_value=0,
                description=" If set to int > 0, all ngrams of that size can only occur once.",
            ),
            "use_vocab_mask": PassConfigParam(
                type_=bool,
                default_value=False,
                description=(
                    "Use vocab_mask as an extra graph input to the beam search op. Only supported in ORT >= 1.16.0"
                ),
            ),
            "use_prefix_vocab_mask": PassConfigParam(
                type_=bool,
                default_value=False,
                description=(
                    "Use prefix_vocab_mask as an extra graph input to the beam search op. Only supported in ORT >="
                    " 1.16.0"
                ),
            ),
            "use_forced_decoder_ids": PassConfigParam(
                type_=bool,
                default_value=False,
                description=(
                    "Use decoder_input_ids as an extra graph input to the beam search op. Only supported in ORT >="
                    " 1.16.0"
                ),
            ),
            "use_logits_processor": PassConfigParam(
                type_=bool,
                default_value=False,
                description=(
                    "Use logits_processor as an extra graph input to the beam search op. Only supported in ORT >="
                    " 1.16.0"
                ),
            ),
            "use_temperature": PassConfigParam(
                type_=bool,
                default_value=False,
                description=(
                    "Use temperature as an extra graph input to the beam search op. Only supported in ORT >= 1.17.1"
                ),
            ),
            "fp16": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Is the model in fp16 precision.",
            ),
            "use_gpu": PassConfigParam(
                type_=bool,
                default_value=use_gpu,
                description="Use GPU for beam search op.",
            ),
        }
        config.update(get_external_data_config())
        return config

    def chain_model(
        self,
        model_A: ModelProto,
        model_A_name: str,
        model_B: ModelProto,
        model_B_name: str,
        model_config,
        options: Type[BasePassConfig],
    ):
        from onnxruntime import __version__ as OrtVersion
        from onnxruntime.transformers.convert_generation import get_shared_initializers

        # version check
        version_1_16 = version.parse(OrtVersion) >= version.parse("1.16.0")
        version_1_17_1 = version.parse(OrtVersion) >= version.parse("1.17.1")
        # NOTE: will ignore cross qk related options for now

        # Chain two models (model_A and model_B) by inserting beam search op in between.
        model_A.graph.name = f"{model_A_name} subgraph"
        model_B.graph.name = f"{model_B_name} subgraph"

        beam_inputs = [
            "input_features_fp16" if options.fp16 else "input_features",
            "max_length",
            "min_length",
            "num_beams",
            "num_return_sequences",
            "length_penalty_fp16" if options.fp16 else "length_penalty",
            "repetition_penalty_fp16" if options.fp16 else "repetition_penalty",
            "vocab_mask" if (version_1_16 and options.use_vocab_mask) else "",
            "prefix_vocab_mask" if (version_1_16 and options.use_prefix_vocab_mask) else "",
            "" if version_1_16 else "attention_mask",
        ]
        if version_1_16:
            beam_inputs.extend(["decoder_input_ids" if options.use_forced_decoder_ids else ""])
            beam_inputs.extend(["logits_processor" if options.use_logits_processor else ""])
        if version_1_17_1:
            beam_inputs.extend(["", ""])
            beam_inputs.extend(
                [("temperature_fp16" if options.fp16 else "temperature") if options.use_temperature else ""]
            )
        # remove empty string from the end of beam_inputs
        # otherwise, the model gives error when the last input is empty
        while beam_inputs[-1] == "":
            beam_inputs.pop()

        # Cast input features to fp16 if required
        graph_nodes = []
        if options.fp16:
            input_features_cast_node = helper.make_node(
                "Cast",
                inputs=["input_features"],
                outputs=["input_features_fp16"],
                name="CastInputFeaturesToFp16",
                to=TensorProto.FLOAT16,
            )
            len_pen_cast_node = helper.make_node(
                "Cast",
                inputs=["length_penalty"],
                outputs=["length_penalty_fp16"],
                name="CastLengthPenaltyToFp16",
                to=TensorProto.FLOAT16,
            )
            rep_pen_cast_node = helper.make_node(
                "Cast",
                inputs=["repetition_penalty"],
                outputs=["repetition_penalty_fp16"],
                name="CastRepetitionPenaltyToFp16",
                to=TensorProto.FLOAT16,
            )
            graph_nodes.extend([input_features_cast_node, len_pen_cast_node, rep_pen_cast_node])

            if version_1_17_1 and options.use_temperature:
                temperature_cast_node = helper.make_node(
                    "Cast",
                    inputs=["temperature"],
                    outputs=["temperature_fp16"],
                    name="CastTemperatureToFp16",
                    to=TensorProto.FLOAT16,
                )
                graph_nodes.append(temperature_cast_node)

        beam_outputs = ["sequences"]

        # beam search op attributes
        beam_search_attrs = [
            helper.make_attribute("eos_token_id", model_config["eos_token_id"]),
            helper.make_attribute("pad_token_id", model_config["pad_token_id"]),
            helper.make_attribute("decoder_start_token_id", model_config["decoder_start_token_id"]),
            helper.make_attribute("no_repeat_ngram_size", options.no_repeat_ngram_size),
            helper.make_attribute("early_stopping", True),
            helper.make_attribute("model_type", 2),
        ]
        if version_1_17_1:
            from transformers import AutoTokenizer

            # get tokenizer
            # can get the base name of the model from the config
            tokenizer = AutoTokenizer.from_pretrained(model_config["_name_or_path"])

            beam_search_attrs.extend(
                [
                    helper.make_attribute("translate_token_id", tokenizer.convert_tokens_to_ids(["<|translate|>"])[0]),
                    helper.make_attribute(
                        "transcribe_token_id", tokenizer.convert_tokens_to_ids(["<|transcribe|>"])[0]
                    ),
                    helper.make_attribute(
                        "start_of_lm_token_id", tokenizer.convert_tokens_to_ids(["<|startoflm|>"])[0]
                    ),
                    helper.make_attribute(
                        "no_timestamps_token_id", tokenizer.convert_tokens_to_ids(["<|notimestamps|>"])[0]
                    ),
                    helper.make_attribute(
                        "beginning_timestamp_token_id", tokenizer.convert_tokens_to_ids(["<|0.00|>"])[0]
                    ),
                ]
            )

        node = helper.make_node(
            "WhisperBeamSearch" if version_1_17_1 else "BeamSearch",
            inputs=beam_inputs,
            outputs=beam_outputs,
            name="BeamSearch_node",
            domain="com.microsoft",
        )
        node.attribute.extend(beam_search_attrs)

        # Graph inputs
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
        else:
            if options.use_vocab_mask:
                vocab_mask = helper.make_tensor_value_info(
                    "vocab_mask", TensorProto.INT32, [model_config["vocab_size"]]
                )
                graph_inputs.append(vocab_mask)

            if options.use_prefix_vocab_mask:
                prefix_vocab_mask = helper.make_tensor_value_info(
                    "prefix_vocab_mask", TensorProto.INT32, ["batch_size", model_config["vocab_size"]]
                )
                graph_inputs.append(prefix_vocab_mask)

            if options.use_forced_decoder_ids:
                decoder_input_ids = helper.make_tensor_value_info(
                    "decoder_input_ids", TensorProto.INT32, ["batch_size", "initial_sequence_length"]
                )
                graph_inputs.append(decoder_input_ids)

            if options.use_logits_processor:
                logits_processor = helper.make_tensor_value_info("logits_processor", TensorProto.INT32, [1])
                graph_inputs.append(logits_processor)

            if version_1_17_1 and options.use_temperature:
                temperature = helper.make_tensor_value_info("temperature", TensorProto.FLOAT, [1])
                graph_inputs.append(temperature)

        # Graph outputs
        sequences = helper.make_tensor_value_info(
            "sequences", TensorProto.INT32, ["batch_size", "num_return_sequences", "max_length"]
        )
        graph_outputs = [sequences]

        # Replace MultiHeadAttention with DecoderMaskedMultiHeadAttention for CUDA EP inference
        if options.use_gpu and version_1_16:
            from onnxruntime.transformers.convert_generation import (
                update_decoder_subgraph_share_buffer_and_use_decoder_masked_mha as update_decoder_with_ort,
            )

            if update_decoder_with_ort(model_B.graph):
                logger.info("Updated whisper decoder subgraph to use DecoderMaskedMultiHeadAttention successfully!")
            else:
                logger.warning("DecoderMaskedMultiHeadAttention could not be applied to whisper decoder subgraph")

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

        graph_nodes.append(node)

        # Make graph with BeamSearch/WhisperBeamSearch op
        beam_graph = helper.make_graph(graph_nodes, "beam-search-test", graph_inputs, graph_outputs, initializers)
        assert model_A.ir_version == model_B.ir_version
        logger.debug("Using IR version %s for chained model", model_A.ir_version)

        # Set IR version of chained model to IR version of subgraphs in order to generate a working E2E model
        return helper.make_model_gen_version(
            beam_graph,
            producer_name="Olive",
            opset_imports=opset_import,
            ir_version=model_A.ir_version,
        )

    def add_attention_mask(self, model: ModelProto):
        mask = helper.make_tensor_value_info(
            "encoder_attention_mask", TensorProto.INT32, shape=["batch", "feature_size", "sequence"]
        )
        model.graph.input.insert(1, mask)

    def _run_for_config(
        self, model: OliveModelHandler, config: Type[BasePassConfig], output_model_path: str
    ) -> ONNXModelHandler:
        from onnxruntime import __version__ as OrtVersion
        from onnxruntime.transformers import onnx_model as ort_onnx_model

        if isinstance(model, ONNXModelHandler):
            return model

        if not isinstance(model, CompositeModelHandler):
            raise ValueError

        # TODO(jambayk): Right now we are assuming that the composite model only has two components and beam search op
        # will be inserted in between them to chain the components. We should add a config option to identify
        # the two components to chain together when there are more than 2 components in the composite model.

        # version check
        version_1_16 = version.parse(OrtVersion) >= version.parse("1.16.0")

        if not version_1_16 and config.use_forced_decoder_ids:
            logger.warning(
                "use_forced_decoder_ids is not supported in ONNX Runtime versions < 1.16.0. Will be ignored."
            )

        # Load encoder/decoder and insert necessary (but unused) graph inputs expected by BeamSearch op
        components = model.get_model_components()
        names, models = zip(*components)

        model_A = models[0]
        model_A_name = names[0]
        model_B = models[1]
        model_B_name = names[1]
        model_proto_A = model_A.load_model()
        model_proto_B = model_B.load_model()
        if not version_1_16:
            self.add_attention_mask(model_proto_A)
            self.add_attention_mask(model_proto_B)

        ort_onnx_model.OnnxModel.graph_topological_sort(model_proto_A.graph)
        ort_onnx_model.OnnxModel.graph_topological_sort(model_proto_B.graph)
        combined_model = self.chain_model(
            model_proto_A, model_A_name, model_proto_B, model_B_name, model.model_attributes, config
        )
        # save the model to the output path and return the model
        output_model_path = resolve_onnx_path(output_model_path, "model_with_beam_search.onnx")
        return model_proto_to_olive_model(combined_model, output_model_path, config, True)
