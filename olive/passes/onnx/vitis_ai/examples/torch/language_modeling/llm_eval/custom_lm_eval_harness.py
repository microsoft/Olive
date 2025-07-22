#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# ruff: noqa: T201
import json
from typing import Optional, Union

import numpy as np
import torch
import transformers
from lm_eval import evaluator
from lm_eval.api.model import LM
from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils import (
    get_dtype,
)
from onnxruntime import InferenceSession
from optimum.onnxruntime import ORTModelForCausalLM
from quark.torch import ModelImporter
from transformers import AutoConfig

eval_logger = evaluator.eval_logger


"""
    LMEVALModelWrapper is a custom class that modifies the functionality of HFLM in LM_Eval
    to additionally allow for evaluating .ONNX models.

"""


class LMEvalModelWrapper(HFLM):
    def __init__(
        self,
        pretrained: str,
        # The following args enable evaluating quark quantized models, onnx models, local pt models, and diff. data types
        import_file_format: Optional[str] = "hf_format",
        import_model_dir: Optional[str] = "",
        model_reload: Optional[bool] = False,
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        **kwargs,
    ) -> None:
        self.import_file_format = import_file_format
        self.import_model_dir = import_model_dir
        self.model_reload = model_reload
        self.pretrained = pretrained

        super().__init__(pretrained=pretrained, dtype=dtype, **kwargs)

    def _create_model(
        self,
        pretrained: str,
        revision: Optional[str] = "main",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        trust_remote_code: Optional[bool] = False,
        parallelize: Optional[bool] = False,
        gpus: Optional[int] = None,
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[str] = "./offload",
        # PEFT, delta weights and quantization options
        peft: Optional[str] = None,
        delta: Optional[str] = None,
        autogptq: Optional[Union[bool, str]] = False,
        gptqmodel: Optional[bool] = False,
        **kwargs,
    ) -> None:
        model_kwargs = kwargs if kwargs else {}
        model_kwargs.update(
            self._get_accelerate_args(
                parallelize=parallelize,
                device_map=kwargs.get("device_map"),
                max_memory_per_gpu=max_memory_per_gpu,
                max_cpu_memory=max_cpu_memory,
                offload_folder=offload_folder,
                gpus=gpus,
            )
        )

        self._model = self.AUTO_MODEL_CLASS.from_pretrained(
            self.pretrained, torch_dtype="auto", trust_remote_code=True, **model_kwargs
        )
        if dtype != "auto":
            self._model = self._model.to(get_dtype(dtype))

        if self.model_reload:
            importer = ModelImporter(model_info_dir=self.import_model_dir, saved_format=self.import_file_format)
            self._model = importer.import_model_info(self._model)

            if dtype != "auto":
                self._model = self._model.to(get_dtype(dtype))
            eval_logger.info(f"LOADING MODEL IN DTYPE:{self._model.dtype}")

        if self.import_file_format == "onnx_format":
            self.session = InferenceSession(self.import_model_dir + "/model.onnx", providers=["CPUExecutionProvider"])
            self.config_ = AutoConfig.from_pretrained(self.pretrained, trust_remote_code=True)
            # also parse the genai config file
            with open(self.import_model_dir + "/genai_config.json") as f:
                self.genai_config = json.load(f)
            # check to see if tokenizer contains pad token, if not set it
            if not hasattr(self.tokenizer, "pad_token_id"):
                self.tokenizer.pad_token_id = self.genai_config["model"]["pad_token_id"]
            # hotfix needed for CHATGLM specifically
            if self.pretrained == "THUDM/chatglm3-6b":
                self.config_.num_key_value_heads = self.genai_config["model"]["decoder"]["num_key_value_heads"]
            self._model = ORTModelForCausalLM(self.session, self.config, use_cache=True)
            eval_logger.info(f"LOADING ONNX EXPORTED MODEL FROM:{self.import_model_dir}")

        return

    def _model_call(self, inps):
        """:param inps: torch.Tensor
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)] or of shape
            [batch, sequence_ctx]. the size of sequence may vary from call to call
        :return
            A torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model's decoder
        """
        with torch.no_grad():
            assert transformers.AutoModelForCausalLM == self.AUTO_MODEL_CLASS

            if self.import_file_format == "onnx_format":
                attention_mask = torch.Tensor(np.where(inps != self.tokenizer.pad_token_id, 1, 0))
                self.model.use_io_binding = False

                if self.pretrained == "THUDM/chatglm3-6b":
                    past_key_values = [
                        (
                            torch.zeros(
                                (
                                    self.batch_size,
                                    self.config_.num_key_value_heads,
                                    inps.shape[1],
                                    self.config_.hidden_size // self.config_.num_attention_heads,
                                )
                            ),
                            torch.zeros(
                                (
                                    self.batch_size,
                                    self.config_.num_key_value_heads,
                                    inps.shape[1],
                                    self.config_.hidden_size // self.config_.num_attention_heads,
                                )
                            ),
                        )
                        for i in range(self.config_.num_layers)
                    ]
                    return self.model(inps, attention_mask=attention_mask, past_key_values=past_key_values).logits
                else:
                    return self.model(inps, attention_mask=attention_mask).logits
            else:
                return self.model(inps).logits


class LMEvalModelGenWrapper(LM):
    def __init__(
        self,
        outputs_path="",
        eor="<done>",
        limit=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.outputs_path = outputs_path
        self.limit = limit
        self.eor = eor

    def generate_until(self, requests, disable_tqdm: bool = False):
        resps = [" "] * len(requests)
        with open(self.outputs_path) as outputs_file:
            self.outputs = outputs_file.read().strip().rstrip(self.eor).split(self.eor)

        resps = self.outputs

        if len(self.outputs) != len(requests):
            raise ValueError(f"Outputs len - {len(self.outputs)}, Requests len - {len(requests)}. Did you set --limit?")

        if self.limit is not None:
            self.outputs = self.outputs[0 : self.limit]
            eval_logger.info(f"Sliced outputs to len {self.outputs}; limit set to {self.limit}")

        until_tokens = [request.args[1]["until"] for request in requests]

        res = []
        for resp, tokens in zip(resps, until_tokens):
            resp_candidates = []
            s = ""
            for until_token in tokens:
                resp_candidates.append(resp.split(until_token)[0])
            res.append(min(resp_candidates, key=len))
        return res

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        pass

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        pass
