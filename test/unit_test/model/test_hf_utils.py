# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import pytest
import torch
import transformers
from packaging import version
from pydantic import ValidationError
from transformers.onnx import OnnxConfig

from olive.model.hf_utils import (
    HFModelLoadingArgs,
    get_onnx_config,
    load_huggingface_model_from_model_class,
    load_huggingface_model_from_task,
)


def test_load_huggingface_model_from_task():
    # The model name and task type is gotten from
    # https://huggingface.co/docs/transformers/v4.28.1/en/main_classes/pipelines#transformers.pipeline
    task = "text-classification"
    model_name = "Intel/bert-base-uncased-mrpc"

    model = load_huggingface_model_from_task(task, model_name)
    assert isinstance(model, torch.nn.Module)


def test_load_huggingface_model_from_model_class():
    model_class = "AutoModelForSequenceClassification"
    model_name = "Intel/bert-base-uncased-mrpc"
    model = load_huggingface_model_from_model_class(model_class, model_name)
    assert isinstance(model, torch.nn.Module)


@pytest.mark.parametrize(
    "model_name,task,feature",
    [
        ("Intel/bert-base-uncased-mrpc", "text-classification", "default"),
        ("facebook/opt-125m", "text-generation", "default"),
    ],
)
def test_get_onnx_config(model_name, task, feature):
    onnx_config = get_onnx_config(model_name, task, feature)
    assert isinstance(onnx_config, OnnxConfig)


class TestHFModelLoadingArgs:
    @pytest.mark.parametrize(
        "input,inner,output",
        [
            ("auto", "auto", "auto"),
            (torch.float32, "float32", torch.float32),
            ("float32", "float32", torch.float32),
            ("torch.float32", "torch.float32", torch.float32),
        ],
    )
    def test_torch_dtype(self, input, inner, output):
        args = HFModelLoadingArgs(torch_dtype=input)
        assert args.torch_dtype == inner
        assert args.get_torch_dtype() == output

    @pytest.mark.parametrize(
        "input,inner",
        [
            ("auto", "auto"),
            (1, 1),
            ("cuda:0", "cuda:0"),
            (torch.device(0), "cuda:0"),
            (torch.device("cuda:0"), "cuda:0"),
        ],
    )
    def test_device_map(self, input, inner):
        args = HFModelLoadingArgs(device_map=input)
        assert args.device_map == inner

        args = HFModelLoadingArgs(device_map={"": input})
        assert args.device_map == {"": inner}

    @pytest.mark.parametrize(
        "quantization_method,quantization_config,valid",
        [
            ("bitsandbyte", None, False),
            (None, None, True),
            (None, {"load_in_8bit": True}, False),
            ("dummy", {"load_in_8bit": True}, False),
            ("bitsandbytes", {"load_in_8bit": True}, True),
        ],
    )
    def test_quant(self, quantization_method, quantization_config, valid):
        if not valid:
            with pytest.raises(ValidationError):
                args = HFModelLoadingArgs(
                    quantization_method=quantization_method, quantization_config=quantization_config
                )

        else:
            args = HFModelLoadingArgs(quantization_method=quantization_method, quantization_config=quantization_config)
            if quantization_method is None:
                return

            # check quantization method and config
            assert args.quantization_method == quantization_method
            assert isinstance(args.quantization_config, dict)
            for k, v in quantization_config.items():
                # assumes there are no unused keys in quantization_config
                assert args.quantization_config[k] == v

    # There is dependency conflict between transformers>=4.27.0 and azureml-evaluate-mlflow
    # TODO: remove this skip when the dependency conflict is resolved
    @pytest.mark.skipif(
        version.parse(transformers.__version__) < version.parse("4.27.0"), reason="requires transformers>=4.27.0"
    )
    def test_get_quantization_config(self):
        from transformers import BitsAndBytesConfig

        quanntization_method = "bitsandbytes"
        quantization_config = {"load_in_8bit": True}
        args = HFModelLoadingArgs(quantization_method=quanntization_method, quantization_config=quantization_config)
        config = args.get_quantization_config()

        assert isinstance(config, BitsAndBytesConfig)
        for k, v in quantization_config.items():
            assert getattr(config, k) == v
        for k in args.quantization_config:
            assert hasattr(config, k)
