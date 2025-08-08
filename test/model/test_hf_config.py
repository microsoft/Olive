# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import pytest
import torch
import transformers
from packaging import version

from olive.common.pydantic_v1 import ValidationError
from olive.model.config.hf_config import HfLoadKwargs


class TestHfLoadKwargs:
    @pytest.mark.parametrize(
        ("inputs", "inner", "output"),
        [
            ("auto", "auto", "auto"),
            (torch.float32, "float32", torch.float32),
            ("float32", "float32", torch.float32),
            ("torch.float32", "torch.float32", torch.float32),
        ],
    )
    def test_torch_dtype(self, inputs, inner, output):
        args = HfLoadKwargs(torch_dtype=inputs)
        assert args.torch_dtype == inner
        assert args.get_torch_dtype() == output

    @pytest.mark.parametrize(
        ("inputs", "inner"),
        [
            ("auto", "auto"),
            (1, 1),
        ],
    )
    def test_device_map(self, inputs, inner):
        args = HfLoadKwargs(device_map=inputs)
        assert args.device_map == inner

        args = HfLoadKwargs(device_map={"": inputs})
        assert args.device_map == {"": inner}

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
    @pytest.mark.parametrize(
        ("inputs", "inner"),
        [
            ("cuda:0", "cuda:0"),
            ("0", "cuda:0"),
        ],
    )
    def test_device_map_cpu(self, inputs, inner):
        if inputs == "0":
            inputs = torch.device(0)

        args = HfLoadKwargs(device_map=inputs)
        assert args.device_map == inner

        args = HfLoadKwargs(device_map={"": inputs})
        assert args.device_map == {"": inner}

    @pytest.mark.parametrize(
        ("quantization_method", "quantization_config", "valid"),
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
                _ = HfLoadKwargs(quantization_method=quantization_method, quantization_config=quantization_config)

        else:
            args = HfLoadKwargs(quantization_method=quantization_method, quantization_config=quantization_config)
            if quantization_method is None:
                return

            # check quantization method and config
            assert args.quantization_method == quantization_method
            assert isinstance(args.quantization_config, dict)
            for k, v in quantization_config.items():
                # assumes there are no unused keys in quantization_config
                assert args.quantization_config[k] == v

    # There is dependency conflict between transformers>=4.27.0 and azureml-evaluate-mlflow
    # TODO(jambayk): remove this skip when the dependency conflict is resolved
    @pytest.mark.skipif(
        version.parse(transformers.__version__) < version.parse("4.27.0"), reason="requires transformers>=4.27.0"
    )
    def test_get_quantization_config(self):
        from transformers import BitsAndBytesConfig

        quanntization_method = "bitsandbytes"
        quantization_config = {"load_in_8bit": True}
        args = HfLoadKwargs(quantization_method=quanntization_method, quantization_config=quantization_config)
        config = args.get_quantization_config()

        assert isinstance(config, BitsAndBytesConfig)
        for k, v in quantization_config.items():
            assert getattr(config, k) == v
        for k in args.quantization_config:
            assert hasattr(config, k)
