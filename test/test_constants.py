# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import pytest

from olive.constants import (
    MSFT_DOMAIN,
    AccuracyLevel,
    DatasetRequirement,
    DiffusersComponent,
    DiffusersModelVariant,
    Framework,
    ModelFileFormat,
    OpType,
    Precision,
    PrecisionBits,
    QuantAlgorithm,
    QuantEncoding,
    precision_bits_from_precision,
)


class TestFramework:
    def test_framework_values(self):
        assert Framework.ONNX == "ONNX"
        assert Framework.PYTORCH == "PyTorch"
        assert Framework.OPENVINO == "OpenVINO"

    def test_framework_str(self):
        assert str(Framework.ONNX) == "ONNX"

    def test_framework_all_members(self):
        expected = {"ONNX", "PYTORCH", "QAIRT", "QNN", "TENSORFLOW", "OPENVINO"}
        assert set(Framework.__members__.keys()) == expected


class TestModelFileFormat:
    def test_model_file_format_values(self):
        assert ModelFileFormat.ONNX == "ONNX"
        assert ModelFileFormat.PYTORCH_STATE_DICT == "PyTorch.StateDict"
        assert ModelFileFormat.COMPOSITE_MODEL == "Composite"

    def test_model_file_format_str(self):
        assert str(ModelFileFormat.OPENVINO_IR) == "OpenVINO.IR"


class TestPrecision:
    def test_precision_values(self):
        assert Precision.INT4 == "int4"
        assert Precision.FP16 == "fp16"
        assert Precision.BF16 == "bf16"

    def test_precision_all_members(self):
        expected_count = 14
        assert len(Precision) == expected_count


class TestPrecisionBits:
    def test_precision_bits_values(self):
        assert PrecisionBits.BITS2 == 2
        assert PrecisionBits.BITS4 == 4
        assert PrecisionBits.BITS8 == 8
        assert PrecisionBits.BITS16 == 16
        assert PrecisionBits.BITS32 == 32

    def test_precision_bits_is_int(self):
        assert isinstance(PrecisionBits.BITS4.value, int)


class TestQuantAlgorithm:
    def test_quant_algorithm_case_insensitive(self):
        assert QuantAlgorithm("awq") == QuantAlgorithm.AWQ
        assert QuantAlgorithm("AWQ") == QuantAlgorithm.AWQ
        assert QuantAlgorithm("Awq") == QuantAlgorithm.AWQ

    def test_quant_algorithm_values(self):
        assert QuantAlgorithm.GPTQ == "gptq"
        assert QuantAlgorithm.RTN == "rtn"

    def test_quant_algorithm_all_members(self):
        expected = {"AWQ", "GPTQ", "HQQ", "RTN", "SPINQUANT", "QUAROT", "LPBQ", "SEQMSE", "ADAROUND"}
        assert set(QuantAlgorithm.__members__.keys()) == expected


class TestQuantEncoding:
    def test_quant_encoding_values(self):
        assert QuantEncoding.QDQ == "qdq"
        assert QuantEncoding.QOP == "qop"


class TestDatasetRequirement:
    def test_dataset_requirement_values(self):
        assert DatasetRequirement.REQUIRED == "dataset_required"
        assert DatasetRequirement.OPTIONAL == "dataset_optional"
        assert DatasetRequirement.NOT_REQUIRED == "dataset_not_required"


class TestOpType:
    def test_op_type_values(self):
        assert OpType.MatMul == "MatMul"
        assert OpType.Add == "Add"
        assert OpType.Custom == "custom"


class TestAccuracyLevel:
    def test_accuracy_level_values(self):
        assert AccuracyLevel.unset == 0
        assert AccuracyLevel.fp32 == 1
        assert AccuracyLevel.fp16 == 2
        assert AccuracyLevel.int8 == 4


class TestDiffusersModelVariant:
    def test_diffusers_variant_values(self):
        assert DiffusersModelVariant.AUTO == "auto"
        assert DiffusersModelVariant.SD == "sd"
        assert DiffusersModelVariant.FLUX == "flux"


class TestDiffusersComponent:
    def test_diffusers_component_values(self):
        assert DiffusersComponent.TEXT_ENCODER == "text_encoder"
        assert DiffusersComponent.UNET == "unet"
        assert DiffusersComponent.VAE_DECODER == "vae_decoder"


class TestPrecisionBitsFromPrecision:
    @pytest.mark.parametrize(
        ("precision", "expected"),
        [
            (Precision.INT4, PrecisionBits.BITS4),
            (Precision.INT8, PrecisionBits.BITS8),
            (Precision.INT16, PrecisionBits.BITS16),
            (Precision.INT32, PrecisionBits.BITS32),
            (Precision.UINT4, PrecisionBits.BITS4),
            (Precision.UINT8, PrecisionBits.BITS8),
            (Precision.UINT16, PrecisionBits.BITS16),
            (Precision.UINT32, PrecisionBits.BITS32),
        ],
    )
    def test_precision_to_bits_mapping(self, precision, expected):
        assert precision_bits_from_precision(precision) == expected

    @pytest.mark.parametrize("precision", [Precision.FP16, Precision.FP32, Precision.BF16, Precision.NF4])
    def test_precision_without_bits_mapping_returns_none(self, precision):
        assert precision_bits_from_precision(precision) is None


class TestMsftDomain:
    def test_msft_domain_value(self):
        assert MSFT_DOMAIN == "com.microsoft"
