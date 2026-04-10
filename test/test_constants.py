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
        # setup

        # execute
        onnx = Framework.ONNX
        pytorch = Framework.PYTORCH
        openvino = Framework.OPENVINO

        # assert
        assert onnx == "ONNX"
        assert pytorch == "PyTorch"
        assert openvino == "OpenVINO"

    def test_framework_str(self):
        # setup

        # execute
        result = str(Framework.ONNX)

        # assert
        assert result == "ONNX"

    def test_framework_all_members(self):
        # setup
        expected = {"ONNX", "PYTORCH", "QAIRT", "QNN", "TENSORFLOW", "OPENVINO"}

        # execute
        result = set(Framework.__members__.keys())

        # assert
        assert result == expected


class TestModelFileFormat:
    def test_model_file_format_values(self):
        # setup

        # execute
        onnx = ModelFileFormat.ONNX
        state_dict = ModelFileFormat.PYTORCH_STATE_DICT
        composite = ModelFileFormat.COMPOSITE_MODEL

        # assert
        assert onnx == "ONNX"
        assert state_dict == "PyTorch.StateDict"
        assert composite == "Composite"

    def test_model_file_format_str(self):
        # setup

        # execute
        result = str(ModelFileFormat.OPENVINO_IR)

        # assert
        assert result == "OpenVINO.IR"


class TestPrecision:
    def test_precision_values(self):
        # setup

        # execute
        int4 = Precision.INT4
        fp16 = Precision.FP16
        bf16 = Precision.BF16

        # assert
        assert int4 == "int4"
        assert fp16 == "fp16"
        assert bf16 == "bf16"

    def test_precision_all_members(self):
        # setup
        expected_count = 14

        # execute
        result = len(Precision)

        # assert
        assert result == expected_count


class TestPrecisionBits:
    def test_precision_bits_values(self):
        # setup

        # execute & assert
        assert PrecisionBits.BITS2 == 2
        assert PrecisionBits.BITS4 == 4
        assert PrecisionBits.BITS8 == 8
        assert PrecisionBits.BITS16 == 16
        assert PrecisionBits.BITS32 == 32

    def test_precision_bits_is_int(self):
        # setup

        # execute
        result = isinstance(PrecisionBits.BITS4.value, int)

        # assert
        assert result


class TestQuantAlgorithm:
    def test_quant_algorithm_case_insensitive(self):
        # setup

        # execute
        lower = QuantAlgorithm("awq")
        upper = QuantAlgorithm("AWQ")
        mixed = QuantAlgorithm("Awq")

        # assert
        assert lower == QuantAlgorithm.AWQ
        assert upper == QuantAlgorithm.AWQ
        assert mixed == QuantAlgorithm.AWQ

    def test_quant_algorithm_values(self):
        # setup

        # execute
        gptq = QuantAlgorithm.GPTQ
        rtn = QuantAlgorithm.RTN

        # assert
        assert gptq == "gptq"
        assert rtn == "rtn"

    def test_quant_algorithm_all_members(self):
        # setup
        expected = {"AWQ", "GPTQ", "HQQ", "RTN", "SPINQUANT", "QUAROT", "LPBQ", "SEQMSE", "ADAROUND"}

        # execute
        result = set(QuantAlgorithm.__members__.keys())

        # assert
        assert result == expected


class TestQuantEncoding:
    def test_quant_encoding_values(self):
        # setup

        # execute
        qdq = QuantEncoding.QDQ
        qop = QuantEncoding.QOP

        # assert
        assert qdq == "qdq"
        assert qop == "qop"


class TestDatasetRequirement:
    def test_dataset_requirement_values(self):
        # setup

        # execute
        required = DatasetRequirement.REQUIRED
        optional = DatasetRequirement.OPTIONAL
        not_required = DatasetRequirement.NOT_REQUIRED

        # assert
        assert required == "dataset_required"
        assert optional == "dataset_optional"
        assert not_required == "dataset_not_required"


class TestOpType:
    def test_op_type_values(self):
        # setup

        # execute
        matmul = OpType.MatMul
        add = OpType.Add
        custom = OpType.Custom

        # assert
        assert matmul == "MatMul"
        assert add == "Add"
        assert custom == "custom"


class TestAccuracyLevel:
    def test_accuracy_level_values(self):
        # setup

        # execute & assert
        assert AccuracyLevel.unset == 0
        assert AccuracyLevel.fp32 == 1
        assert AccuracyLevel.fp16 == 2
        assert AccuracyLevel.int8 == 4


class TestDiffusersModelVariant:
    def test_diffusers_variant_values(self):
        # setup

        # execute
        auto = DiffusersModelVariant.AUTO
        sd = DiffusersModelVariant.SD
        flux = DiffusersModelVariant.FLUX

        # assert
        assert auto == "auto"
        assert sd == "sd"
        assert flux == "flux"


class TestDiffusersComponent:
    def test_diffusers_component_values(self):
        # setup

        # execute
        text_encoder = DiffusersComponent.TEXT_ENCODER
        unet = DiffusersComponent.UNET
        vae_decoder = DiffusersComponent.VAE_DECODER

        # assert
        assert text_encoder == "text_encoder"
        assert unet == "unet"
        assert vae_decoder == "vae_decoder"


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
        # setup

        # execute
        result = precision_bits_from_precision(precision)

        # assert
        assert result == expected

    @pytest.mark.parametrize("precision", [Precision.FP16, Precision.FP32, Precision.BF16, Precision.NF4])
    def test_precision_without_bits_mapping_returns_none(self, precision):
        # setup

        # execute
        result = precision_bits_from_precision(precision)

        # assert
        assert result is None


class TestMsftDomain:
    def test_msft_domain_value(self):
        # setup

        # execute
        result = MSFT_DOMAIN

        # assert
        assert result == "com.microsoft"
