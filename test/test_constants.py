# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import pytest

from olive.constants import (
    Framework,
    Precision,
    PrecisionBits,
    QuantAlgorithm,
    precision_bits_from_precision,
)


class TestFramework:
    def test_framework_all_members(self):
        # setup
        expected = {"ONNX", "PYTORCH", "QAIRT", "QNN", "TENSORFLOW", "OPENVINO"}

        # execute
        result = set(Framework.__members__.keys())

        # assert
        assert result == expected


class TestPrecision:
    def test_precision_all_members(self):
        # setup
        expected_count = 14

        # execute
        result = len(Precision)

        # assert
        assert result == expected_count


class TestQuantAlgorithm:
    def test_quant_algorithm_case_insensitive(self):
        # execute
        lower = QuantAlgorithm("awq")
        upper = QuantAlgorithm("AWQ")
        mixed = QuantAlgorithm("Awq")

        # assert
        assert lower == QuantAlgorithm.AWQ
        assert upper == QuantAlgorithm.AWQ
        assert mixed == QuantAlgorithm.AWQ

    def test_quant_algorithm_all_members(self):
        # setup
        expected = {"AWQ", "GPTQ", "HQQ", "RTN", "SPINQUANT", "QUAROT", "LPBQ", "SEQMSE", "ADAROUND"}

        # execute
        result = set(QuantAlgorithm.__members__.keys())

        # assert
        assert result == expected


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
        # execute
        result = precision_bits_from_precision(precision)

        # assert
        assert result == expected

    @pytest.mark.parametrize("precision", [Precision.FP16, Precision.FP32, Precision.BF16, Precision.NF4])
    def test_precision_without_bits_mapping_returns_none(self, precision):
        # execute
        result = precision_bits_from_precision(precision)

        # assert
        assert result is None
