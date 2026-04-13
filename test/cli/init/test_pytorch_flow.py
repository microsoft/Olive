# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
from unittest.mock import patch


class TestBuildModelArgs:
    def test_model_path_only(self):
        from olive.cli.init.pytorch_flow import _build_model_args

        result = _build_model_args({"model_path": "meta-llama/Llama-3.1-8B"})
        assert result == "-m meta-llama/Llama-3.1-8B"

    def test_with_model_script(self):
        from olive.cli.init.pytorch_flow import _build_model_args

        config = {"model_path": "my_model", "model_script": "train.py", "script_dir": "./src"}
        result = _build_model_args(config)
        assert "-m my_model" in result
        assert "--model_script train.py" in result
        assert "--script_dir ./src" in result

    def test_script_only_no_model_path(self):
        from olive.cli.init.pytorch_flow import _build_model_args

        config = {"model_script": "train.py"}
        result = _build_model_args(config)
        assert not result.startswith("-m ")
        assert "--model_script train.py" in result

    def test_empty_config(self):
        from olive.cli.init.pytorch_flow import _build_model_args

        result = _build_model_args({})
        assert result == ""


class TestBuildExportCommand:
    def test_model_builder(self):
        from olive.cli.init.pytorch_flow import EXPORTER_MODEL_BUILDER, _build_export_command

        cmd = _build_export_command("-m model", {"exporter": EXPORTER_MODEL_BUILDER, "precision": "fp16"})
        assert cmd == "olive capture-onnx-graph -m model --use_model_builder --precision fp16"

    def test_model_builder_int4_with_block_size(self):
        from olive.cli.init.pytorch_flow import EXPORTER_MODEL_BUILDER, PRECISION_INT4, _build_export_command

        config = {"exporter": EXPORTER_MODEL_BUILDER, "precision": PRECISION_INT4, "int4_block_size": "32"}
        cmd = _build_export_command("-m model", config)
        assert "--use_model_builder --precision int4" in cmd
        assert "--int4_block_size 32" in cmd

    def test_dynamo(self):
        from olive.cli.init.pytorch_flow import EXPORTER_DYNAMO, _build_export_command

        cmd = _build_export_command("-m model", {"exporter": EXPORTER_DYNAMO, "torch_dtype": "float16"})
        assert cmd == "olive capture-onnx-graph -m model --torch_dtype float16"

    def test_torchscript(self):
        from olive.cli.init.pytorch_flow import EXPORTER_TORCHSCRIPT, _build_export_command

        cmd = _build_export_command("-m model", {"exporter": EXPORTER_TORCHSCRIPT})
        assert cmd == "olive capture-onnx-graph -m model"


class TestBuildQuantizeCommand:
    def test_rtn_no_implementation(self):
        from olive.cli.init.pytorch_flow import _build_quantize_command

        cmd = _build_quantize_command("-m model", {"algorithm": "rtn", "precision": "int4"})
        assert cmd == "olive quantize -m model --algorithm rtn --precision int4"
        assert "--implementation" not in cmd

    def test_awq_with_implementation(self):
        from olive.cli.init.pytorch_flow import _build_quantize_command

        cmd = _build_quantize_command("-m model", {"algorithm": "awq", "precision": "int4"})
        assert "--algorithm awq" in cmd
        assert "--implementation awq" in cmd

    def test_quarot_with_implementation(self):
        from olive.cli.init.pytorch_flow import _build_quantize_command

        cmd = _build_quantize_command("-m model", {"algorithm": "quarot", "precision": "int4"})
        assert "--implementation quarot" in cmd

    def test_spinquant_with_implementation(self):
        from olive.cli.init.pytorch_flow import _build_quantize_command

        cmd = _build_quantize_command("-m model", {"algorithm": "spinquant", "precision": "int4"})
        assert "--implementation spinquant" in cmd

    def test_with_calibration(self):
        from olive.cli.init.helpers import SourceType
        from olive.cli.init.pytorch_flow import _build_quantize_command

        config = {
            "algorithm": "gptq",
            "precision": "int4",
            "calibration": {
                "source": SourceType.HF,
                "data_name": "Salesforce/wikitext",
                "subset": "wikitext-2-raw-v1",
                "split": "train",
                "num_samples": "128",
            },
        }
        cmd = _build_quantize_command("-m model", config)
        assert "--algorithm gptq" in cmd
        assert "--implementation" not in cmd  # gptq uses default olive
        assert "-d Salesforce/wikitext" in cmd
        assert "--split train" in cmd


class TestOptimizeAutoMode:
    @patch("olive.cli.init.pytorch_flow._ask")
    def test_generates_command(self, mock_ask):
        from olive.cli.init.pytorch_flow import _optimize_auto_mode

        mock_ask.side_effect = ["CUDAExecutionProvider", "int4"]
        result = _optimize_auto_mode({"model_path": "my-model"})
        assert result["command"] == "olive optimize -m my-model --provider CUDAExecutionProvider --precision int4"


class TestQuantizeFlow:
    @patch("olive.cli.init.pytorch_flow.prompt_calibration_source")
    @patch("olive.cli.init.pytorch_flow._ask")
    def test_rtn_no_calibration(self, mock_ask, mock_calib):
        from olive.cli.init.pytorch_flow import _quantize_flow

        mock_ask.side_effect = ["rtn", "int4"]
        result = _quantize_flow({"model_path": "my-model"})
        assert "--algorithm rtn" in result["command"]
        assert "--implementation" not in result["command"]
        mock_calib.assert_not_called()

    @patch("olive.cli.init.pytorch_flow.prompt_calibration_source", return_value=None)
    @patch("olive.cli.init.pytorch_flow._ask")
    def test_awq_with_default_calibration(self, mock_ask, mock_calib):
        from olive.cli.init.pytorch_flow import _quantize_flow

        mock_ask.side_effect = ["awq", "int4"]
        result = _quantize_flow({"model_path": "my-model"})
        assert "--algorithm awq" in result["command"]
        assert "--implementation awq" in result["command"]
        mock_calib.assert_called_once()

    @patch(
        "olive.cli.init.pytorch_flow.build_calibration_args", return_value=" -d data --split train --max_samples 128"
    )
    @patch("olive.cli.init.pytorch_flow.prompt_calibration_source")
    @patch("olive.cli.init.pytorch_flow._ask")
    def test_gptq_with_calibration(self, mock_ask, mock_calib, mock_build):
        from olive.cli.init.helpers import SourceType
        from olive.cli.init.pytorch_flow import _quantize_flow

        mock_calib.return_value = {
            "source": SourceType.HF,
            "data_name": "data",
            "subset": "",
            "split": "train",
            "num_samples": "128",
        }
        mock_ask.side_effect = ["gptq", "int4"]
        result = _quantize_flow({"model_path": "my-model"})
        assert "--algorithm gptq" in result["command"]
        assert "-d data" in result["command"]


class TestExportFlow:
    @patch("olive.cli.init.pytorch_flow._ask")
    def test_dynamo_exporter(self, mock_ask):
        from olive.cli.init.pytorch_flow import EXPORTER_DYNAMO, _export_flow

        mock_ask.side_effect = [EXPORTER_DYNAMO, "float16"]
        result = _export_flow({"model_path": "my-model"})
        assert result["command"] == "olive capture-onnx-graph -m my-model --torch_dtype float16"

    @patch("olive.cli.init.pytorch_flow._ask")
    def test_model_builder_fp16(self, mock_ask):
        from olive.cli.init.pytorch_flow import EXPORTER_MODEL_BUILDER, _export_flow

        mock_ask.side_effect = [EXPORTER_MODEL_BUILDER, "fp16"]
        result = _export_flow({"model_path": "my-model"})
        assert "--use_model_builder --precision fp16" in result["command"]

    @patch("olive.cli.init.pytorch_flow._ask")
    def test_model_builder_int4(self, mock_ask):
        from olive.cli.init.pytorch_flow import EXPORTER_MODEL_BUILDER, PRECISION_INT4, _export_flow

        mock_ask.side_effect = [EXPORTER_MODEL_BUILDER, PRECISION_INT4, "32", "4"]
        result = _export_flow({"model_path": "my-model"})
        assert "--precision int4" in result["command"]
        assert "--int4_block_size 32" in result["command"]
        assert "--int4_accuracy_level 4" in result["command"]

    @patch("olive.cli.init.pytorch_flow._ask")
    def test_torchscript(self, mock_ask):
        from olive.cli.init.pytorch_flow import EXPORTER_TORCHSCRIPT, _export_flow

        mock_ask.side_effect = [EXPORTER_TORCHSCRIPT]
        result = _export_flow({"model_path": "my-model"})
        assert result["command"] == "olive capture-onnx-graph -m my-model"

    @patch("olive.cli.init.pytorch_flow._ask")
    def test_with_model_script(self, mock_ask):
        from olive.cli.init.pytorch_flow import EXPORTER_DYNAMO, _export_flow

        mock_ask.side_effect = [EXPORTER_DYNAMO, "float32"]
        result = _export_flow({"model_script": "script.py", "script_dir": "./src"})
        assert "--model_script script.py" in result["command"]
        assert "--script_dir ./src" in result["command"]


class TestFinetuneFlow:
    @patch("olive.cli.init.pytorch_flow._ask")
    def test_lora_hf_dataset(self, mock_ask):
        from olive.cli.init.helpers import SourceType
        from olive.cli.init.pytorch_flow import TEXT_FIELD, _finetune_flow

        mock_ask.side_effect = [
            "lora",  # method
            "64",  # lora_r
            "16",  # lora_alpha
            SourceType.HF,  # data_source
            "tatsu-lab/alpaca",  # data_name
            "train",  # train_split
            "",  # eval_split (skip)
            TEXT_FIELD,  # text_mode
            "text",  # text_field
            "1024",  # max_seq_len
            "256",  # max_samples
            "bfloat16",  # torch_dtype
        ]
        result = _finetune_flow({"model_path": "my-model"})
        cmd = result["command"]
        assert "olive finetune -m my-model" in cmd
        assert "--method lora" in cmd
        assert "-d tatsu-lab/alpaca" in cmd
        assert "--train_split train" in cmd
        assert "--text_field text" in cmd

    @patch("olive.cli.init.pytorch_flow._ask")
    def test_qlora_local_data_template(self, mock_ask):
        from olive.cli.init.helpers import SourceType
        from olive.cli.init.pytorch_flow import TEXT_TEMPLATE, _finetune_flow

        mock_ask.side_effect = [
            "qlora",  # method
            "16",  # lora_r
            "16",  # lora_alpha
            SourceType.LOCAL,  # data_source
            "/data/train.json",  # data_files
            TEXT_TEMPLATE,  # text_mode
            "Q: {q} A: {a}",  # template
            "512",  # max_seq_len
            "100",  # max_samples
            "float16",  # torch_dtype
        ]
        result = _finetune_flow({"model_path": "my-model"})
        cmd = result["command"]
        assert "--method qlora" in cmd
        assert "--data_files /data/train.json" in cmd
        assert '--text_template "Q: {q} A: {a}"' in cmd

    @patch("olive.cli.init.pytorch_flow._ask")
    def test_hf_with_eval_split(self, mock_ask):
        from olive.cli.init.helpers import SourceType
        from olive.cli.init.pytorch_flow import TEXT_FIELD, _finetune_flow

        mock_ask.side_effect = [
            "lora",  # method
            "64",  # lora_r
            "16",  # lora_alpha
            SourceType.HF,  # data_source
            "tatsu-lab/alpaca",  # data_name
            "train",  # train_split
            "test",  # eval_split (provided)
            TEXT_FIELD,  # text_mode
            "text",  # text_field
            "1024",  # max_seq_len
            "256",  # max_samples
            "bfloat16",  # torch_dtype
        ]
        result = _finetune_flow({"model_path": "my-model"})
        assert "--eval_split test" in result["command"]

    @patch("olive.cli.init.pytorch_flow._ask")
    def test_chat_template(self, mock_ask):
        from olive.cli.init.helpers import SourceType
        from olive.cli.init.pytorch_flow import TEXT_CHAT_TEMPLATE, _finetune_flow

        mock_ask.side_effect = [
            "lora",  # method
            "64",  # lora_r
            "16",  # lora_alpha
            SourceType.HF,  # data_source
            "dataset",  # data_name
            "train",  # train_split
            "",  # eval_split (skip)
            TEXT_CHAT_TEMPLATE,  # text_mode
            "1024",  # max_seq_len
            "256",  # max_samples
            "bfloat16",  # torch_dtype
        ]
        result = _finetune_flow({"model_path": "my-model"})
        assert "--use_chat_template" in result["command"]


class TestOptimizeCustomMode:
    @patch("olive.cli.init.pytorch_flow._ask")
    def test_export_and_quantize(self, mock_ask):
        from olive.cli.init.pytorch_flow import EXPORTER_DYNAMO, OP_EXPORT, OP_QUANTIZE, _optimize_custom_mode

        mock_ask.side_effect = [
            [OP_EXPORT, OP_QUANTIZE],  # operations checkbox
            EXPORTER_DYNAMO,  # exporter
            "float32",  # torch_dtype
            "rtn",  # algorithm
            "int4",  # precision
            "CUDAExecutionProvider",  # provider
        ]
        result = _optimize_custom_mode({"model_path": "my-model"})
        cmd = result["command"]
        assert "olive optimize" in cmd
        assert "--provider CUDAExecutionProvider" in cmd
        assert "--exporter dynamo_exporter" in cmd

    @patch("olive.cli.init.pytorch_flow._ask")
    def test_export_only(self, mock_ask):
        from olive.cli.init.pytorch_flow import EXPORTER_DYNAMO, OP_EXPORT, _optimize_custom_mode

        mock_ask.side_effect = [
            [OP_EXPORT],  # operations checkbox
            EXPORTER_DYNAMO,  # exporter
            "float16",  # torch_dtype
        ]
        result = _optimize_custom_mode({"model_path": "my-model"})
        assert "olive capture-onnx-graph" in result["command"]
        assert "--torch_dtype float16" in result["command"]

    @patch(
        "olive.cli.init.pytorch_flow.build_calibration_args", return_value=" -d data --split train --max_samples 128"
    )
    @patch("olive.cli.init.pytorch_flow.prompt_calibration_source")
    @patch("olive.cli.init.pytorch_flow._ask")
    def test_quantize_only(self, mock_ask, mock_calib, mock_build):
        from olive.cli.init.helpers import SourceType
        from olive.cli.init.pytorch_flow import OP_QUANTIZE, _optimize_custom_mode

        mock_calib.return_value = {
            "source": SourceType.HF,
            "data_name": "data",
            "subset": "",
            "split": "train",
            "num_samples": "128",
        }
        mock_ask.side_effect = [
            [OP_QUANTIZE],  # operations checkbox
            "gptq",  # algorithm
            "int4",  # precision
        ]
        result = _optimize_custom_mode({"model_path": "my-model"})
        assert "olive quantize" in result["command"]
        assert "--algorithm gptq" in result["command"]

    @patch("olive.cli.init.pytorch_flow._ask")
    def test_graph_opt_only(self, mock_ask):
        from olive.cli.init.pytorch_flow import OP_GRAPH_OPT, _optimize_custom_mode

        mock_ask.side_effect = [
            [OP_GRAPH_OPT],  # operations checkbox
            "CPUExecutionProvider",  # provider
        ]
        result = _optimize_custom_mode({"model_path": "my-model"})
        assert "olive optimize" in result["command"]
        assert "--precision fp32" in result["command"]

    @patch("olive.cli.init.pytorch_flow._ask")
    def test_export_and_graph_opt(self, mock_ask):
        from olive.cli.init.pytorch_flow import EXPORTER_MODEL_BUILDER, OP_EXPORT, OP_GRAPH_OPT, _optimize_custom_mode

        mock_ask.side_effect = [
            [OP_EXPORT, OP_GRAPH_OPT],  # operations checkbox
            EXPORTER_MODEL_BUILDER,  # exporter
            "fp16",  # precision
            "CPUExecutionProvider",  # provider
        ]
        result = _optimize_custom_mode({"model_path": "my-model"})
        cmd = result["command"]
        assert "olive optimize" in cmd
        assert "--precision fp32" in cmd
        assert "--exporter model_builder" in cmd

    @patch("olive.cli.init.pytorch_flow._ask")
    def test_quantize_and_graph_opt(self, mock_ask):
        from olive.cli.init.pytorch_flow import OP_GRAPH_OPT, OP_QUANTIZE, _optimize_custom_mode

        mock_ask.side_effect = [
            [OP_QUANTIZE, OP_GRAPH_OPT],  # operations checkbox
            "rtn",  # algorithm
            "int4",  # precision
            "CUDAExecutionProvider",  # provider
        ]
        result = _optimize_custom_mode({"model_path": "my-model"})
        cmd = result["command"]
        assert "olive optimize" in cmd
        assert "--precision int4" in cmd

    @patch("olive.cli.init.pytorch_flow._ask")
    def test_no_operations_selected(self, mock_ask):
        from olive.cli.init.pytorch_flow import _optimize_custom_mode

        mock_ask.return_value = []  # empty checkbox
        result = _optimize_custom_mode({"model_path": "my-model"})
        assert not result


class TestOptimizeFlow:
    @patch("olive.cli.init.pytorch_flow._optimize_auto_mode")
    @patch("olive.cli.init.pytorch_flow._ask")
    def test_routes_to_auto(self, mock_ask, mock_auto):
        from olive.cli.init.pytorch_flow import MODE_AUTO, _optimize_flow

        mock_ask.return_value = MODE_AUTO
        mock_auto.return_value = {"command": "test"}
        _optimize_flow({"model_path": "m"})
        mock_auto.assert_called_once()

    @patch("olive.cli.init.pytorch_flow._optimize_custom_mode")
    @patch("olive.cli.init.pytorch_flow._ask")
    def test_routes_to_custom(self, mock_ask, mock_custom):
        from olive.cli.init.pytorch_flow import MODE_CUSTOM, _optimize_flow

        mock_ask.return_value = MODE_CUSTOM
        mock_custom.return_value = {"command": "test"}
        _optimize_flow({"model_path": "m"})
        mock_custom.assert_called_once()


class TestPromptExportOptionsInt4:
    @patch("olive.cli.init.pytorch_flow._ask")
    def test_model_builder_int4_block_size(self, mock_ask):
        from olive.cli.init.pytorch_flow import EXPORTER_MODEL_BUILDER, PRECISION_INT4, _prompt_export_options

        mock_ask.side_effect = [
            EXPORTER_MODEL_BUILDER,  # exporter
            PRECISION_INT4,  # precision
            "64",  # block_size
        ]
        config = _prompt_export_options()
        assert config == {"exporter": EXPORTER_MODEL_BUILDER, "precision": PRECISION_INT4, "int4_block_size": "64"}


class TestRunPytorchFlowRouting:
    @patch("olive.cli.init.pytorch_flow._optimize_flow")
    @patch("olive.cli.init.pytorch_flow._ask_select")
    def test_routes_to_optimize(self, mock_select, mock_flow):
        from olive.cli.init.pytorch_flow import OP_OPTIMIZE, run_pytorch_flow

        mock_select.return_value = OP_OPTIMIZE
        mock_flow.return_value = {"command": "test"}
        run_pytorch_flow({"model_path": "m"})
        mock_flow.assert_called_once_with({"model_path": "m"})

    @patch("olive.cli.init.pytorch_flow._export_flow")
    @patch("olive.cli.init.pytorch_flow._ask_select")
    def test_routes_to_export(self, mock_select, mock_flow):
        from olive.cli.init.pytorch_flow import OP_EXPORT, run_pytorch_flow

        mock_select.return_value = OP_EXPORT
        mock_flow.return_value = {"command": "test"}
        run_pytorch_flow({"model_path": "m"})
        mock_flow.assert_called_once()

    @patch("olive.cli.init.pytorch_flow._quantize_flow")
    @patch("olive.cli.init.pytorch_flow._ask_select")
    def test_routes_to_quantize(self, mock_select, mock_flow):
        from olive.cli.init.pytorch_flow import OP_QUANTIZE, run_pytorch_flow

        mock_select.return_value = OP_QUANTIZE
        mock_flow.return_value = {"command": "test"}
        run_pytorch_flow({"model_path": "m"})
        mock_flow.assert_called_once()

    @patch("olive.cli.init.pytorch_flow._finetune_flow")
    @patch("olive.cli.init.pytorch_flow._ask_select")
    def test_routes_to_finetune(self, mock_select, mock_flow):
        from olive.cli.init.pytorch_flow import OP_FINETUNE, run_pytorch_flow

        mock_select.return_value = OP_FINETUNE
        mock_flow.return_value = {"command": "test"}
        run_pytorch_flow({"model_path": "m"})
        mock_flow.assert_called_once()

    @patch("olive.cli.init.pytorch_flow._ask_select", return_value="unknown")
    def test_unknown_operation_returns_empty(self, mock_select):
        from olive.cli.init.pytorch_flow import run_pytorch_flow

        result = run_pytorch_flow({"model_path": "m"})
        assert not result
