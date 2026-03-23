# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
from unittest.mock import patch


class TestExportFlow:
    @patch("olive.cli.init.diffusers_flow._ask")
    def test_export_with_variant(self, mock_ask):
        from olive.cli.init.diffusers_flow import _export_flow

        mock_ask.return_value = "float16"
        result = _export_flow("stabilityai/sdxl", "sdxl")
        cmd = result["command"]
        assert "olive capture-onnx-graph -m stabilityai/sdxl" in cmd
        assert "--torch_dtype float16" in cmd
        assert "--model_variant sdxl" in cmd

    @patch("olive.cli.init.diffusers_flow._ask")
    def test_export_auto_variant(self, mock_ask):
        from olive.cli.init.diffusers_flow import _export_flow
        from olive.cli.init.helpers import DiffuserVariant

        mock_ask.return_value = "float32"
        result = _export_flow("my-model", DiffuserVariant.AUTO)
        assert "--model_variant" not in result["command"]


class TestLoraFlow:
    @patch("olive.cli.init.diffusers_flow._ask")
    def test_basic_lora_local_data(self, mock_ask):
        from olive.cli.init.diffusers_flow import _lora_flow
        from olive.cli.init.helpers import DiffuserVariant, SourceType

        mock_ask.side_effect = [
            "16",  # lora_r
            "16",  # lora_alpha
            "0.0",  # lora_dropout
            SourceType.LOCAL,  # data_source
            "/images",  # data_dir
            False,  # enable_dreambooth
            "1000",  # max_train_steps
            "1e-4",  # learning_rate
            "1",  # train_batch_size
            "4",  # gradient_accumulation
            "bf16",  # mixed_precision
            "constant",  # lr_scheduler
            "0",  # warmup_steps
            "",  # seed (skip)
            False,  # merge_lora
        ]
        result = _lora_flow("my-model", DiffuserVariant.AUTO)
        cmd = result["command"]
        assert "olive diffusion-lora -m my-model" in cmd
        assert "-r 16 --alpha 16" in cmd
        assert "-d /images" in cmd
        assert "--max_train_steps 1000" in cmd
        assert "--model_variant" not in cmd

    @patch("olive.cli.init.diffusers_flow._ask")
    def test_flux_with_dreambooth(self, mock_ask):
        from olive.cli.init.diffusers_flow import _lora_flow
        from olive.cli.init.helpers import DiffuserVariant, SourceType

        mock_ask.side_effect = [
            "16",  # lora_r
            "16",  # lora_alpha
            "0.1",  # lora_dropout
            SourceType.LOCAL,  # data_source
            "/images",  # data_dir
            True,  # enable_dreambooth
            "a photo of sks dog",  # instance_prompt
            True,  # with_prior
            "a photo of a dog",  # class_prompt
            "",  # class_data_dir (skip)
            "200",  # num_class_images
            "500",  # max_train_steps
            "1e-4",  # learning_rate
            "1",  # train_batch_size
            "4",  # gradient_accumulation
            "bf16",  # mixed_precision
            "constant",  # lr_scheduler
            "0",  # warmup_steps
            "",  # seed (skip)
            "3.5",  # guidance_scale (flux-specific)
            True,  # merge_lora
        ]
        result = _lora_flow("my-flux-model", DiffuserVariant.FLUX)
        cmd = result["command"]
        assert f"--model_variant {DiffuserVariant.FLUX}" in cmd
        assert "--dreambooth" in cmd
        assert '--instance_prompt "a photo of sks dog"' in cmd
        assert "--with_prior_preservation" in cmd
        assert "--guidance_scale 3.5" in cmd
        assert "--merge_lora" in cmd

    @patch("olive.cli.init.diffusers_flow._ask")
    def test_hf_data_source_with_caption(self, mock_ask):
        from olive.cli.init.diffusers_flow import _lora_flow
        from olive.cli.init.helpers import DiffuserVariant, SourceType

        mock_ask.side_effect = [
            "16",  # lora_r
            "16",  # lora_alpha
            "0.0",  # lora_dropout
            SourceType.HF,  # data_source
            "linoyts/Tuxemon",  # data_name
            "train",  # data_split
            "image",  # image_column
            "caption",  # caption_column
            False,  # enable_dreambooth
            "1000",  # max_train_steps
            "1e-4",  # learning_rate
            "1",  # train_batch_size
            "4",  # gradient_accumulation
            "bf16",  # mixed_precision
            "constant",  # lr_scheduler
            "0",  # warmup_steps
            "42",  # seed (provided)
            False,  # merge_lora
        ]
        result = _lora_flow("my-model", DiffuserVariant.AUTO)
        cmd = result["command"]
        assert "--data_name linoyts/Tuxemon" in cmd
        assert "--data_split train" in cmd
        assert "--image_column image" in cmd
        assert "--caption_column caption" in cmd
        assert "--seed 42" in cmd

    @patch("olive.cli.init.diffusers_flow._ask")
    def test_custom_max_train_steps(self, mock_ask):
        from olive.cli.init.diffusers_flow import TRAIN_STEPS_CUSTOM, _lora_flow
        from olive.cli.init.helpers import DiffuserVariant, SourceType

        mock_ask.side_effect = [
            "16",  # lora_r
            "16",  # lora_alpha
            "0.0",  # lora_dropout
            SourceType.LOCAL,  # data_source
            "/images",  # data_dir
            False,  # enable_dreambooth
            TRAIN_STEPS_CUSTOM,  # max_train_steps
            "3000",  # custom value
            "1e-4",  # learning_rate
            "1",  # train_batch_size
            "4",  # gradient_accumulation
            "bf16",  # mixed_precision
            "constant",  # lr_scheduler
            "0",  # warmup_steps
            "",  # seed (skip)
            False,  # merge_lora
        ]
        result = _lora_flow("my-model", DiffuserVariant.AUTO)
        assert "--max_train_steps 3000" in result["command"]

    @patch("olive.cli.init.diffusers_flow._ask")
    def test_dreambooth_with_class_data_dir(self, mock_ask):
        from olive.cli.init.diffusers_flow import _lora_flow
        from olive.cli.init.helpers import DiffuserVariant, SourceType

        mock_ask.side_effect = [
            "16",  # lora_r
            "16",  # lora_alpha
            "0.0",  # lora_dropout
            SourceType.LOCAL,  # data_source
            "/images",  # data_dir
            True,  # enable_dreambooth
            "a photo of sks dog",  # instance_prompt
            True,  # with_prior
            "a photo of a dog",  # class_prompt
            "/class_images",  # class_data_dir (provided)
            "200",  # num_class_images
            "1000",  # max_train_steps
            "1e-4",  # learning_rate
            "1",  # train_batch_size
            "4",  # gradient_accumulation
            "bf16",  # mixed_precision
            "constant",  # lr_scheduler
            "0",  # warmup_steps
            "",  # seed (skip)
            False,  # merge_lora
        ]
        result = _lora_flow("my-model", DiffuserVariant.AUTO)
        assert "--class_data_dir /class_images" in result["command"]


class TestRunDiffusersFlowRouting:
    @patch("olive.cli.init.diffusers_flow._export_flow")
    @patch("olive.cli.init.diffusers_flow._ask_select")
    def test_routes_to_export(self, mock_select, mock_flow):
        from olive.cli.init.diffusers_flow import OP_EXPORT, run_diffusers_flow

        mock_select.return_value = OP_EXPORT
        mock_flow.return_value = {"command": "test"}
        run_diffusers_flow({"model_path": "m", "variant": "sdxl"})
        mock_flow.assert_called_once_with("m", "sdxl")

    @patch("olive.cli.init.diffusers_flow._lora_flow")
    @patch("olive.cli.init.diffusers_flow._ask_select")
    def test_routes_to_lora(self, mock_select, mock_flow):
        from olive.cli.init.diffusers_flow import OP_LORA, run_diffusers_flow
        from olive.cli.init.helpers import DiffuserVariant

        mock_select.return_value = OP_LORA
        mock_flow.return_value = {"command": "test"}
        run_diffusers_flow({"model_path": "m", "variant": DiffuserVariant.FLUX})
        mock_flow.assert_called_once_with("m", DiffuserVariant.FLUX)

    @patch("olive.cli.init.diffusers_flow._ask_select", return_value="unknown")
    def test_unknown_operation_returns_empty(self, mock_select):
        from olive.cli.init.diffusers_flow import run_diffusers_flow
        from olive.cli.init.helpers import DiffuserVariant

        result = run_diffusers_flow({"model_path": "m", "variant": DiffuserVariant.AUTO})
        assert not result
