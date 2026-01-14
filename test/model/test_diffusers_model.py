# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from unittest.mock import MagicMock, patch

import pytest

from olive.constants import DiffusersModelVariant
from olive.model.handler.diffusers import DiffusersModelHandler


class TestDiffusersModelHandler:
    model_path = "runwayml/stable-diffusion-v1-5"

    def test_model_to_json(self):
        model = DiffusersModelHandler(model_path=self.model_path, model_variant=DiffusersModelVariant.SD)
        model_json = model.to_json()
        assert model_json["config"]["model_path"] == self.model_path
        assert model_json["config"]["model_variant"] == "sd"

    def test_model_to_json_with_adapter(self):
        model = DiffusersModelHandler(
            model_path=self.model_path,
            model_variant=DiffusersModelVariant.SD,
            adapter_path="/path/to/lora",
        )
        model_json = model.to_json()
        assert model_json["config"]["model_path"] == self.model_path
        assert model_json["config"]["adapter_path"] == "/path/to/lora"

    def test_model_to_json_with_load_kwargs(self):
        model = DiffusersModelHandler(
            model_path=self.model_path,
            model_variant=DiffusersModelVariant.SD,
            load_kwargs={"torch_dtype": "float16", "variant": "fp16"},
        )
        model_json = model.to_json()
        assert model_json["config"]["load_kwargs"] == {"torch_dtype": "float16", "variant": "fp16"}

    @pytest.mark.parametrize(
        ("model_variant", "expected"),
        [
            ("sd", DiffusersModelVariant.SD),
            ("sdxl", DiffusersModelVariant.SDXL),
            ("flux", DiffusersModelVariant.FLUX),
            (DiffusersModelVariant.SD, DiffusersModelVariant.SD),
            (DiffusersModelVariant.SDXL, DiffusersModelVariant.SDXL),
            (DiffusersModelVariant.FLUX, DiffusersModelVariant.FLUX),
        ],
    )
    def test_detected_model_variant_explicit(self, model_variant, expected):
        model = DiffusersModelHandler(model_path=self.model_path, model_variant=model_variant)
        assert model.detected_model_variant == expected

    @patch("olive.model.handler.diffusers.is_valid_diffusers_model", return_value=True)
    @pytest.mark.parametrize(
        ("model_path", "mock_config", "expected"),
        [
            # SDXL: unet config with cross_attention_dim >= 2048
            (
                "stabilityai/stable-diffusion-xl-base-1.0",
                {"unet": {"cross_attention_dim": 2048}},
                DiffusersModelVariant.SDXL,
            ),
            # SD: unet config with cross_attention_dim < 2048
            ("runwayml/stable-diffusion-v1-5", {"unet": {"cross_attention_dim": 768}}, DiffusersModelVariant.SD),
            # Flux: transformer config with Flux class name
            (
                "black-forest-labs/FLUX.1-dev",
                {"transformer": {"_class_name": "FluxTransformer2DModel"}},
                DiffusersModelVariant.FLUX,
            ),
            # SD3: transformer config with SD3 class name
            (
                "stabilityai/stable-diffusion-3-medium",
                {"transformer": {"_class_name": "SD3Transformer2DModel"}},
                DiffusersModelVariant.SD3,
            ),
            # Sana: transformer config with Sana class name
            ("some-sana-model", {"transformer": {"_class_name": "SanaTransformer2DModel"}}, DiffusersModelVariant.SANA),
        ],
    )
    def test_detected_model_variant_auto_from_config(self, mock_is_valid, model_path, mock_config, expected):
        def mock_load_config(cls_or_path, subfolder=None):
            if subfolder in mock_config:
                return mock_config[subfolder]
            raise FileNotFoundError(f"Config not found for {subfolder}")

        with patch("diffusers.ConfigMixin.load_config", side_effect=mock_load_config):
            model = DiffusersModelHandler(model_path=model_path, model_variant=DiffusersModelVariant.AUTO)
            assert model.detected_model_variant == expected

    @patch("olive.model.handler.diffusers.is_valid_diffusers_model", return_value=True)
    @patch("diffusers.ConfigMixin.load_config", side_effect=FileNotFoundError("not found"))
    def test_detected_model_variant_auto_raises_error(self, mock_load_config, mock_is_valid):
        model = DiffusersModelHandler(model_path="some-random-model", model_variant=DiffusersModelVariant.AUTO)
        with pytest.raises(ValueError, match="Cannot detect model variant"):
            _ = model.detected_model_variant

    def test_default_model_variant_is_auto(self):
        model = DiffusersModelHandler(model_path=self.model_path)
        assert model.model_variant == DiffusersModelVariant.AUTO

    @patch("diffusers.DiffusionPipeline")
    def test_load_model(self, mock_diffusion_pipeline):
        mock_pipeline = MagicMock()
        mock_diffusion_pipeline.from_pretrained.return_value = mock_pipeline

        model = DiffusersModelHandler(model_path=self.model_path, model_variant=DiffusersModelVariant.SD)
        result = model.load_model()

        assert result == mock_pipeline
        mock_diffusion_pipeline.from_pretrained.assert_called_once_with(self.model_path)

    @patch("diffusers.DiffusionPipeline")
    def test_load_model_with_kwargs(self, mock_diffusion_pipeline):
        mock_pipeline = MagicMock()
        mock_diffusion_pipeline.from_pretrained.return_value = mock_pipeline

        model = DiffusersModelHandler(
            model_path=self.model_path,
            model_variant=DiffusersModelVariant.SD,
            load_kwargs={"torch_dtype": "float16"},
        )
        model.load_model()

        mock_diffusion_pipeline.from_pretrained.assert_called_once_with(self.model_path, torch_dtype="float16")

    @patch("diffusers.DiffusionPipeline")
    def test_load_model_with_adapter(self, mock_diffusion_pipeline):
        mock_pipeline = MagicMock()
        mock_diffusion_pipeline.from_pretrained.return_value = mock_pipeline

        adapter_path = "/path/to/lora"
        model = DiffusersModelHandler(
            model_path=self.model_path,
            model_variant=DiffusersModelVariant.SD,
            adapter_path=adapter_path,
        )
        model.load_model()

        mock_diffusion_pipeline.from_pretrained.assert_called_once_with(self.model_path)
        mock_pipeline.load_lora_weights.assert_called_once_with(adapter_path)

    @patch("diffusers.DiffusionPipeline")
    def test_get_component(self, mock_diffusion_pipeline):
        mock_pipeline = MagicMock()
        mock_pipeline.unet = MagicMock()
        mock_pipeline.vae = MagicMock()
        mock_diffusion_pipeline.from_pretrained.return_value = mock_pipeline

        model = DiffusersModelHandler(model_path=self.model_path, model_variant=DiffusersModelVariant.SD)

        unet = model.get_component("unet")
        vae = model.get_component("vae")

        assert unet == mock_pipeline.unet
        assert vae == mock_pipeline.vae

    @patch("diffusers.DiffusionPipeline")
    def test_get_component_not_found(self, mock_diffusion_pipeline):
        mock_pipeline = MagicMock(spec=["unet", "vae"])
        mock_diffusion_pipeline.from_pretrained.return_value = mock_pipeline

        model = DiffusersModelHandler(model_path=self.model_path, model_variant=DiffusersModelVariant.SD)

        with pytest.raises(ValueError, match="Component 'nonexistent' not found"):
            model.get_component("nonexistent")

    @patch("diffusers.DiffusionPipeline")
    def test_run_session_with_dict_inputs(self, mock_diffusion_pipeline):
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = "output"
        mock_diffusion_pipeline.from_pretrained.return_value = mock_pipeline

        model = DiffusersModelHandler(model_path=self.model_path, model_variant=DiffusersModelVariant.SD)

        inputs = {"prompt": "a cat", "num_inference_steps": 20}
        result = model.run_session(inputs=inputs)

        mock_pipeline.assert_called_once_with(prompt="a cat", num_inference_steps=20)
        assert result == "output"

    @patch("diffusers.DiffusionPipeline")
    def test_run_session_with_list_inputs(self, mock_diffusion_pipeline):
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = "output"
        mock_diffusion_pipeline.from_pretrained.return_value = mock_pipeline

        model = DiffusersModelHandler(model_path=self.model_path, model_variant=DiffusersModelVariant.SD)

        inputs = ["a cat"]
        result = model.run_session(inputs=inputs)

        mock_pipeline.assert_called_once_with("a cat")
        assert result == "output"

    def test_adapter_path_property(self):
        model = DiffusersModelHandler(
            model_path=self.model_path,
            model_variant=DiffusersModelVariant.SD,
            adapter_path="/path/to/lora",
        )
        assert model.adapter_path == "/path/to/lora"

    def test_adapter_path_property_none(self):
        model = DiffusersModelHandler(model_path=self.model_path, model_variant=DiffusersModelVariant.SD)
        assert model.adapter_path is None
