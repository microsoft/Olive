# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from unittest.mock import MagicMock, patch

import pytest

from olive.model.handler.diffusers import DiffusersModelHandler, DiffusersModelType


class TestDiffusersModelHandler:
    model_path = "runwayml/stable-diffusion-v1-5"

    def test_model_to_json(self):
        model = DiffusersModelHandler(model_path=self.model_path, model_type="sd15")
        model_json = model.to_json()
        assert model_json["config"]["model_path"] == self.model_path
        assert model_json["config"]["model_type"] == "sd15"

    def test_model_to_json_with_adapter(self):
        model = DiffusersModelHandler(
            model_path=self.model_path,
            model_type="sd15",
            adapter_path="/path/to/lora",
        )
        model_json = model.to_json()
        assert model_json["config"]["model_path"] == self.model_path
        assert model_json["config"]["adapter_path"] == "/path/to/lora"

    def test_model_to_json_with_load_kwargs(self):
        model = DiffusersModelHandler(
            model_path=self.model_path,
            model_type="sd15",
            load_kwargs={"torch_dtype": "float16", "variant": "fp16"},
        )
        model_json = model.to_json()
        assert model_json["config"]["load_kwargs"] == {"torch_dtype": "float16", "variant": "fp16"}

    @pytest.mark.parametrize(
        ("model_type", "expected"),
        [
            ("sd15", DiffusersModelType.SD15),
            ("sdxl", DiffusersModelType.SDXL),
            ("flux", DiffusersModelType.FLUX),
            (DiffusersModelType.SD15, DiffusersModelType.SD15),
            (DiffusersModelType.SDXL, DiffusersModelType.SDXL),
            (DiffusersModelType.FLUX, DiffusersModelType.FLUX),
        ],
    )
    def test_detected_model_type_explicit(self, model_type, expected):
        model = DiffusersModelHandler(model_path=self.model_path, model_type=model_type)
        assert model.detected_model_type == expected

    @patch("olive.model.handler.diffusers.DiffusersModelHandler.is_valid_model", return_value=True)
    @patch("diffusers.UNet2DConditionModel.load_config", side_effect=Exception("not found"))
    @patch("diffusers.FluxTransformer2DModel.load_config", side_effect=Exception("not found"))
    @pytest.mark.parametrize(
        ("model_path", "expected"),
        [
            ("stabilityai/stable-diffusion-xl-base-1.0", DiffusersModelType.SDXL),
            ("some-model-sdxl-variant", DiffusersModelType.SDXL),
            ("black-forest-labs/FLUX.1-dev", DiffusersModelType.FLUX),
            ("some-flux-model", DiffusersModelType.FLUX),
            ("runwayml/stable-diffusion-v1-5", DiffusersModelType.SD15),
            ("my-custom-sd-model", DiffusersModelType.SD15),
        ],
    )
    def test_detected_model_type_auto_from_path(self, mock_flux, mock_unet, mock_is_valid, model_path, expected):
        model = DiffusersModelHandler(model_path=model_path, model_type=DiffusersModelType.AUTO)
        assert model.detected_model_type == expected

    @patch("olive.model.handler.diffusers.DiffusersModelHandler.is_valid_model", return_value=True)
    @patch("diffusers.UNet2DConditionModel.load_config", side_effect=Exception("not found"))
    @patch("diffusers.FluxTransformer2DModel.load_config", side_effect=Exception("not found"))
    def test_detected_model_type_auto_raises_error(self, mock_flux, mock_unet, mock_is_valid):
        model = DiffusersModelHandler(model_path="some-random-model", model_type=DiffusersModelType.AUTO)
        with pytest.raises(ValueError, match="Cannot detect model type"):
            _ = model.detected_model_type

    def test_default_model_type_is_auto(self):
        model = DiffusersModelHandler(model_path=self.model_path)
        assert model.model_type == DiffusersModelType.AUTO

    @patch("diffusers.DiffusionPipeline")
    def test_load_model(self, mock_diffusion_pipeline):
        mock_pipeline = MagicMock()
        mock_diffusion_pipeline.from_pretrained.return_value = mock_pipeline

        model = DiffusersModelHandler(model_path=self.model_path, model_type="sd15")
        result = model.load_model()

        assert result == mock_pipeline
        mock_diffusion_pipeline.from_pretrained.assert_called_once_with(self.model_path)

    @patch("diffusers.DiffusionPipeline")
    def test_load_model_with_kwargs(self, mock_diffusion_pipeline):
        mock_pipeline = MagicMock()
        mock_diffusion_pipeline.from_pretrained.return_value = mock_pipeline

        model = DiffusersModelHandler(
            model_path=self.model_path,
            model_type="sd15",
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
            model_type="sd15",
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

        model = DiffusersModelHandler(model_path=self.model_path, model_type="sd15")

        unet = model.get_component("unet")
        vae = model.get_component("vae")

        assert unet == mock_pipeline.unet
        assert vae == mock_pipeline.vae

    @patch("diffusers.DiffusionPipeline")
    def test_get_component_not_found(self, mock_diffusion_pipeline):
        mock_pipeline = MagicMock(spec=["unet", "vae"])
        mock_diffusion_pipeline.from_pretrained.return_value = mock_pipeline

        model = DiffusersModelHandler(model_path=self.model_path, model_type="sd15")

        with pytest.raises(ValueError, match="Component 'nonexistent' not found"):
            model.get_component("nonexistent")

    @patch("diffusers.DiffusionPipeline")
    def test_run_session_with_dict_inputs(self, mock_diffusion_pipeline):
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = "output"
        mock_diffusion_pipeline.from_pretrained.return_value = mock_pipeline

        model = DiffusersModelHandler(model_path=self.model_path, model_type="sd15")

        inputs = {"prompt": "a cat", "num_inference_steps": 20}
        result = model.run_session(inputs=inputs)

        mock_pipeline.assert_called_once_with(prompt="a cat", num_inference_steps=20)
        assert result == "output"

    @patch("diffusers.DiffusionPipeline")
    def test_run_session_with_list_inputs(self, mock_diffusion_pipeline):
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = "output"
        mock_diffusion_pipeline.from_pretrained.return_value = mock_pipeline

        model = DiffusersModelHandler(model_path=self.model_path, model_type="sd15")

        inputs = ["a cat"]
        result = model.run_session(inputs=inputs)

        mock_pipeline.assert_called_once_with("a cat")
        assert result == "output"

    def test_adapter_path_property(self):
        model = DiffusersModelHandler(
            model_path=self.model_path,
            model_type="sd15",
            adapter_path="/path/to/lora",
        )
        assert model.adapter_path == "/path/to/lora"

    def test_adapter_path_property_none(self):
        model = DiffusersModelHandler(model_path=self.model_path, model_type="sd15")
        assert model.adapter_path is None
