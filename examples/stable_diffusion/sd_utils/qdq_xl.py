import inspect
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional, Union

import numpy as np
import onnxruntime as ort
import torch
from diffusers import StableDiffusionXLPipeline
from optimum.onnxruntime.modeling_diffusion import (
    ORTDiffusionPipeline,
    ORTModelTextEncoder,
    ORTModelUnet,
    ORTModelVaeDecoder,
    ORTModelVaeEncoder,
    ORTWrapperVae,
)
from transformers.modeling_outputs import ModelOutput


class ORTDiffusionPipelineWithSave(ORTDiffusionPipeline):
    def __init__(
        self,
        scheduler: "SchedulerMixin",
        unet_session: ort.InferenceSession,
        vae_decoder_session: ort.InferenceSession,
        # optional pipeline models
        vae_encoder_session: Optional[ort.InferenceSession] = None,
        text_encoder_session: Optional[ort.InferenceSession] = None,
        text_encoder_2_session: Optional[ort.InferenceSession] = None,
        # optional pipeline submodels
        tokenizer: Optional["CLIPTokenizer"] = None,
        tokenizer_2: Optional["CLIPTokenizer"] = None,
        feature_extractor: Optional["CLIPFeatureExtractor"] = None,
        # stable diffusion xl specific arguments
        force_zeros_for_empty_prompt: bool = True,
        requires_aesthetics_score: bool = False,
        add_watermarker: Optional[bool] = None,
        # onnxruntime specific arguments
        use_io_binding: Optional[bool] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        **kwargs,
    ):
        self.unet = ORTModelUnetWithSave(unet_session, self)
        self.vae_decoder = ORTModelVaeDecoderWithSave(vae_decoder_session, self)
        self.vae_encoder = ORTModelVaeEncoder(vae_encoder_session, self) if vae_encoder_session is not None else None
        self.text_encoder = (
            ORTModelTextEncoderWithSave(text_encoder_session, self) if text_encoder_session is not None else None
        )
        self.text_encoder_2 = (
            ORTModelTextEncoderWithSave(text_encoder_2_session, self) if text_encoder_2_session is not None else None
        )
        # We wrap the VAE Decoder & Encoder in a single object to simulate diffusers API
        self.vae = ORTWrapperVae(self.vae_encoder, self.vae_decoder)

        # we allow passing these as torch models for now
        self.image_encoder = kwargs.pop("image_encoder", None)  # TODO: maybe implement ORTModelImageEncoder
        self.safety_checker = kwargs.pop("safety_checker", None)  # TODO: maybe implement ORTModelSafetyChecker

        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.feature_extractor = feature_extractor

        all_pipeline_init_args = {
            "vae": self.vae,
            "unet": self.unet,
            "text_encoder": self.text_encoder,
            "text_encoder_2": self.text_encoder_2,
            "safety_checker": self.safety_checker,
            "image_encoder": self.image_encoder,
            "scheduler": self.scheduler,
            "tokenizer": self.tokenizer,
            "tokenizer_2": self.tokenizer_2,
            "feature_extractor": self.feature_extractor,
            "requires_aesthetics_score": requires_aesthetics_score,
            "force_zeros_for_empty_prompt": force_zeros_for_empty_prompt,
            "add_watermarker": add_watermarker,
        }

        diffusers_pipeline_args = {}
        for key in inspect.signature(self.auto_model_class).parameters.keys():
            if key in all_pipeline_init_args:
                diffusers_pipeline_args[key] = all_pipeline_init_args[key]
        # inits diffusers pipeline specific attributes (registers modules and config)
        self.auto_model_class.__init__(self, **diffusers_pipeline_args)

        # inits ort specific attributes
        self.shared_attributes_init(
            model=unet_session, use_io_binding=use_io_binding, model_save_dir=model_save_dir, **kwargs
        )

    @property
    def save_data_dir(self):
        return self.text_encoder.save_data_dir

    @save_data_dir.setter
    def save_data_dir(self, dir: Path):
        self.text_encoder.save_data_dir = dir
        self.text_encoder.save_data_index = 10
        self.text_encoder_2.save_data_dir = dir
        self.text_encoder_2.save_data_index = 20
        self.unet.save_data_dir = dir
        self.unet.save_data_index = 0
        self.vae_decoder.save_data_dir = dir


class ORTStableDiffusionXLPipelineWithSave(ORTDiffusionPipelineWithSave, StableDiffusionXLPipeline):
    """ONNX Runtime-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionXLPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLPipeline)."""

    main_input_name = "prompt"
    export_feature = "text-to-image"
    auto_model_class = StableDiffusionXLPipeline

    def _get_add_time_ids(
        self,
        original_size,
        crops_coords_top_left,
        target_size,
        dtype,
        text_encoder_projection_dim=None,
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids


# Wrappers


class ORTModelTextEncoderWithSave(ORTModelTextEncoder):
    def forward(
        self,
        input_ids: Union[np.ndarray, torch.Tensor],
        attention_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: bool = False,
    ):
        use_torch = isinstance(input_ids, torch.Tensor)

        model_inputs = {"input_ids": input_ids}

        onnx_inputs = self.prepare_onnx_inputs(use_torch, **model_inputs)
        if self.save_data_dir:
            np.savez(self.save_data_dir / f"text_encoder_{self.save_data_index}.npz", **onnx_inputs)
            self.save_data_index += 1
        onnx_outputs = self.session.run(None, onnx_inputs)
        model_outputs = self.prepare_onnx_outputs(use_torch, *onnx_outputs)

        if output_hidden_states:
            model_outputs["hidden_states"] = []
            for i in range(self.config.num_hidden_layers):
                model_outputs["hidden_states"].append(model_outputs.pop(f"hidden_states.{i}"))
            model_outputs["hidden_states"].append(model_outputs.get("last_hidden_state"))
        else:
            for i in range(self.config.num_hidden_layers):
                model_outputs.pop(f"hidden_states.{i}", None)

        if return_dict:
            return model_outputs

        return ModelOutput(**model_outputs)


class ORTModelUnetWithSave(ORTModelUnet):
    def forward(
        self,
        sample: Union[np.ndarray, torch.Tensor],
        timestep: Union[np.ndarray, torch.Tensor],
        encoder_hidden_states: Union[np.ndarray, torch.Tensor],
        text_embeds: Optional[Union[np.ndarray, torch.Tensor]] = None,
        time_ids: Optional[Union[np.ndarray, torch.Tensor]] = None,
        timestep_cond: Optional[Union[np.ndarray, torch.Tensor]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = False,
    ):
        use_torch = isinstance(sample, torch.Tensor)

        if len(timestep.shape) == 0:
            timestep = timestep.unsqueeze(0)

        model_inputs = {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "text_embeds": text_embeds,
            "time_ids": time_ids,
            "timestep_cond": timestep_cond,
            **(cross_attention_kwargs or {}),
            **(added_cond_kwargs or {}),
        }

        onnx_inputs = self.prepare_onnx_inputs(use_torch, **model_inputs)
        if self.save_data_dir:
            np.savez(self.save_data_dir / f"unet_{self.save_data_index}.npz", **onnx_inputs)
            self.save_data_index += 1
        onnx_outputs = self.session.run(None, onnx_inputs)
        model_outputs = self.prepare_onnx_outputs(use_torch, *onnx_outputs)

        if return_dict:
            return model_outputs

        return ModelOutput(**model_outputs)


class ORTModelVaeDecoderWithSave(ORTModelVaeDecoder):
    def forward(
        self,
        latent_sample: Union[np.ndarray, torch.Tensor],
        generator: Optional[torch.Generator] = None,
        return_dict: bool = False,
    ):
        use_torch = isinstance(latent_sample, torch.Tensor)

        model_inputs = {"latent_sample": latent_sample}

        onnx_inputs = self.prepare_onnx_inputs(use_torch, **model_inputs)
        if self.save_data_dir:
            np.savez(self.save_data_dir / "vae_decoder.npz", **onnx_inputs)
        onnx_outputs = self.session.run(None, onnx_inputs)
        if self.save_data_dir:
            np.savez(self.save_data_dir / "vae_decoder_output.npz", sample=onnx_outputs[0])
        model_outputs = self.prepare_onnx_outputs(use_torch, *onnx_outputs)

        if "latent_sample" in model_outputs:
            model_outputs["latents"] = model_outputs.pop("latent_sample")

        if return_dict:
            return model_outputs

        return ModelOutput(**model_outputs)
