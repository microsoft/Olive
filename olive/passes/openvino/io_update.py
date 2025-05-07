# -------------------------------------------------------------------------
# Copyright (c) Intel Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
from pathlib import Path
from typing import Union

from olive.common.utils import hardlink_copy_dir, hardlink_copy_file
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import OpenVINOModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam, get_user_script_data_config


class OpenVINOIoUpdate(Pass):
    """Converts dynamic OpenVINO Model to static OpenVino Model and updates IO names."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            **get_user_script_data_config(),
            "extra_configs": PassConfigParam(
                type_=dict,
                default_value=None,
                required=False,
                description=(
                    "Extra configurations for OpenVINO model conversion. extra_config can be set by "
                    "passing a dictionary where key is the parameter name, and the value is the parameter value. "
                    "Please check Conversion Parameters documentation for more details: "
                    "https://docs.openvino.ai/2025/openvino-workflow/model-preparation/conversion-parameters.html"
                ),
            ),
            "input_shapes": PassConfigParam(
                type_=list,
                default_value=None,
                required=False,
                description=(
                    "Reshapes the model with given inputs. "
                    "It configures dynamic and static dimensions in model inputs "
                    "depending on your inference requirements. "
                    "Static parameter is required to be enabled if static dimensions are required. "
                ),
            ),
            "input_names": PassConfigParam(
                type_=list,
                default_value=None,
                required=False,
                description=("Reshapes the model input names with given inputs. "),
            ),
            "static": PassConfigParam(
                type_=bool,
                default_value=False,
                required=False,
                description=("Create a static model instead of a dynamic model.Enabled by default."),
            ),
            "reuse_cache": PassConfigParam(
                type_=bool,
                default_value=False,
                required=False,
                description=("Reuse cache of previous passes to reduce storage footprint."),
            ),
        }

    def _run_for_config(
        self,
        model: Union[OpenVINOModelHandler],
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> OpenVINOModelHandler:
        if config.reuse_cache:
            output_model_path = model.model_path

        model_name = model.model_config["model_name"]
        model_name_path = Path(model.model_path) / (f"{model_name}.xml")
        weight_name_path = Path(model.model_path) / (f"{model_name}.bin")

        update_io_names = self._run_pass(model, config, output_model_path)

        if config.reuse_cache and (config.static or update_io_names):
            if os.path.exists(model_name_path):
                os.remove(model_name_path)
            if os.path.exists(weight_name_path):
                os.remove(weight_name_path)
        elif config.reuse_cache and not update_io_names:
            output_model_name = f"{model_name}_dy"
            Path.rename(model_name_path, Path(output_model_path) / (f"{output_model_name}.xml"))
            Path.rename(weight_name_path, Path(output_model_path) / (f"{output_model_name}.bin"))

        return OpenVINOModelHandler(model_path=output_model_path)

    def _run_pass(
        self,
        model: Union[OpenVINOModelHandler],
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> bool:
        try:
            import openvino as ov

        except ImportError:
            raise ImportError("Please install olive-ai[openvino] to use OpenVINO model") from None

        model_name = model.model_config["model_name"]

        core = ov.Core()
        model_name_path = Path(model.model_path) / (f"{model_name}.xml")
        weight_name_path = Path(model.model_path) / (f"{model_name}.bin")

        loaded_model = core.read_model(model_name_path, weights=weight_name_path)

        # Ensure at least 1 input name is present for all inputs
        update_io_names = False
        for i, _ in enumerate(loaded_model.inputs):
            if config.input_names:
                if len(config.input_names) == len(loaded_model.inputs):
                    loaded_model.input(i).set_names({config.input_names[i]})
                    update_io_names = True
                else:
                    raise ValueError(
                        f"Number of model inputs {len(loaded_model.inputs)} does not match number of given inputs {len(config.input_names)}"
                    ) from None
            if not loaded_model.input(i).get_names():
                loaded_model.input(i).set_names({f"input_{i}"})
                update_io_names = True

        # Ensure at least 1 output name is present for all outputs
        for i, _ in enumerate(loaded_model.outputs):
            if not loaded_model.output(i).get_names():
                loaded_model.output(i).set_names({f"output_{i}"})
                update_io_names = True

        output_model_name = f"{model_name}_st" if config.static else f"{model_name}_dy"
        if config.static:
            if config.input_shapes and len(config.input_shapes) == len(loaded_model.inputs):
                inputs = {}
                for i, dim in enumerate(config.input_shapes):
                    inputs[loaded_model.input(i)] = ov.PartialShape(dim)

                loaded_model.reshape(inputs)
                model_name_path_dst = Path(output_model_path) / (f"{output_model_name}.xml")
                ov.save_model(loaded_model, model_name_path_dst)

            elif not config.input_shapes:
                msg = "Error! Missing input shapes"
                raise ValueError(msg) from None

            else:
                msg = (
                    f"Error! The number of inputs in model {len(loaded_model.inputs)}"
                    f"do not match the number of entries in the input_shapes in config, {len(config.input_shapes)}"
                )
                raise ValueError(msg) from None

        # dynamic model will be saved to new location only if io names change
        elif update_io_names:
            ov.save_model(loaded_model, Path(output_model_path) / (f"{output_model_name}.xml"))

        # dynamic model with no io name change and hence it can just be copied to new location if reuse_cache is false
        elif not config.reuse_cache:
            model_name_path_dst = Path(output_model_path) / (f"{output_model_name}.xml")
            weight_name_path_dst = Path(output_model_path) / (f"{output_model_name}.bin")
            hardlink_copy_file(model_name_path, model_name_path_dst, follow_symlinks=True)
            hardlink_copy_file(weight_name_path, weight_name_path_dst, follow_symlinks=True)

        if not config.reuse_cache:
            # copy JSON and text files for genai models
            all_genai_files = [name for name in Path(model.model_path).iterdir() if name.suffix in [".json", ".txt"]]
            for genai_file in all_genai_files:
                src_pth = Path(model.model_path) / genai_file
                dest_path = Path(output_model_path)
                hardlink_copy_file(src_pth, dest_path, follow_symlinks=True)

            # copy tokenizer folder if it exists
            src_tokenizer = Path(model.model_path) / "openvino_tokenizer"
            if src_tokenizer.exists() and src_tokenizer.is_dir():
                dest_tokenizer = Path(output_model_path) / "openvino_tokenizer"
                hardlink_copy_dir(src_tokenizer, dest_tokenizer, symlinks=True)

            # copy detokenizer folder if it exists
            src_detokenizer = Path(model.model_path) / "openvino_detokenizer"
            if src_detokenizer.exists() and src_detokenizer.is_dir():
                dest_detokenizer = Path(output_model_path) / "openvino_detokenizer"
                hardlink_copy_dir(src_detokenizer, dest_detokenizer, symlinks=True)

        return update_io_names
