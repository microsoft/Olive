# -------------------------------------------------------------------------
# Copyright (c) Intel Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from typing import Dict,Type, Union
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import OpenVINOModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam, get_user_script_data_config
import shutil

class OpenVINOReshape(Pass):
    """Converts PyTorch, ONNX or TensorFlow Model to OpenVino Model."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            **get_user_script_data_config(),
            "extra_configs": PassConfigParam(
                type_=Dict,
                default_value=None,
                required=False,
                description=(
                    "Extra configurations for OpenVINO model conversion. extra_config can be set by "
                    "passing a dictionary where key is the parameter name, and the value is the parameter value. "
                    "Please check Conversion Parameters documentation for more details: "
                    "https://docs.openvino.ai/2023.3/openvino_docs_OV_Converter_UG_Conversion_Options.html"
                ),
            ),
            "input_model": PassConfigParam(
                type_=str,
                default_value=None,
                required=False,
                description="Name of the input OpenVINO model.",
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
            "shared_cache": PassConfigParam(
                type_=bool,
                default_value=False,
                required=False,
                description=(
                    "Uses the same cache folder of the previous pass."
                    "Can only be used with dynamic models."
                    "Ignored if model I/O names are found to be missing."
                    "Enahances file reusablility between each pass."
                ),
            ),
            "static": PassConfigParam(
                type_=bool,
                default_value=False,
                required=False,
                description=(
                    "Create a static model instead of a dynamic model."
                    "Enabled by default."
                ),
            )
        }

    def _run_for_config(
        self,
        model: Union[OpenVINOModelHandler],
        config: Type[BasePassConfig],
        output_model_path: str,
    ) -> OpenVINOModelHandler:
        try:
            import openvino as ov
        except ImportError:
            raise ImportError("Please install olive-ai[openvino] to use OpenVINO model") from None

        if config.input_model:
            model_name = config.input_model
        else:
            model_name = model.model_config["model_name"]
        
        input_dir = Path(model.model_path) / (model_name)
        if config.shared_cache and not config.static:
            output_model_path = model.model_path
        elif config.shared_cache and config.static:
            msg = str("Error! Cannot use shared cache with creating static models")
            raise ImportError(msg) from None
        
        core = ov.Core()
        model_name_path = Path(model.model_path) / (model_name+".xml")
        
        loaded_model = core.read_model(model_name_path, weights=Path(model.model_path) / (model_name+".bin"))
        
        # Ensure atleast 1 input name is present for all inputs
        update_io_names = False
        for i,v in enumerate(loaded_model.inputs):
            if not loaded_model.input(i).get_names():
                loaded_model.input(i).set_names({f"input_{i}"})
                update_io_names = True
        
        # Ensure atleast 1 output name is present for all inputs
        for i,v in enumerate(loaded_model.outputs):
            if not loaded_model.output(i).get_names():
                loaded_model.output(i).set_names({f"output_{i}"})
                update_io_names = True

        if config.static:
            if config.input_shapes and len(config.input_shapes) == len(loaded_model.inputs):
                inputs = {}
                for i,dim in enumerate(config.input_shapes):
                    inputs[loaded_model.input(i)] = ov.PartialShape(dim)

                loaded_model.reshape(inputs)
                
                model_name_path = Path(output_model_path) / (model_name+"_st"+".xml")
                ov.save_model(loaded_model,model_name_path)

            elif not config.input_shapes:
                msg = str("Error! Missing input shapes")
                raise ImportError(msg) from None

            else:
                msg = str("Error! The number of inputs in model " + str(len(loaded_model.inputs)) +
                    " do not match the number of entries in the input_shapes in config, " + str(len(config.input_shapes)))
                raise ImportError(msg) from None
        
        else:
            if update_io_names:
                ov.save_model(loaded_model,Path(output_model_path) / (model_name+".xml"))
            
            if not config.shared_cache and not update_io_names:
                xml_file = input_dir.with_suffix(".xml")
                bin_file = input_dir.with_suffix(".bin")
                shutil.copy2(xml_file, output_model_path)
                shutil.copy2(bin_file, output_model_path)
        
        return OpenVINOModelHandler(model_path=output_model_path)
