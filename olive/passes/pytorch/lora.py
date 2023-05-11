# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from typing import Any, Callable, Dict, Union

from olive.common.user_module_loader import UserModuleLoader
from olive.model import ModelStorageKind, PyTorchModel
from olive.passes import Pass
from olive.passes.olive_pass import PassConfigParam
from olive.constants import ModelFileFormat


class HFLoRA(Pass):
    """Finetune PyTorch model using LoRA"""

    _requires_user_script = True

    @staticmethod
    def _default_config() -> Dict[str, PassConfigParam]:
        return {
            "training_loop_func": PassConfigParam(
                type_=Union[Callable, str], is_object=True, description="Customized training loop function."
            ),
            "lora_alpha" : PassConfigParam(type_=int, default_value=16, description="LoRA Alpha, scaling factor for the weight matrics"),
            "r" : PassConfigParam(type_=int, default_value=16, description="Dimesion of the low-rank matrics"),
            "lora_dropout" : PassConfigParam(type_=float, default_value=0.1, description="Dropout probability of the LoRA layers"),
            "bias" : PassConfigParam(type_=str, default_value="all", description="If set to all then train all bias parameters"),
            "task_type" : PassConfigParam(type_=str, default_value="TaskType.CAUSAL_LM", description="Finetuing task type. For example TOKEN_CLS, CAUSAL_LM"),
        }

    def _run_for_config(self, input_model: PyTorchModel, config: Dict[str, Any], output_model_path: str) -> PyTorchModel:
        from peft import get_peft_model, LoraConfig

        input_pt_model = input_model.load_model()

        # Apply LORA
        peft_config = LoraConfig(
            task_type=config["task_type"], inference_mode=False, r=config["r"], lora_alpha=config["lora_alpha"], lora_dropout=config["lora_dropout"], 
            bias=config["bias"]
        )
        model = get_peft_model(input_pt_model, peft_config)
        model.print_trainable_parameters()

        # Train
        if config["training_loop_func"] is None:
            raise ValueError("Please provide training_loop_func")

        loader = UserModuleLoader(user_script=config["user_script"], script_dir=config["script_dir"])
        trained_lora_model = loader.call_object(config["training_loop_func"], model)

        trained_lora_model.save_pretrained(output_model_path)
        return PyTorchModel(
            model_path=output_model_path,
            model_file_format=ModelFileFormat.PYTORCH_LORA_MODEL,
            model_storage_kind=ModelStorageKind.LocalFile,
            model_script= input_model.model_script,
            script_dir= input_model.script_dir,
            io_config= input_model.io_config,
            dummy_inputs_func= input_model.dummy_inputs_func,
            hf_config=input_model.hf_config,
        )
