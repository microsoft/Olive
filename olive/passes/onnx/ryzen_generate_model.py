from olive.passes import Pass
from olive.model import ONNXModelHandler, HfModelHandler
from olive.passes.pass_config import BasePassConfig, PassConfigParam
from model_generate import generate_npu_model
from pathlib import Path
from olive.hardware.accelerator import AcceleratorSpec
from olive.model.utils import resolve_onnx_path
from olive.passes.onnx.common import model_proto_to_olive_model
import onnx


class RyzenGenerateModel(Pass):
    @classmethod
    def _default_config(cls, accelerator_spec):
        return {
            "packed_const": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Enable packed constants optimization in NPU export."
            ), 
            "cpu_only": PassConfigParam(
            type_=bool,
            default_value=False,
            description="Run only model builder -OGA CPU only model, skip NPU-related steps."
            )
        }

    @staticmethod
    def is_accelerator_agnostic(accelerator_spec: AcceleratorSpec) -> bool:
        return False

    def _run_for_config(
        self, model: HfModelHandler, config: BasePassConfig, output_model_path: str
    ) -> ONNXModelHandler:
        print(f"[DEBUG] Running RyzenGenerateModel with config: {config}")

        # assert isinstance(model, ONNXModelHandler), "RyzenGenerateModel only supports ONNXModelHandler input." # uncomment once quark and model bu

        input_model_path = model.model_path
        output_dir = Path(output_model_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[RyzenGenerateModel] Generating Ryzen NPU model from: {input_model_path}")
        print(f"[RyzenGenerateModel] Output directory: {output_dir}")
        print(f"[RyzenGenerateModel] Packed constants: {config.packed_const}")

        # Generate the NPU model
        generate_npu_model(
            input_model=str(input_model_path),
            output_dir=str(output_dir),
            packed_const=config.packed_const, 
            cpu_only=config.cpu_only
        )

        # Load final ONNX model to wrap into Olive model
        final_model_path = resolve_onnx_path(str(output_dir), "model.onnx")
        onnx_model = onnx.load(final_model_path)
        print(f"[DEBUG] Model generated at: {final_model_path}")

        return model_proto_to_olive_model(onnx_model, final_model_path, config)
