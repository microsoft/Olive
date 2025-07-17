from olive.passes import Pass
from olive.model import ONNXModelHandler
from olive.passes.pass_config import BasePassConfig, PassConfigParam
from model_generate import generate_npu_model
from pathlib import Path
from olive.hardware.accelerator import AcceleratorSpec
from olive.model.utils import resolve_onnx_path
from olive.passes.onnx.common import model_proto_to_olive_model
import onnx


class VitisGenerateModel(Pass):
    @classmethod
    def _default_config(cls, accelerator_spec):
        return {
            "packed_const": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Enable packed constants optimization in NPU export."
            )
        }

    @staticmethod
    def is_accelerator_agnostic(accelerator_spec: AcceleratorSpec) -> bool:
        return False

    def _run_for_config(
        self, model: ONNXModelHandler, config: BasePassConfig, output_model_path: str
    ) -> ONNXModelHandler:
        print(f"[DEBUG] Running VitisGenerateModel with config: {config}")

        # assert isinstance(model, ONNXModelHandler), "VitisGenerateModel only supports ONNXModelHandler input."

        input_model_path = model.model_path
        output_dir = Path(output_model_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[VitisGenerateModel] Generating Vitis NPU model from: {input_model_path}")
        print(f"[VitisGenerateModel] Output directory: {output_dir}")
        print(f"[VitisGenerateModel] Packed constants: {config.packed_const}")

        # Generate the NPU model
        generate_npu_model(
            input_model=str(input_model_path),
            output_dir=str(output_dir),
            packed_const=config.packed_const
        )

        # Load final ONNX model to wrap into Olive model
        final_model_path = resolve_onnx_path(str(output_dir), "model.onnx")
        onnx_model = onnx.load(final_model_path)
        print(f"[DEBUG] Model generated at: {final_model_path}")

        return model_proto_to_olive_model(onnx_model, final_model_path, config)
