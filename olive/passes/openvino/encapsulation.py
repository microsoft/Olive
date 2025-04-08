# -------------------------------------------------------------------------
# Copyright (c) Intel Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from typing import Dict, Type, Union
from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.model import ONNXModelHandler, OpenVINOModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam

from openvino import get_version
import onnx.helper as helper
from onnx import TensorProto, save
import shutil

class OpenVINOEncapsulation(Pass):
    openvino_to_onnx_dtype = {
        'f32': TensorProto.FLOAT,
        'float32': TensorProto.FLOAT,
        'f64': TensorProto.DOUBLE,
        'float64': TensorProto.DOUBLE,
        'f16': TensorProto.FLOAT16,
        'bf16': TensorProto.BFLOAT16,
        'i8': TensorProto.INT8,
        'int8_t': TensorProto.INT8,
        'i16': TensorProto.INT16,
        'int16_t': TensorProto.INT16,
        'i32': TensorProto.INT32,
        'int32_t': TensorProto.INT32,
        'i64': TensorProto.INT64,
        'int64_t': TensorProto.INT64,
        'u8': TensorProto.UINT8,
        'uint8_t': TensorProto.UINT8,
        'u16': TensorProto.UINT16,
        'uint16_t': TensorProto.UINT16,
        'u32': TensorProto.UINT32,
        'uint32_t': TensorProto.UINT32,
        'u64': TensorProto.UINT64,
        'uint64_t': TensorProto.UINT64,
        'bool': TensorProto.BOOL,
        'boolean': TensorProto.BOOL,
        # Add more if needed
    }
    # Add any required data members to the class
    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            "target_device": PassConfigParam(
                type_=Device,
                default_value=accelerator_spec.accelerator_type.CPU,
                required=False,
                description=(
                    "Device the encapsulated model should run on."
                    "Available devices are cpu, gpu, npu."
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
            "input_model": PassConfigParam(
                type_=str,
                default_value=None,
                required=False,
                description="Name of the input OpenVINO model.",
            ),
            "ov_version": PassConfigParam(
                type_=str,
                default_value=None,
                required=False,
                description=(
                    "Name of the OpenVINO version to override in model SDK version."
                    "Requires a minimum version of OpenVino 2025.1"
                ),
            ),
            "opset_imports": PassConfigParam(
                type_=list,
                default_value=[
                    ["com.microsoft.nchwc", 1],
                    ["",11],
                    ["ai.onnx.ml", 5],
                    ["com.ms.internal.nhwc", 11],
                    ["ai.onnx.training", 1],
                    ["ai.onnx.preview.training", 1],
                    ["com.microsoft.experimental", 1],
                    ["com.microsoft", 1],
                    ["org.pytorch.aten", 1],
                ],
                required=False,
                description="Opset name and version to be add in the generate context model",
            ),
        }

    def _run_for_config(
        self,
        model: Union[OpenVINOModelHandler],
        config: Type[BasePassConfig],
        output_model_path: str,
    ) -> ONNXModelHandler:
        try:
            import openvino as ov
        except ImportError:
            raise ImportError("Please install olive-ai[openvino] to use OpenVINO model") from None
        
        if config.shared_cache:
            output_model_path = model.model_path
        
        if config.input_model:
            model_name = config.input_model
        else:
            model_name = model.model_config["model_name"]

        if config.ov_version:
            ov_version = config.ov_version
        else:
            ov_version = get_version()

        input_dir = Path(model.model_path) / (model_name)
        core = ov.Core()

        loaded_model = core.read_model(model=input_dir.with_suffix(".xml"))

        context_name = model_name + ".xml"
        

        # Get/Fix input names & ov shapes.
        input_info = {}
        for i,input in enumerate(loaded_model.inputs):
            name = "input_" + str(i)
            if input:
                name = input.get_any_name()
            input_info[name] = (input.get_partial_shape(), input.get_element_type())

        # Get/Fix input names & ov shapes.
        output_info = {}
        for i,out in enumerate(loaded_model.outputs):
            name = "output_" + str(i)
            if out:
                name = out.get_any_name()
            output_info[name] = (out.get_partial_shape(), out.get_element_type())

        # Transform to onnx input shapes
        inputs = []
        outputs = []
        for i, (name, (shape, datatype)) in enumerate(input_info.items()):
            shape_list = []
            for j, dim in enumerate(shape):
                if not dim.is_dynamic:
                    shape_list.append(dim.get_length())
                else:
                    shape_list.append("input_" + f"{i}_"+f"{j}")

            # Normalize the datatype string & map to ONNX data type
            normalized_dtype = str(datatype).split("'")[1]  # Extract 'int64_t' from "<Type: 'int64_t'>"
            onnx_dtype = self.openvino_to_onnx_dtype.get(normalized_dtype)

            inputs.append(helper.make_tensor_value_info(name, onnx_dtype, shape_list))

        # Transform to onnx output shapes
        for i, (name, (shape, datatype)) in enumerate(output_info.items()):
            shape_list = []
            for j, dim in enumerate(shape):
                if not dim.is_dynamic:
                    shape_list.append(dim.get_length())
                else:
                    shape_list.append("output_" + f"{i}_"+f"{j}")

            # Normalize the datatype string & map to ONNX data type and extract 'int64_t' from "<Type: 'int64_t'>"
            normalized_dtype = str(datatype).split("'")[1]
            onnx_dtype = self.openvino_to_onnx_dtype.get(normalized_dtype)

            outputs.append(helper.make_tensor_value_info(name, onnx_dtype, shape_list))
        
        # Create context node (simulates a custom EP context schema operation)
        context_node = helper.make_node(
            "EPContext",  
            inputs = [name for name, _ in input_info.items()],
            outputs = [name for name, _ in output_info.items()],
            name="ContextNode",
            domain="com.microsoft"
        )

        # Properties of the context node, currently only support context node that points to the payload content
        context_node.attribute.extend([helper.make_attribute("embed_mode", 0)])
        context_node.attribute.extend([helper.make_attribute("ep_cache_context", context_name)])
        context_node.attribute.extend([helper.make_attribute("ep_sdk_version", ov_version)])
        context_node.attribute.extend([helper.make_attribute("main_context", 1)])
        context_node.attribute.extend([helper.make_attribute("max_size", 0)])
        context_node.attribute.extend([helper.make_attribute("source", "OpenVINOExecutionProvider")])
        context_node.attribute.extend([helper.make_attribute("DeviceClass", config.target_device.upper())])

        # Create the ONNX Graph
        graph_def = helper.make_graph(
            nodes=[context_node],
            name="EP_Context_Model",
            inputs=inputs,
            outputs=outputs
        )
        op_imports = [helper.make_opsetid(i[0],i[1]) for i in config.opset_imports]

        # Define the model with an Execution Provider (EP) Context
        model_def = helper.make_model(
            graph_def,
            opset_imports= op_imports
        )

        # Save the model
        context_model_output = model_name + ".onnx"
        context_model_output_dir = Path(output_model_path) / (context_model_output)
        save(model_def, context_model_output_dir)
        
        xml_file = input_dir.with_suffix(".xml")
        bin_file = input_dir.with_suffix(".bin")
        if not config.shared_cache :
            shutil.copy2(xml_file, output_model_path)
            shutil.copy2(bin_file, output_model_path)
        
        return ONNXModelHandler(model_path=output_model_path)