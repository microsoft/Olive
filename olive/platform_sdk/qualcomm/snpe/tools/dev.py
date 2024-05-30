# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import csv
import logging
import platform
import tempfile
from pathlib import Path
from typing import List

import onnx
from onnx import TensorProto, helper

from olive.common.constants import OS
from olive.platform_sdk.qualcomm.runner import SNPESDKRunner as SNPERunner

logger = logging.getLogger(__name__)


def get_snpe_version() -> str:
    """Get the version of the SNPE SDK at SNPE_ROOT."""
    cmd = "snpe-net-run --version"
    stdout, _ = SNPERunner().run(cmd)
    return stdout.split("SNPE v")[1].strip()


def get_dlc_info(dlc_path: str, csv_path: str = None) -> str:
    """Get the info of a DLC file."""
    cmd = f"snpe-dlc-info -i {dlc_path}"
    if csv_path:
        cmd += f" -s {csv_path}"
    stdout, _ = SNPERunner(use_dev_tools=True).run(cmd)

    prefix = "DLC info for:"
    return prefix + stdout.split(prefix)[1]


def get_dlc_io_config(dlc_path: str, input_names: List[str], output_names: List[str]) -> str:
    """Get the input/output config of a DLC file.

    dlc_path: path to the DLC file.
    input_names: list of input names of source model.
    output_names: list of output names of source model.
    """
    with tempfile.TemporaryDirectory() as tmp_folder:
        tmp_csv = Path(tmp_folder) / "dlc_info.csv"
        dlc_info = get_dlc_info(dlc_path, csv_path=str(tmp_csv))

        # add the :0 suffix to the input/output names if present in the DLC
        # SNPE adds this suffix to the input/output names in the DLC if the source model is TensorFlow
        if f"{input_names[0]}:0" in dlc_info:
            input_names = [f"{name}:0" for name in input_names]
            output_names = [f"{name}:0" for name in output_names]

        input_dims = {}
        output_dims = {}
        with tmp_csv.open() as f:
            out = csv.reader(f)
            for row in out:
                if len(row) == 8:
                    _, name, _, input_item, output, shape, _, _ = tuple(row)
                    # version 2.x has 'name type'
                    output = output.split(" ")[0]
                    # input name in this format for versions 1.x
                    if name == input_item and name == output and name and name in input_names:
                        input_dims[name] = list(map(int, shape.split("x")))
                    elif output in output_names:
                        output_dims[output] = list(map(int, shape.split("x")))
                if len(row) == 3:
                    name, shape, _ = tuple(row)
                    # input name in this format for versions 2.x
                    if name in input_names:
                        input_dims[name] = list(map(int, shape.split(",")))

    return {
        "input_names": input_names,
        "input_shapes": [input_dims[name] for name in input_names],
        "output_names": output_names,
        "output_shapes": [output_dims[name] for name in output_names],
    }


def get_dlc_metrics(dlc_path: str) -> str:
    """Get the metrics of a DLC file."""
    dlc_info = get_dlc_info(dlc_path)

    # number of parameters
    parameters = int(dlc_info.split("Total parameters:")[1].split()[0])

    # number of MACs
    macs = dlc_info.split("Total MACs per inference:")[1].split()[0]
    if "k" in macs:
        macs = int(macs.split("k")[0]) * 1000
    elif "M" in macs:
        macs = int(macs.split("M")[0]) * 1000000
    else:
        macs = int(macs)

    # memory
    memory_prefix = "Est. Steady-State Memory Needed to Run:"
    memory = float(dlc_info.split(memory_prefix)[1].split()[0])
    memory_unit = dlc_info.split(memory_prefix)[1].split()[1]

    # SNPE version used to create the DLC
    version = dlc_info.split("DLC created with converter version:")[1].split()[0]

    return {"parameters": parameters, "macs": macs, f"memory ({memory_unit})": memory, "snpe-version": version}


def get_dlc_snpe_version(dlc_path: str) -> str:
    """Get the SNPE version used to create a DLC file."""
    return get_dlc_metrics(dlc_path)["snpe-version"]


def _get_conversion_arg_str(arg_type: str, input_names: List[str], input_values: List[str]):
    """Get conversion argument string for snpe dlc converter tools.

    arg_type: "-d", "-t", or "-l" for input shapes, types, and layouts respectively.
    input_names: list of input names.
    input_values: list of input values. Values can be None. If so, the corresponding input will be skipped
    """
    valid_arg_types = ["-d", "-t", "-l"]
    if arg_type not in valid_arg_types:
        raise ValueError(f"Invalid arg_type: {arg_type}. Valid args are: {valid_arg_types}")

    arg_str = ""
    if input_values is None:
        return arg_str

    for name, value in zip(input_names, input_values):
        if value is None:
            continue  # skip this input
        if arg_type == "-d":
            # convert shape list to string
            # e.g. [1, 3, 224, 224] -> "1,3,224,224"
            value_str = ",".join([str(v) for v in value])
        else:
            value_str = value
        arg_str += f" {arg_type} {name} {value_str}"

    return arg_str.lstrip()


def to_dlc(model_file: str, model_framework: str, config: dict, output_file: str):
    """Convert a model into a SNPE DLC.

    model_file: path to the model file.
    model_framework: "onnx" or "tensorflow".
    config: a config dict with the following keys:
        input_names: List[str] = list of input names.
        input_shapes: List[str] = list of input shapes.
        input_types: Union[None, List[Union[str, None]]] = list of input types.
            None means we don't specify the input type.
        input_layouts: Union[None, List[Union[str, None]]] = list of input layouts.
            None means we don't specify the input layout.
        output_names: List[str] = list of output names.
    output_file: path to the output DLC file.
    """
    if model_framework.lower() == "onnx":
        snpe_dlc_converter_tool_name = "snpe-onnx-to-dlc"
    elif model_framework.lower() == "tensorflow":
        snpe_dlc_converter_tool_name = "snpe-tensorflow-to-dlc"
    else:
        raise NotImplementedError("Only ONNX and TensorFlow models are supported")

    d_str = _get_conversion_arg_str("-d", config["input_names"], config["input_shapes"])
    t_str = _get_conversion_arg_str("-t", config["input_names"], config["input_types"])
    l_str = _get_conversion_arg_str("-l", config["input_names"], config["input_layouts"])
    out_node_str = " ".join(["--out_node " + x for x in config["output_names"]])

    cmd = f"{snpe_dlc_converter_tool_name} -i {model_file} -o {output_file} {d_str} {out_node_str} {t_str} {l_str}"
    if config["extra_args"] is not None:
        cmd += " " + config["extra_args"]

    _, stderr = SNPERunner(use_dev_tools=True).run(cmd)

    # check if conversion succeeded
    if "Conversion completed successfully" not in stderr:
        raise RuntimeError(stderr)


def quantize_dlc(dlc_path: str, input_list: str, config: dict, output_file: str):
    """Quantize a SNPE DLC.

    dlc_path: path to the DLC file.
    input_list: path to the input list file for trial inputs.
    config: a config dict with the following keys:
        use_enhanced_quantizer: bool = whether to use the enhanced quantizer.
        enable_htp: bool = whether to enable HTP.
        htp_socs: List[str] = list of HTP SoCs.
        extra_args: str = extra arguments to pass to the quantizer.
    """
    quant_cmd = "snpe-dlc-quantize"
    if platform.system() == OS.WINDOWS:
        # snpe-dlc-quant is the Windows version of the quantizer tool
        # and it does not support the --enable_htp flag
        quant_cmd = "snpe-dlc-quant"
    cmd = f"{quant_cmd} --input_dlc {dlc_path} --input_list {input_list} --output_dlc {output_file}"
    if config["use_enhanced_quantizer"]:
        cmd += " --use_enhanced_quantizer"
    if config["enable_htp"]:
        if platform.system() == OS.WINDOWS:
            logger.warning("--enable_htp is not supported on Windows")
        else:
            cmd += " --enable_htp"
    if config["htp_socs"] is not None:
        cmd += f" --htp_socs {','.join(config['htp_socs'])}"
    if config["extra_args"] is not None:
        cmd += " " + config["extra_args"]

    _, stderr = SNPERunner(use_dev_tools=True).run(cmd)

    # check if quantization succeeded
    if not ("Writing quantized model" in stderr or "Saved quantized dlc" in stderr):
        raise RuntimeError(stderr)


def dlc_to_onnx(
    dlc_path: str,
    config: dict,
    input_names: List[str],
    input_shapes: List[List[int]],
    output_names: List[str],
    output_shapes: List[List[int]],
) -> onnx.ModelProto:
    """Convert a SNPE DLC to ONNX. The DLC is wrapped in a ONNX model for use with onnxruntime SNPE EP.

    Returns an ONNX ModelProto.

    dlc_path: path to the DLC file.
    config: Config dict with the following keys:
        target_device: str = target device for the ONNX-wrapped SNPE model.
        target_opset: int = target ONNX opset for the ONNX-wrapped SNPE model.
    input_names: List[str] = list of input names.
    input_shapes: List[List[int]] = list of input shapes.
    output_names: List[str] = list of output names.
    output_shapes: List[List[int]] = list of output shapes.
    """
    # get the SNPE version used to create the DLC
    snpe_version = get_dlc_snpe_version(dlc_path)

    # create the dlc info and read the content
    dlc_info = get_dlc_info(dlc_path)
    if f"{input_names[0]}:0" in dlc_info:
        input_names = [f"{x}:0" for x in input_names]
        output_names = [f"{x}:0" for x in output_names]

    with Path(dlc_path, "rb").open() as file:
        dlc_content = file.read()

    model_name = Path(dlc_path).stem

    # Loop over all input and output tensor and its dim for the model
    inputs_tensor_val = []
    for input_name, input_shape in zip(input_names, input_shapes):
        inputs_tensor_val.append(helper.make_tensor_value_info(input_name, TensorProto.FLOAT, input_shape))

    outputs_tensor_val = []
    for output_name, output_shape in zip(output_names, output_shapes):
        outputs_tensor_val.append(helper.make_tensor_value_info(output_name, TensorProto.FLOAT, output_shape))

    # Create the node
    snpe_node = helper.make_node(
        "Snpe",
        input_names,
        output_names,
        name=model_name,
        DLC=dlc_content,
        snpe_version=snpe_version,
        target_device=config["target_device"].upper(),
        notes="snpe dlc model",
        domain="com.microsoft",
    )

    # Create the graph
    graph_def = helper.make_graph([snpe_node], model_name, inputs_tensor_val, outputs_tensor_val)

    op = onnx.OperatorSetIdProto()
    op.version = config["target_opset"]
    return helper.make_model(graph_def, producer_name="Olive", opset_imports=[op])
