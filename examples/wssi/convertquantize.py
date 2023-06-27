# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
from pathlib import Path
from typing import Dict, Union

from config import ConvertQuantizeConfig, IOConfig

from olive.common.config_utils import validate_config
from olive.data.template import raw_data_config_template
from olive.workflows import run as olive_run


def convertquantize(config: Union[str, Dict, ConvertQuantizeConfig], tool: str = None):
    if type(config) is str:
        config = json.load(open(config))
    config = validate_config(config, ConvertQuantizeConfig)

    model_name = config.model.stem
    output_dir = config.model.resolve().parent if config.output_dir is None else config.output_dir.resolve()

    model = get_model(config.model)
    data_config = get_data_config(config)

    if tool == "snpe":
        passes = get_snpe_passes(config.io_config, config.tools.get("snpe", {}), model_name)
    elif tool == "openvino":
        passes = get_ov_passes(config.io_config, config.tools.get("openvino", {}), model_name)
    else:
        raise ValueError(f"Unsupported tool: {tool}")

    run_config = {
        "input_model": model,
        "data_configs": {"raw_data": data_config},
        "passes": passes,
        "engine": {
            "log_severity_level": config.log_severity_level,
            "search_strategy": False,
            "output_dir": output_dir,
            "clean_cache": True,
        },
    }
    olive_run(run_config)


def get_model(model_path: Path):
    if model_path.suffix == ".onnx":
        model_type = "ONNXModel"
    elif model_path.suffix == ".pb":
        model_type = "TensorFlowModel"
    else:
        raise Exception(f"Unsupported model format: {model_path.suffix}")
    return {"type": model_type, "config": {"model_path": str(model_path)}}


def get_data_config(config: ConvertQuantizeConfig):
    return raw_data_config_template(
        data_dir=config.quant_data,
        input_names=config.io_config.input_names,
        input_shapes=config.io_config.input_shapes,
        input_dirs=config.input_dirs,
        input_order_file=config.input_order_file,
    )


def get_snpe_passes(io_config: IOConfig, tool_options: Dict, model_name: str):
    snpe_conversion = {
        "type": "SNPEConversion",
        "config": {
            "input_names": io_config.input_names,
            "input_shapes": io_config.input_shapes,
            "output_names": io_config.output_names,
            **(tool_options.convert_options or {}),
        },
        "output_name": f"{model_name}_snpe",
    }
    snpe_quantization = {
        "type": "SNPEQuantization",
        "config": {
            "data_config": "raw_data",
            **(tool_options.quantize_options or {}),
        },
        "output_name": f"{model_name}_snpe_quantized",
    }
    return {"snpe_conversion": snpe_conversion, "snpe_quantization": snpe_quantization}


def get_ov_passes(io_config: IOConfig, tool_options: Dict, model_name: str):
    io_args = {
        "input": ",".join(io_config.input_names),
        "input_shape": ",".join([str(x) for x in io_config.input_shapes]).replace(" ", ""),
        "output": ",".join(io_config.output_names),
    }
    ov_conversion = {
        "type": "OpenVINOConversion",
        "config": {"extra_config": {**io_args, **(tool_options.convert_options or {})}},
        "output_name": f"{model_name}_ov",
    }
    ov_quantization = {
        "type": "OpenVINOQuantization",
        "config": {
            "data_config": "raw_data",
            **(tool_options.quantize_options or {}),
        },
        "output_name": f"{model_name}_ov_quantized",
    }
    return {"ov_conversion": ov_conversion, "ov_quantization": ov_quantization}


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Olive Workflow: ConvertQuantize")
    parser.add_argument("--config", type=str, help="Path to json config file", required=True)
    parser.add_argument(
        "--tool", type=str, help="Tool to use for conversion", choices=["snpe", "openvino"], required=True
    )

    args = parser.parse_args()

    convertquantize(**vars(args))
