# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
import shutil
import tempfile
from pathlib import Path

from onnxruntime import __version__ as ort_version
from packaging import version

from olive.common.config_utils import ParamCategory, validate_config
from olive.common.utils import aml_runner_hf_login, copy_dir
from olive.data.config import DataConfig
from olive.hardware import AcceleratorSpec
from olive.logging import set_verbosity_from_env
from olive.model import ModelConfig
from olive.package_config import OlivePackageConfig
from olive.passes import REGISTRY as PASS_REGISTRY
from olive.passes import FullPassConfig, Pass
from olive.resource_path import create_resource_path
from olive.systems.utils import get_common_args


def parse_pass_config_arg(raw_args):
    parser = argparse.ArgumentParser("Pass config")

    # parse config arg
    parser.add_argument("--pass_config", type=str, help="pass config", required=True)
    parser.add_argument("--pass_accelerator_type", type=str, help="pass accelerator type", default="cpu")
    parser.add_argument(
        "--pass_execution_provider", type=str, help="pass execution provider", default="CPUExecutionProvider"
    )

    return parser.parse_known_args(raw_args)


def parse_pass_args(pass_type, accelerator_spec, raw_args):
    pass_class = PASS_REGISTRY[pass_type]

    parser = argparse.ArgumentParser(f"{pass_type} pass args")

    # parse pass args
    for param, param_config in pass_class.default_config(accelerator_spec).items():
        if param_config.category in (ParamCategory.PATH, ParamCategory.DATA):
            parser.add_argument(f"--pass_{param}", type=str, help=f"pass {param}", required=param_config.required)

    return parser.parse_known_args(raw_args)


def create_pass(pass_config, pass_args):
    for key, value in vars(pass_args).items():
        if value is not None:
            # remove the pass_ prefix, the 1 is to only replace the first occurrence
            normalized_key = key.replace("pass_", "", 1)
            pass_config["config"][normalized_key] = value

    return FullPassConfig.from_json(pass_config).create_pass()


def parse_data_item(data_name, item, extra_args):
    data_config_parser = argparse.ArgumentParser(f"Data {item} parser")
    data_config_parser.add_argument(f"--{data_name}_{item}", type=str, help=f"{data_name} {item}")

    item_args, extra_args = data_config_parser.parse_known_args(extra_args)

    for key, value in vars(item_args).items():
        if item in key:
            return value
    return None


def update_data_config(p: "Pass", extra_args):
    data_map = {}
    for param, param_config in p.config.items():
        if param.endswith("data_config") and param_config is not None:
            data_name = param_config["name"]
            if data_map.get(data_name):
                user_script, script_dir, data_dir, data_files = data_map[data_name]
            else:
                user_script = parse_data_item(data_name, "user_script", extra_args)
                script_dir = parse_data_item(data_name, "script_dir", extra_args)
                data_dir = parse_data_item(data_name, "data_dir", extra_args)
                data_files = parse_data_item(data_name, "data_files", extra_args)
                data_map[data_name] = (user_script, script_dir, data_dir, data_files)

            param_config["user_script"] = user_script
            param_config["script_dir"] = script_dir
            if param_config.get("params_config"):
                param_config["params_config"]["data_dir"] = data_dir
                param_config["params_config"]["data_files"] = data_files

            p.config[param] = validate_config(param_config, DataConfig)


def main(raw_args=None):
    set_verbosity_from_env()

    # login to hf if HF_LOGIN is set to True
    aml_runner_hf_login()

    input_model_config, pipeline_output, extra_args = get_common_args(raw_args)
    pass_config_arg, extra_args = parse_pass_config_arg(extra_args)

    # pass config
    with open(pass_config_arg.pass_config) as f:
        pass_config = json.load(f)
    pass_type = pass_config["type"].lower()

    # Import the pass package configuration from the package_config
    package_config = OlivePackageConfig.load_default_config()
    package_config.import_pass_module(pass_config["type"])

    if version.parse(ort_version) < version.parse("1.16.0"):
        # In onnxruntime, the following PRs will make the optimize_model save external data in the temporary folder
        # * https://github.com/microsoft/onnxruntime/pull/16531
        # * https://github.com/microsoft/onnxruntime/pull/16716
        # * https://github.com/microsoft/onnxruntime/pull/16912
        # So, in 1.16.0 afterwards, we don't need to copy the model to a temp directory

        # Some passes create temporary files in the same directory as the model
        # original directory for model path is read only, so we need to copy the model to a temp directory
        input_model_path = input_model_config["config"].get("model_path")
        if input_model_path is not None:
            tmp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
            old_path = Path(input_model_path).resolve()
            new_path = Path(tmp_dir.name).resolve() / old_path.name
            if old_path.is_file():
                shutil.copy(old_path, new_path)
            else:
                new_path.mkdir(parents=True, exist_ok=True)
                copy_dir(old_path, new_path, dirs_exist_ok=True)
            input_model_config["config"]["model_path"] = str(new_path)

    # pass specific args
    accelerator_spec = AcceleratorSpec(pass_config_arg.pass_accelerator_type, pass_config_arg.pass_execution_provider)
    pass_args, extra_args = parse_pass_args(pass_type, accelerator_spec, extra_args)

    # load input_model
    input_model = ModelConfig.from_json(input_model_config).create_model()

    # load pass
    p = create_pass(pass_config, pass_args)

    update_data_config(p, extra_args)

    # output model path
    output_model_path = str(Path(pipeline_output) / "output_model")

    # run pass
    output_model = p.run(input_model, None, output_model_path)

    # save model json
    model_json = output_model.to_json()

    # Replace local paths with relative paths
    # keep track of the resource names that are relative paths
    # during download in aml system, the relative paths will be resolved to the pipeline output
    model_json["resource_names"] = []
    # keep track of the resource names that are the same as the input model
    model_json["same_resources_as_input"] = []
    for resource_name, resource_path_value in output_model.resource_paths.items():
        resource_path = create_resource_path(resource_path_value)  # just in case
        if not resource_path or resource_path.is_string_name():
            # nothing to do if the path is None or a string name
            continue
        # check if the path is the same as the input model's path
        resource_path_str = resource_path.get_path()
        # input model might not have the resource, e.g. pytorch model -> onnx model
        input_resource_path = create_resource_path(input_model.resource_paths.get(resource_name))
        input_resource_path_str = input_resource_path.get_path() if input_resource_path else None
        if input_resource_path_str == resource_path_str:
            # if the path is the same as the input path, set the path to None
            # and add the resource name to the same_resources_as_input list
            model_json["config"][resource_name] = None
            model_json["same_resources_as_input"].append(resource_name)
            continue
        # need to ensure that the path is a local resource
        assert resource_path.is_local_resource(), f"Expected local resource, got {resource_path.type}"
        # if the model is a local file or folder, set the model path to be relative to the pipeline output
        # the aml system will resolve the relative path to the pipeline output during download
        relative_path = str(Path(resource_path_str).relative_to(Path(pipeline_output)))
        path_json = resource_path.to_json()
        path_json["config"]["path"] = relative_path
        model_json["config"][resource_name] = path_json
        model_json["resource_names"].append(resource_name)

    # save model json
    with (Path(pipeline_output) / "output_model_config.json").open("w") as f:
        json.dump(model_json, f, indent=4)


if __name__ == "__main__":
    main()
