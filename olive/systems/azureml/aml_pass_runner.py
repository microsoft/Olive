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

from olive.common.config_utils import ParamCategory
from olive.hardware import AcceleratorSpec
from olive.model import ModelConfig
from olive.passes import REGISTRY as PASS_REGISTRY
from olive.passes import FullPassConfig
from olive.resource_path import create_resource_path
from olive.systems.utils import get_model_config, parse_common_args


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

    return parser.parse_args(raw_args)


def create_pass(pass_config, pass_args):
    for key, value in vars(pass_args).items():
        if value is not None:
            key = key.replace("pass_", "")
            pass_config["config"][key] = value

    p = FullPassConfig.from_json(pass_config).create_pass()
    return p


def main(raw_args=None):
    common_args, extra_args = parse_common_args(raw_args)
    pass_config_arg, extra_args = parse_pass_config_arg(extra_args)

    # pass config
    with open(pass_config_arg.pass_config) as f:
        pass_config = json.load(f)
    pass_type = pass_config["type"].lower()

    if version.parse(ort_version) < version.parse("1.16.0"):
        # In onnxruntime, the following PRs will make the optimize_model save external data in the temporary folder
        # * https://github.com/microsoft/onnxruntime/pull/16531
        # * https://github.com/microsoft/onnxruntime/pull/16716
        # * https://github.com/microsoft/onnxruntime/pull/16912
        # So, in 1.16.0 afterwards, we don't need to copy the model to a temp directory

        # Some passes create temporary files in the same directory as the model
        # original directory for model path is read only, so we need to copy the model to a temp directory
        if common_args.model_path is not None:
            tmp_dir = tempfile.TemporaryDirectory()
            old_path = Path(common_args.model_path).resolve()
            new_path = Path(tmp_dir.name).resolve() / old_path.name
            if old_path.is_file():
                shutil.copy(old_path, new_path)
            else:
                new_path.mkdir(parents=True, exist_ok=True)
                shutil.copytree(old_path, new_path, dirs_exist_ok=True)
            common_args.model_path = str(new_path)

    # pass specific args
    accelerator_spec = AcceleratorSpec(pass_config_arg.pass_accelerator_type, pass_config_arg.pass_execution_provider)
    pass_args = parse_pass_args(pass_type, accelerator_spec, extra_args)

    # load input_model
    input_model_config = get_model_config(common_args)
    input_model = ModelConfig.from_json(input_model_config).create_model()
    input_model_path = str(Path(input_model.model_path).resolve()) if input_model.model_path is not None else None

    # load pass
    p = create_pass(pass_config, pass_args)

    # output model path
    output_model_path = str(Path(common_args.pipeline_output) / "output_model")

    # run pass
    output_model = p.run(input_model, None, output_model_path)

    # save model json
    model_json = output_model.to_json()

    # Replace output model HF config with input model HF config
    if input_model_config["config"].get("hf_config"):
        model_json["config"]["hf_config"] = input_model_config["config"]["hf_config"]

    # this is to handle passes like OrtPerfTuning that use the same model file as input
    model_json["same_model_path_as_input"] = False
    if model_json["config"]["model_path"] is not None:
        # create a resource path from the model path
        model_resource_path = create_resource_path(model_json["config"]["model_path"])
        # we currently only have passes that generate local files or folders
        # check that this is true
        assert model_resource_path.is_local_resource(), f"Expected local resource, got {model_resource_path.type}"
        # string representation of the model path
        model_path_str = model_resource_path.get_path()
        if model_path_str == input_model_path:
            # if the model path is the same as the input model path, set model_path to None
            # and set same_model_path_as_input to True
            model_json["config"]["model_path"] = None
            model_json["same_model_path_as_input"] = True
        else:
            # if the model is a local file or folder, set the model path to be relative to the pipeline output
            # the aml system will resolve the relative path to the pipeline output during download
            relative_path = str(Path(model_path_str).relative_to(Path(common_args.pipeline_output)))
            model_path_json = model_resource_path.to_json()
            model_path_json["config"]["path"] = relative_path
            model_json["config"]["model_path"] = model_path_json

    # save model json
    with open(Path(common_args.pipeline_output) / "output_model_config.json", "w") as f:
        json.dump(model_json, f, indent=4)


if __name__ == "__main__":
    main()
