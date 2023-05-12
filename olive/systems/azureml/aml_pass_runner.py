# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
import shutil
import tempfile
from pathlib import Path

from olive.model import ModelConfig
from olive.passes import REGISTRY as PASS_REGISTRY
from olive.passes import FullPassConfig
from olive.systems.utils import get_model_config, parse_common_args


def parse_pass_config_arg(raw_args):
    parser = argparse.ArgumentParser("Pass config")

    # parse config arg
    parser.add_argument("--pass_config", type=str, help="pass config", required=True)

    return parser.parse_known_args(raw_args)


def parse_pass_args(pass_type, raw_args):
    pass_class = PASS_REGISTRY[pass_type]

    parser = argparse.ArgumentParser(f"{pass_type} pass args")

    # TODO: get accelerator specs from args when it is implemented
    # parse pass args
    for param, param_config in pass_class.default_config().items():
        if param_config.is_path:
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
    pass_config = json.load(open(pass_config_arg.pass_config))
    pass_type = pass_config["type"].lower()

    # TODO: contact ort team for a workaround
    # Some passes create temporary files in the same directory as the model
    # original directory for model path is read only, so we need to copy the model to a temp directory
    # TODO: test if sym link solves it
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
    pass_args = parse_pass_args(pass_type, extra_args)

    # load input_model
    input_model_config = get_model_config(common_args)
    # Replace HF config model_name with input model path to load model from input model path
    hf_config = None
    if input_model_config["config"].get("hf_config") and not input_model_config["config"]["hf_config"]["load_model_from_hub"]:
        hf_config = input_model_config["config"]["hf_config"].copy()
        input_model_config["config"]["hf_config"]["model_name"] = common_args.model_path

    input_model = ModelConfig.from_json(input_model_config).create_model()
    input_model_path = str(Path(input_model.model_path).resolve()) if input_model.model_path is not None else None

    # load pass
    p = create_pass(pass_config, pass_args)

    # output model path
    output_model_path = str(Path(common_args.pipeline_output) / "output_model")

    # run pass
    output_model = p.run(input_model, output_model_path)

    # save model json
    model_json = output_model.to_json()

    # Replace output model HF config with input model HF config
    if hf_config:
        model_json["config"]["hf_config"] = hf_config

    # this is to handle passes like OrtPerfTuning that use the same model file as input
    model_json["same_model_path_as_input"] = False
    if model_json["config"]["model_path"] is not None:
        model_path = str(Path(model_json["config"]["model_path"]).resolve())
        if model_path == input_model_path:
            model_json["config"]["model_path"] = None
            model_json["same_model_path_as_input"] = True
        else:
            model_json["config"]["model_path"] = str(Path(model_path).relative_to(Path(common_args.pipeline_output)))
    json.dump(model_json, open(Path(common_args.pipeline_output) / "output_model_config.json", "w"), indent=4)


if __name__ == "__main__":
    main()
