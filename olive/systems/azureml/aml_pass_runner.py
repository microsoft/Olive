# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
from pathlib import Path

from olive.common.hf.login import aml_runner_hf_login
from olive.common.utils import set_nested_dict_value
from olive.logging import set_verbosity_from_env
from olive.model import ModelConfig
from olive.package_config import OlivePackageConfig
from olive.passes import FullPassConfig
from olive.resource_path import find_all_resources
from olive.systems.utils import get_common_args, parse_config


def main(raw_args=None):
    set_verbosity_from_env()

    # login to hf if HF_LOGIN is set to True
    aml_runner_hf_login()

    pipeline_output, resources, input_model_config, extra_args = get_common_args(raw_args)
    pass_config, extra_args = parse_config(extra_args, "pass", resources)

    # Import the pass package configuration from the package_config
    package_config = OlivePackageConfig.load_default_config()
    package_config.import_pass_module(pass_config["type"])

    # load input_model
    input_model = ModelConfig.from_json(input_model_config).create_model()

    # load pass
    p = FullPassConfig.from_json(pass_config).create_pass()

    # output model path
    output_model_path = str(Path(pipeline_output) / "output_model")

    # run pass
    output_model = p.run(input_model, output_model_path)

    # save model json
    model_json = output_model.to_json()

    # Replace local paths with relative paths
    # keep track of the resource names that are relative paths
    # during download in aml system, the relative paths will be resolved to the pipeline output
    model_json["resources"] = []
    # keep track of the resource names that are the same as the input model
    model_json["same_resources_as_input"] = []
    # resources, no need to look at model_attributes. It might have unintended resources like
    # _model_name_or_path from hf config which points to where the config was loaded from
    input_model_resources = find_all_resources(input_model_config, ignore_keys=["model_attributes"])
    output_model_resources = find_all_resources(model_json, ignore_keys=["model_attributes"])
    for resource_key, resource_path in output_model_resources.items():
        input_model_resource_path = input_model_resources.get(resource_key)
        if resource_path.get_path() == (input_model_resource_path.get_path() if input_model_resource_path else None):
            set_nested_dict_value(model_json, resource_key, None)
            model_json["same_resources_as_input"].append(resource_key)
            continue
        # need to ensure that the path is a local resource
        assert resource_path.is_local_resource(), f"Expected local resource, got {resource_path.type}"
        # if the model is a local file or folder, set the model path to be relative to the pipeline output
        # the aml system will resolve the relative path to the pipeline output during download
        relative_path = str(Path(resource_path.get_path()).relative_to(Path(pipeline_output)))
        path_json = resource_path.to_json()
        path_json["config"]["path"] = relative_path
        set_nested_dict_value(model_json, resource_key, path_json)
        model_json["resources"].append(resource_key)

    # save model json
    with (Path(pipeline_output) / "output_model_config.json").open("w") as f:
        json.dump(model_json, f, indent=4)


if __name__ == "__main__":
    main()
