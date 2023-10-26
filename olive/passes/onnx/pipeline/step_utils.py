# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import importlib as imp
from typing import Dict, List, Union

import onnx

# pylint: disable=wildcard-import
from onnxruntime_extensions.tools.pre_post_processing import *  # noqa: F401, F403, RUF100
from onnxruntime_extensions.tools.pre_post_processing.utils import create_named_value

from olive.passes.onnx.pipeline import resolve_placeholder

# ruff: noqa: RUF100, PLW2901


def parse_steps(model: onnx.ModelProto, config: List[Dict]):
    """Parse the config and return a dictionary of step name and its parameters.

    ## Config examples:
    1. if there io_map is missing, the config could be simplified as:
        ```
        [
            {
                "ReverseAxis": {
                    "axis": 2,
                    "dim_value": 3,
                    "name": "BGR_to_RGB"
                }
            },
            {
                "Normalize": {
                    "normalization_values": [[0.485, 0.229],[0.456,0.224], [0.406,0.225]],
                    "layout": "CHW"
                }
            }
        ]
        ```

    2. if there is io_map, the config looks like:
        ```
        {
            "Resize": {
                "params": {
                    "resize_to": [
                        {
                            "type": "__model_input__",
                            "input_index": 0,
                            "dim_index": -1
                        },
                        {
                            "type": "__model_output__",
                            "output_index": 0,
                            "dim_index": -2
                        }
                    ]
                },
                "io_map": [["ConvertImageToBGR",0,0]]
            }
        }
        ```
    ## Config schema:

    1. simple form:
         ```
         {"step_name": {"param_name": param_value}}
         ```

    2. full form:
        ```
        {
            "step_name": {
                "params": {"param_name": param_value},
                "io_map": [[step_name, input_index, output_index]]
            }
        }
        ```

       The detailed io_map definition and usage is
       https://github.com/microsoft/onnxruntime-extensions/blob/main/onnxruntime_extensions/tools/Example%20usage%20of%20the%20PrePostProcessor.md#iomapentry-usage

       Please note the io_map is optional, if it is missing, the config is simplified as the simple form.

    3. Customized parameter type for parameter value:
        ```
        {
            "SentencePieceTokenizer": {
                "tokenizer_param": {
                    "type": "TokenizerParam",
                    "params": {
                        "vocab_or_file": "test_data/sentencepiece/test_model.model"
                    }
                }
            }
        }
        ```
       In this example, the type will be looked in the imported modules. So, please remember to import it before
       looking up the type. The params dictionary is the parameters for the type class.

    4. model input and output placeholder.
       The PrePostPipeline will add some operators into the onnx graph. Sometimes, we need specify the model input or
       output pins for the steps definition. For example, we need use w_in and h_in (input model shape dimension) in
       Resize steps in superresolution. For this purpose, the following two placeholder type is introduced:
       a), `__model_input__`: the input model shape dimension. The input_index is the input index of the model.
            ```
            {
                "type": "__model_input__",
                "input_index": 0,
                "dim_index": -1
            }
            ```
        b), `__model_output__`: the output model shape dimension. The output_index is the output index of the model.
            ```
            {
                "type": "__model_output__",
                "output_index": 0,
                "dim_index": -2
            }
            ```
        When these config is parsed, they will be converted to using the ModelProto shape information.

    5. For list value, they will be treated to tuple in the function. User could explicitly specify the type as list if
       the default tuple behavior is not expected. Instead of using `"param_name": param_value`, user could use the
       following form by using `type` and `value`:
            ```
            "explicit_list": {
                "type": "list",
                "value": [
                    1,
                    2,
                    3
                ]
            }
            ```
    6. the full example of the config is here:
        TODO: add the file link demonstration here.

    """
    step_configs = []
    for step_config in config:
        step_config_result = parse_step_config(model, step_config)
        step_configs.append(step_config_result[0])
    return step_configs


def parse_step_config(model: onnx.ModelProto, config: Dict):
    """Parse a single step with its config parameters."""
    assert len(config) == 1, "The config should only have one step name and its parameters."

    step_config_result = []
    for step_name, step_config in config.items():
        step_config_result.append((step_name, parse_step_params(model, step_config)))

    return step_config_result


def parse_step_params(model: onnx.ModelProto, step_config: Dict):
    """Parse the step parameters for a single step."""
    step_params = step_config.get("params")
    if step_params is None:
        step_params = step_config

    params = {}
    for param_name, param_value in step_params.items():
        # print(param_name, param_value)
        if isinstance(param_value, list):
            resolved_params = []
            for p_v in param_value:
                if isinstance(p_v, dict):
                    resolved_params.append(resolve_placeholder(model, p_v))
                else:
                    resolved_params.append(p_v)
            params[param_name] = tuple(resolved_params)
        elif isinstance(param_value, dict):
            param_type = param_value.get("type")
            param_args = param_value.get("params")
            if param_type and param_args:
                # Customized type definition
                param_cls = get_customized_class(param_type)
                params[param_name] = param_cls(**param_args)
            elif param_type in ("tuple", "list"):
                param_value = param_value.get("value")

                # explicitly list or tuple type is specified
                assert isinstance(param_value, list)

                resolved_params = []
                for p_v in param_value:
                    if isinstance(p_v, dict):
                        # handle placeholder
                        resolved_params.append(resolve_placeholder(model, p_v))
                    else:
                        resolved_params.append(p_v)

                if param_type == "tuple":
                    params[param_name] = tuple(resolved_params)
                else:
                    params[param_name] = list(resolved_params)
            elif param_type in ("__model_input__", "__model_output__"):
                params[param_name] = resolve_placeholder(model, param_value)
        else:
            params[param_name] = param_value

    io_map = step_config.get("io_map")
    if io_map:
        assert isinstance(io_map, list)
        params = (params, io_map)

    return params


def create_pipeline_inputs(name: str, data_type: int, shape: List[Union[int, str]]) -> onnx.ValueInfoProto:
    return create_named_value(name, data_type, shape)


def get_customized_class(class_name: str):
    cls = globals().get(class_name)
    if cls is None:
        # cannot find the class in the current module, try to import it by using the full name
        def import_object(name: str):
            components = name.split(".")
            mod = imp.import_module(components[0])
            for comp in components[1:]:
                mod = getattr(mod, comp)
            return mod

        cls = import_object(class_name)

    if cls is None:
        raise ValueError(f"Cannot find the class {class_name} in the current module or imported modules.")

    return cls
