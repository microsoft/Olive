# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from typing import TYPE_CHECKING

from olive.common.hf.login import aml_runner_hf_login
from olive.evaluator.olive_evaluator import OliveEvaluatorConfig
from olive.hardware import AcceleratorSpec
from olive.logging import set_verbosity_from_env
from olive.model import ModelConfig
from olive.systems.local import LocalSystem
from olive.systems.utils import get_common_args, parse_config

if TYPE_CHECKING:
    from olive.systems.olive_system import OliveSystem


def main(raw_args=None):
    set_verbosity_from_env()

    # login to hf if HF_LOGIN is set to True
    aml_runner_hf_login()

    pipeline_output, resources, model_config, extra_args = get_common_args(raw_args)
    evaluator_config, extra_args = parse_config(extra_args, "evaluator", resources)
    accelerator_config, extra_args = parse_config(extra_args, "accelerator", resources)

    # load evaluator config
    evaluator_config = OliveEvaluatorConfig.from_json(evaluator_config)

    # load model config
    model_config = ModelConfig.from_json(model_config)

    # load accelerator spec
    accelerator_spec = AcceleratorSpec(**accelerator_config)

    target: OliveSystem = LocalSystem()

    # metric result
    metric_result = target.evaluate_model(model_config, evaluator_config, accelerator_spec)

    # save metric result json
    with (Path(pipeline_output) / "metric_result.json").open("w") as f:
        f.write(metric_result.json())


if __name__ == "__main__":
    main()
