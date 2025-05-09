from __future__ import annotations

import json
from pathlib import Path
from time import time
from typing import TYPE_CHECKING

from logger import get_logger, set_logger_level
from pydantic import BaseModel, ValidationError

from olive.model.config import ModelConfig
from olive.package_config import OlivePackageConfig
from olive.resource_path import LocalFile, create_resource_path
from olive.systems.accelerator_creator import create_accelerators
from olive.systems.system_config import SystemConfig
from olive.workflows.run.config import RunConfig

if TYPE_CHECKING:
    from olive.evaluator.metric_result import MetricResult
    from olive.systems.olive_system import OliveSystem

logger = get_logger("Evaluate")


class InferenceSetting(BaseModel):
    execution_provider: str
    provider_options: list[dict] | None = None
    session_options: dict | None = {}


class InferenceConfig(BaseModel):
    inference_settings: list[InferenceSetting] | None = None


def load_systems(config_file: Path) -> dict[str, SystemConfig]:
    try:
        with open(config_file) as f:
            data = json.load(f)

        result = {}
        for key, value in data.items():
            try:
                result[key] = SystemConfig(**value)
            except ValidationError:
                logger.exception("Validation error for key {key}!")
                raise
        return result
    except FileNotFoundError:
        logger.exception("Systome config file %s not found!", config_file)
        raise
    except json.JSONDecodeError:
        logger.exception("Invalid JSON in %s!", config_file)
        raise


def load_inference_config(config_file: Path) -> dict[str, InferenceConfig]:
    try:
        with open(config_file) as f:
            data = json.load(f)

        settings = {}
        for key, value in data.items():
            try:
                settings[key] = InferenceConfig(**value)
            except ValidationError:
                logger.exception("Validation error for key %s!", key)
                raise
        return settings
    except FileNotFoundError:
        logger.exception("Inference setting config file %s not found!", config_file)
        raise
    except json.JSONDecodeError:
        logger.exception("Invalid JSON in %s!", config_file)
        raise


def evaluate(
    config_file: Path,
    evaluator: str | None = None,
    target: str | None = None,
    extra_systems: dict[str, SystemConfig] | Path | None = None,
    inference_configs: dict[str, InferenceConfig] | Path | None = None,
):
    import os
    from typing import cast

    logger.info("Set working directory to %s", config_file.parent)
    os.chdir(config_file.parent)

    logger.info("Parsing Olive config file %s...", config_file)
    run_config = cast("RunConfig", RunConfig.parse_file_or_obj(config_file))
    package_config = OlivePackageConfig.parse_file_or_obj(OlivePackageConfig.get_default_config_path())

    logger.info("Creating Olive engine ...")
    engine = run_config.engine.create_engine(
        olive_config=package_config,
        azureml_client_config=None,
        workflow_id=run_config.workflow_id,
    )
    engine.initialize()

    logger.info("Creating target system ...")
    start_time = time()
    if target is not None:
        systems = (
            load_systems(extra_systems) if isinstance(extra_systems, Path) else extra_systems or run_config.systems
        )
        target_config = systems.get(target, engine.target_config)
    else:
        target_config = engine.target_config

    accelerator_specs = create_accelerators(
        target_config,
        skip_supported_eps_check=False,
        is_ep_required=True,
    )

    target_system: OliveSystem = target_config.create_system()
    logger.info("Target system created in %f seconds", time() - start_time)

    # load converted model from output dir
    p = Path(run_config.engine.output_dir) / "model_config.json"
    if not p.exists():
        raise FileNotFoundError(f"Model config file {p} does not exist.")

    logger.info("Parsing model config ...")
    model_config_file: LocalFile = cast("LocalFile", create_resource_path(p))
    model_config = cast(
        "ModelConfig",
        ModelConfig.parse_file_or_obj(model_config_file.get_path()),
    )
    logger.info("Model path: %s", model_config.config["model_path"])

    logger.info("Parsing evaluator config: locating %s...", evaluator)
    evaluator_config = (
        run_config.evaluators.get(
            evaluator,
            engine.evaluator_config,
        )
        if evaluator is not None
        else engine.evaluator_config
    )
    if evaluator_config is None:
        raise ValueError(
            "Evaluator is either not specified or doesn't exist. Available "
            f"evaluators are: {list(run_config.evaluators.keys())}"
        )
    else:
        logger.info("Evaluator config found: %s", evaluator_config.name)

    inference_settings: list[InferenceSetting] = []
    inference_configs = (
        load_inference_config(inference_configs) if isinstance(inference_configs, Path) else inference_configs
    )
    if inference_configs and target:
        inference_cfg = inference_configs.get(target, None)
        if inference_cfg and inference_cfg.inference_settings:
            inference_settings.extend(inference_cfg.inference_settings)
    if len(inference_settings) > 0:
        logger.info("Adopting inference settings: %s", inference_settings[0])
        model_config.config["inference_settings"] = inference_settings[0].model_dump()

    logger.info("Evaluating model ...")
    result: MetricResult = target_system.evaluate_model(
        model_config=model_config,
        evaluator_config=evaluator_config,
        accelerator=accelerator_specs[0],
    )
    logger.info("Evaluation result: %s", result.to_json())


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description=("Evaluate ONNX model that is converted with Olive using ONNX Runtime.")
    )
    parser.add_argument(
        "--run-config",
        "--config",
        type=str,
        help="Path to json config file",
        required=True,
    )
    parser.add_argument(
        "--tempdir",
        type=str,
        help="Root directory for tempfile directories and files",
        required=False,
    )
    parser.add_argument(
        "--evaluator",
        type=str,
        help="Evaluator name to use for evaluation",
        required=False,
    )
    parser.add_argument(
        "--target",
        type=str,
        help="Target system to use for evaluation",
        required=False,
    )
    parser.add_argument(
        "--system-config",
        type=Path,
        help="Path to json config file for extra systems",
        required=False,
    )

    return parser.parse_args()


if __name__ == "__main__":
    set_logger_level(logger, "INFO")

    args = parse_args()

    config_file_path = Path(args.run_config).resolve()
    if not config_file_path.exists():
        raise FileNotFoundError(f"Config file {config_file_path} does not exist.")

    extra_config = args.system_config.resolve() if args.system_config else Path(__file__).parent / "system_config.json"

    evaluate(
        config_file=config_file_path,
        evaluator=args.evaluator,
        target=args.target,
        extra_systems=extra_config,
        inference_configs=extra_config,
    )
