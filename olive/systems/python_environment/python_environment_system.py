# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
import os
import pickle
import platform
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from olive.common.utils import run_subprocess
from olive.evaluator.metric import Metric, MetricResult
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ModelConfig
from olive.systems.common import SystemType
from olive.systems.olive_system import OliveSystem
from olive.systems.system_config import PythonEnvironmentTargetUserConfig
from olive.systems.utils import get_package_name_from_ep

if TYPE_CHECKING:
    from olive.passes.olive_pass import Pass


logger = logging.getLogger(__name__)


class PythonEnvironmentSystem(OliveSystem):
    system_type = SystemType.PythonEnvironment

    def __init__(
        self,
        python_environment_path: Union[Path, str] = None,
        environment_variables: Dict[str, str] = None,
        prepend_to_path: List[str] = None,
        accelerators: List[str] = None,
        olive_managed_env: bool = False,
        requirements_file: Union[Path, str] = None,
        hf_token: bool = None,
    ):
        super().__init__(accelerators=accelerators, olive_managed_env=olive_managed_env)
        self.config = PythonEnvironmentTargetUserConfig(
            python_environment_path=python_environment_path,
            environment_variables=environment_variables,
            prepend_to_path=prepend_to_path,
            accelerators=accelerators,
            olive_managed_env=olive_managed_env,
            requirements_file=requirements_file,
        )
        self.environ = deepcopy(os.environ)
        if self.config.environment_variables:
            self.environ.update(self.config.environment_variables)
        if self.config.prepend_to_path:
            self.environ["PATH"] = os.pathsep.join(self.config.prepend_to_path) + os.pathsep + self.environ["PATH"]
        if self.config.python_environment_path:
            self.environ["PATH"] = str(self.config.python_environment_path) + os.pathsep + self.environ["PATH"]
        if self.config.olive_managed_env:
            if platform.system() == "Linux":
                temp_dir = os.path.join(os.environ.get("HOME", ""), "tmp")
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)
                self.environ["TMPDIR"] = temp_dir
            else:
                self.environ["TMPDIR"] = tempfile.TemporaryDirectory().name  # pylint: disable=consider-using-with

        # available eps. This will be populated the first time self.get_supported_execution_providers() is called.
        # used for caching the available eps
        self.available_eps = None

        # path to inference script
        self.pass_runner_path = Path(__file__).parent.resolve() / "pass_runner.py"
        self.evaluation_runner_path = Path(__file__).parent.resolve() / "evaluation_runner.py"

    def _run_command(self, script_path: Path, config_jsons: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Run a script with the given config jsons and return the output json."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir).resolve()

            # write config jsons to files
            args = []
            for key, config_json in config_jsons.items():
                config_json_path = tmp_dir_path / f"{key}.json"
                with config_json_path.open("w") as f:
                    json.dump(config_json, f, indent=4)
                args.append(f"--{key} {config_json_path}")

            # command to run
            command = f"python {script_path} {' '.join(args)}"
            # add extra args
            for key, value in kwargs.items():
                if value is None:
                    continue
                command += f" --{key} {value}"
            # output path
            output_path = tmp_dir_path / "output.json"
            command += f" --output_path {output_path}"

            run_subprocess(command, env=self.environ, check=True)

            with output_path.open() as f:
                output = json.load(f)

        return output

    def run_pass(
        self,
        the_pass: "Pass",
        model_config: ModelConfig,
        data_root: str,
        output_model_path: str,
        point: Optional[Dict[str, Any]] = None,
    ) -> ModelConfig:
        """Run the pass on the model at a specific point in the search space."""
        config = the_pass.config_at_search_point(point or {})
        pass_config = the_pass.to_json(check_object=True)
        pass_config["config"].update(the_pass.serialize_config(config, check_object=True))
        config_jsons = {
            "model_config": model_config.to_json(check_object=True),
            "pass_config": pass_config,
        }
        output_model_json = self._run_command(
            self.pass_runner_path,
            config_jsons,
            data_root=data_root,
            tempdir=tempfile.tempdir,
            output_model_path=output_model_path,
        )
        return ModelConfig.parse_obj(output_model_json)

    def evaluate_model(
        self, model_config: ModelConfig, data_root: str, metrics: List[Metric], accelerator: AcceleratorSpec
    ) -> MetricResult:
        """Evaluate the model."""
        config_jsons = {
            "model_config": model_config.to_json(check_object=True),
            "metrics_config": [metric.to_json(check_object=True) for metric in metrics],
            "accelerator_config": accelerator.to_json(),
        }
        metric_results = self._run_command(
            self.evaluation_runner_path, config_jsons, data_root=data_root, tempdir=tempfile.tempdir
        )
        return MetricResult.parse_obj(metric_results)

    def get_supported_execution_providers(self) -> List[str]:
        """Get the available execution providers."""
        if self.available_eps:
            return self.available_eps

        with tempfile.TemporaryDirectory() as temp_dir:
            available_eps_path = Path(__file__).parent.resolve() / "available_eps.py"
            output_path = Path(temp_dir).resolve() / "available_eps.pb"
            run_subprocess(
                f"python {available_eps_path} --output_path {output_path}",
                env=self.environ,
                check=True,
            )
            with output_path.open("rb") as f:
                available_eps = pickle.load(f)
            self.available_eps = available_eps
            return available_eps

    def install_requirements(self, accelerator: AcceleratorSpec):
        """Install required packages."""
        # install common packages
        common_requirements_file = Path(__file__).parent.resolve() / "common_requirements.txt"
        run_subprocess(
            f"pip install --cache-dir {self.environ['TMPDIR']} -r {common_requirements_file}",
            env=self.environ,
            check=True,
        )

        # install onnxruntime package
        onnxruntime_package = get_package_name_from_ep(accelerator.execution_provider)[0]
        run_subprocess(
            f"pip install --cache-dir {self.environ['TMPDIR']} {onnxruntime_package}",
            env=self.environ,
            check=True,
        )

        # install user requirements
        if self.config.requirements_file:
            run_subprocess(
                f"pip install --cache-dir {self.environ['TMPDIR']} -r {self.config.requirements_file}",
                env=self.environ,
                check=True,
            )

    def remove(self):
        import shutil

        vitual_env_path = Path(self.config.python_environment_path).resolve().parent

        try:
            shutil.rmtree(vitual_env_path)
            logger.info("Virtual environment '%s' removed.", vitual_env_path)
        except FileNotFoundError:
            pass

        if platform.system() == "Linux":
            try:
                shutil.rmtree(self.environ["TMPDIR"])
                logger.info("Temporary directory '%s' removed.", self.environ["TMPDIR"])
            except FileNotFoundError:
                pass
