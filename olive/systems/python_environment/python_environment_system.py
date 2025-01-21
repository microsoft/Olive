# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
import os
import platform
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Union

from olive.common.constants import OS
from olive.common.utils import run_subprocess
from olive.evaluator.metric_result import MetricResult
from olive.model import ModelConfig
from olive.systems.common import AcceleratorConfig, SystemType
from olive.systems.olive_system import OliveSystem
from olive.systems.system_config import PythonEnvironmentTargetUserConfig
from olive.systems.utils import create_new_environ, get_package_name_from_ep, run_available_providers_runner

if TYPE_CHECKING:
    from olive.evaluator.olive_evaluator import OliveEvaluatorConfig
    from olive.hardware.accelerator import AcceleratorSpec
    from olive.passes.olive_pass import Pass


logger = logging.getLogger(__name__)


class PythonEnvironmentSystem(OliveSystem):
    system_type = SystemType.PythonEnvironment

    def __init__(
        self,
        python_environment_path: Union[Path, str] = None,
        environment_variables: Dict[str, str] = None,
        prepend_to_path: List[str] = None,
        accelerators: List[AcceleratorConfig] = None,
        olive_managed_env: bool = False,
        requirements_file: Union[Path, str] = None,
        hf_token: bool = None,
    ):
        if python_environment_path is None:
            raise ValueError("python_environment_path is required for PythonEnvironmentSystem.")

        super().__init__(accelerators=accelerators, hf_token=hf_token)
        self.config = PythonEnvironmentTargetUserConfig(**locals())
        self.environ = create_new_environ(
            python_environment_path=python_environment_path,
            environment_variables=environment_variables,
            prepend_to_path=prepend_to_path,
        )
        if olive_managed_env:
            if platform.system() == OS.LINUX:
                temp_dir = os.path.join(os.environ.get("HOME", ""), "tmp")
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)
                self.environ["TMPDIR"] = temp_dir
            else:
                self.environ["TMPDIR"] = tempfile.TemporaryDirectory().name  # pylint: disable=consider-using-with

        self.executable = shutil.which("python", path=self.environ["PATH"])
        # available eps. This will be populated the first time self.get_supported_execution_providers() is called.
        # used for caching the available eps
        self.available_eps = None

        # path to inference script
        parent_dir = Path(__file__).parent.resolve()
        self.pass_runner_path = parent_dir / "pass_runner.py"
        self.evaluation_runner_path = parent_dir / "evaluation_runner.py"

    def _run_command(self, script_path: Path, config_jsons: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Run a script with the given config jsons and return the output json."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir).resolve()

            # command to run
            command = [self.executable, str(script_path)]

            # write config jsons to files
            for key, config_json in config_jsons.items():
                config_json_path = tmp_dir_path / f"{key}.json"
                with config_json_path.open("w") as f:
                    json.dump(config_json, f, indent=4)
                command.extend([f"--{key}", str(config_json_path)])

            # add extra args
            for key, value in kwargs.items():
                if value is None:
                    continue
                command.extend([f"--{key}", str(value)])

            # output path
            output_path = tmp_dir_path / "output.json"
            command.extend(["--output_path", str(output_path)])

            # run the command
            _, stdout, _ = run_subprocess(command, env=self.environ, check=True)
            log_stdout(stdout)

            with output_path.open() as f:
                output = json.load(f)

        return output

    def run_pass(
        self,
        the_pass: "Pass",
        model_config: ModelConfig,
        output_model_path: str,
    ) -> ModelConfig:
        """Run the pass on the model."""
        pass_config = the_pass.to_json(check_object=True)
        config_jsons = {
            "model_config": model_config.to_json(check_object=True),
            "pass_config": pass_config,
        }
        output_model_json = self._run_command(
            self.pass_runner_path,
            config_jsons,
            tempdir=tempfile.tempdir,
            output_model_path=output_model_path,
        )
        return ModelConfig.parse_obj(output_model_json)

    def evaluate_model(
        self, model_config: ModelConfig, evaluator_config: "OliveEvaluatorConfig", accelerator: "AcceleratorSpec"
    ) -> MetricResult:
        """Evaluate the model."""
        config_jsons = {
            "model_config": model_config.to_json(check_object=True),
            "evaluator_config": evaluator_config.to_json(check_object=True),
            "accelerator_config": accelerator.to_json(),
        }
        metric_results = self._run_command(self.evaluation_runner_path, config_jsons, tempdir=tempfile.tempdir)
        return MetricResult.parse_obj(metric_results)

    def get_supported_execution_providers(self) -> List[str]:
        """Get the available execution providers."""
        if self.available_eps:
            return self.available_eps

        self.available_eps = run_available_providers_runner(self.environ)
        return self.available_eps

    def install_requirements(self, accelerator: "AcceleratorSpec"):
        """Install required packages."""
        # install common packages
        common_requirements_file = Path(__file__).parent.resolve() / "common_requirements.txt"
        packages = [
            f"-r {common_requirements_file}",
        ]

        if self.config.requirements_file:
            # install user requirements
            packages.append(f"-r {self.config.requirements_file}")

        # install onnxruntime package
        onnxruntime_package = get_package_name_from_ep(accelerator.execution_provider)[0]
        packages.append(onnxruntime_package)

        _, stdout, _ = run_subprocess(
            f"{self.executable} -m pip install --cache-dir {self.environ['TMPDIR']} {' '.join(packages)}",
            env=self.environ,
            check=True,
        )
        log_stdout(stdout)

        _, stdout, _ = run_subprocess(
            f"{self.executable} -m pip show {onnxruntime_package}", env=self.environ, check=True
        )
        log_stdout(stdout)

    def remove(self):
        vitual_env_path = Path(self.config.python_environment_path).resolve().parent

        try:
            shutil.rmtree(vitual_env_path)
            logger.info("Virtual environment '%s' removed.", vitual_env_path)
        except FileNotFoundError:
            pass

        if platform.system() == OS.LINUX:
            try:
                shutil.rmtree(self.environ["TMPDIR"])
                logger.info("Temporary directory '%s' removed.", self.environ["TMPDIR"])
            except FileNotFoundError:
                pass


def log_stdout(stdout: str):
    for line in stdout.splitlines():
        logger.debug("%s", line.strip())
