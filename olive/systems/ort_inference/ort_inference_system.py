# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Union

from olive.common.utils import run_subprocess
from olive.systems.olive_system import OliveSystem
from olive.systems.system_config import ORTInferenceTargetUserConfig


class ORTInferenceSystem(OliveSystem):
    system_type = "ORTInference"

    def __init__(
        self,
        python_environment_path: Union[Path, str] = None,
        environment_variables: Dict[str, str] = None,
        prepend_to_path: List[str] = None,
        accelerators: List[str] = None,
        hf_token: bool = None,
    ):
        super().__init__(accelerators=accelerators, olive_managed_env=False)
        self.config = ORTInferenceTargetUserConfig(
            python_environment_path=python_environment_path,
            environment_variables=environment_variables,
            prepend_to_path=prepend_to_path,
            accelerators=accelerators,
        )
        self.environ = deepcopy(os.environ)
        if self.config.environment_variables:
            self.environ.update(self.config.environment_variables)
        if self.config.prepend_to_path:
            self.environ["PATH"] = os.pathsep.join(self.config.prepend_to_path) + os.pathsep + self.environ["PATH"]
        if self.config.python_environment_path:
            self.environ["PATH"] = str(self.config.python_environment_path) + os.pathsep + self.environ["PATH"]

        # available eps. This will be populated the first time self.get_supported_execution_providers() is called.
        # used for caching the available eps
        self.available_eps = None

        # path to inference script
        parent_dir = Path(__file__).parent.resolve()
        self.pass_runner_path = parent_dir / "pass_runner.py"
        self.evaluation_runner_path = parent_dir / "evaluation_runner.py"
        self.available_eps_path = parent_dir / "available_eps.py"

    def get_supported_execution_providers(self) -> List[str]:
        """Get the available execution providers."""
        if self.available_eps:
            return self.available_eps

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir).resolve() / "available_eps.json"
            run_subprocess(
                f"python {self.available_eps_path} --output_path {output_path}",
                env=self.environ,
                check=True,
            )
            with output_path.open("r") as f:
                available_eps = json.load(f)
            self.available_eps = available_eps
            return available_eps
