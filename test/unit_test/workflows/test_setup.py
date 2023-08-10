import json
from pathlib import Path

from olive.workflows.run.config import RunConfig
from olive.workflows.run.run import dependency_setup


def test_dependency_setup():
    user_script_config_file = Path(__file__).parent / "mock_data" / "user_script.json"
    with open(user_script_config_file, "r") as f:
        user_script_config = json.load(f)
    run_config = RunConfig.model_validate(user_script_config)
    dependency_setup(run_config)
