from pathlib import Path

import pytest

from olive.workflows.run.config import RunConfig
from olive.workflows.run.run import dependency_setup


def test_dependency_setup():
    user_script_config_file = Path(__file__).parent / "mock_data" / "user_script.json"
    run_config = RunConfig.parse_file(user_script_config_file)
    try:
        dependency_setup(run_config)
    except Exception as e:
        pytest.fail(f"setup failed with an exception: {str(e)}")
