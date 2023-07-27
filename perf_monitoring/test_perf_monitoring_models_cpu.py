import os
from pathlib import Path

import pytest
from utils import extract_best_models, patch_config


@pytest.fixture(scope="module", autouse=True)
def setup():
    """setup any state specific to the execution of the given module."""
    cur_dir = Path(__file__).resolve().parent.parent
    example_dir = cur_dir / "perf_monitoring"
    os.chdir(example_dir)
    yield
    os.chdir(cur_dir)


def test_models():
    model_name = os.environ["TEST_MODEL"]
    olive_json = f"perf_models/cpu_models/{model_name}_cpu_config.json"
    print(olive_json)
    from olive.workflows import run as olive_run

    olive_config = patch_config(olive_json)
    footprint = olive_run(olive_config)
    extract_best_models(footprint, model_name)
