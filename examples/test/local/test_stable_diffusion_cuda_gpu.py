# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
import sys
from pathlib import Path

import pytest

from examples.test.utils import get_example_dir


@pytest.fixture(scope="module", autouse=True)
def setup():
    """Setups any state specific to the execution of the given module."""
    example_dir = get_example_dir("stable_diffusion")
    os.chdir(example_dir)
    sys.path.insert(0, example_dir)

    yield
    sys.path.remove(example_dir)


@pytest.mark.skip(reason="Disable failing tests")
@pytest.mark.parametrize("model_id", [None, "stabilityai/sd-turbo", "sayakpaul/sd-model-finetuned-lora-t4"])
def test_stable_diffusion(model_id):
    # clean previous artifacts
    for image_file in Path().glob("result_*.png"):
        image_file.unlink()

    from stable_diffusion import main as stable_diffusion_main

    # common arguments
    cmd_args = ["--provider", "cuda"]
    if model_id is not None:
        cmd_args.extend(["--model_id", model_id])

    # run the optimization
    stable_diffusion_main([*cmd_args, "--optimize"])

    # test inference
    num_images = 2
    stable_diffusion_main([*cmd_args, "--num_images", str(num_images)])

    # check the results
    for i in range(num_images):
        assert Path(f"result_{i}.png").exists()
