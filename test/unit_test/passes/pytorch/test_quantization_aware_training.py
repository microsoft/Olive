# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import tempfile
from pathlib import Path
from test.unit_test.utils import create_dataloader, get_pytorch_model

from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.pytorch import QuantizationAwareTraining
from olive.systems.local import LocalSystem


def test_quantization_aware_training_pass_default():
    # setup

    with tempfile.TemporaryDirectory() as tempdir:
        local_system = LocalSystem()
        input_model = get_pytorch_model()
        config = {
            "train_dataloader_func": create_dataloader,
            "checkpoint_path": str(Path(tempdir) / "checkpoint"),
        }

        p = create_pass_from_dict(QuantizationAwareTraining, config, disable_search=True)
        output_folder = str(Path(tempdir) / "onnx")

        # execute
        local_system.run_pass(p, input_model, output_folder)
