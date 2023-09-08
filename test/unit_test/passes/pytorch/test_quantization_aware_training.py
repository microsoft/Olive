# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from test.unit_test.utils import create_dataloader, get_pytorch_model

from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.pytorch import QuantizationAwareTraining


def test_quantization_aware_training_pass_default(tmpdir):
    # setup
    input_model = get_pytorch_model()
    config = {
        "train_dataloader_func": create_dataloader,
        "checkpoint_path": str(Path(tmpdir) / "checkpoint"),
    }

    p = create_pass_from_dict(QuantizationAwareTraining, config, disable_search=True)
    output_folder = str(Path(tmpdir) / "onnx")

    # execute
    p.run(input_model, None, output_folder)
