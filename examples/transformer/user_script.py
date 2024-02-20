# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import sys
from pathlib import Path

import torch

wd = Path(__file__).parent.parent.parent.parent.resolve() / "orchard"
sys.path.append(str(wd))


def load_model(model_name):
    from orchard.networks.transformer import Transformer

    with torch.device("meta"):
        model = Transformer.from_name(model_name)

    checkpoint_path = f"checkpoints/{model_name}/model.pth"
    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    model.load_state_dict(checkpoint, assign=True)

    return model
