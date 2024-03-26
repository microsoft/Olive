# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import torch


def load_model(model_name):
    from orchard.networks.transformer import Transformer

    model = Transformer.from_name(model_name)
    checkpoint_path = f"checkpoints/{model_name}/model.pth"
    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    checkpoint = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(checkpoint, assign=True)

    return model
