# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Based on the original implementation at
# https://github.com/IST-DASLab/sparsegpt
# https://arxiv.org/abs/2301.00774
# -------------------------------------------------------------------------
import logging
import math

import torch
import transformers

logger = logging.getLogger(__name__)

# ruff: noqa: N802, N806, RUF100

# model types supported by SparseGPT
supported_models = ("bloom", "gpt2", "gpt_neox", "llama", "opt")

# additional inputs to the layers for each model type
# all model types are expected to have "input_ids" and "attention_mask"
additional_inputs = {"bloom": ["alibi"], "gpt_neox": ["position_ids"]}


def get_layer_submodules(module, submodule_types=None, layer_name_filter=None, name=""):
    """Get the submodules of a module based on the submodule types."""
    submodule_types = submodule_types or [torch.nn.Conv2d, torch.nn.Linear, transformers.Conv1D]
    if type(module) in submodule_types:
        if layer_name_filter and not any(s in name for s in layer_name_filter):
            # skip this layer
            return {}
        return {name: module}

    submodules = {}
    for submodule_name_k, submodule in module.named_children():
        submodule_name = name + "." + submodule_name_k if name else submodule_name_k
        submodules.update(get_layer_submodules(submodule, submodule_types, layer_name_filter, submodule_name))
    return submodules


def validate_min_max_layers(min_layer, max_layer, num_layers):
    """Verify min_layer and max_layer are valid and return the valid range."""
    min_layer = min_layer or 0
    if min_layer < 0:
        # if user specified min_layer < 0, set min_layer to 0
        logger.warning("min_layer (%d) is less than 0. Setting to 0.", min_layer)
        min_layer = 0
    max_layer = max_layer or num_layers
    if max_layer > num_layers:
        # if user specified max_layer > number of layers, set max_layer to number of layers
        logger.warning(
            "max_layer (%d) is greater than number of layers (%(num_layers)d). Setting to %(num_layers)d.",
            max_layer,
            num_layers=num_layers,
        )
        max_layer = num_layers
        # don't need to worry about min_layer since if min_layer >= max_layer, the range will be empty
    return min_layer, max_layer


@torch.no_grad()
def catch_layer_inputs(model_wrapper, dataloader, device, num_samples=None):
    """Get the layers from model based on model type."""
    num_samples = num_samples or len(dataloader.dataset)
    first_batch = next(iter(dataloader))
    # sequence length
    seqlen = first_batch[0]["input_ids"].shape[1]
    # embedding dimension
    hidden_size = model_wrapper.hidden_size
    # data type
    dtype = next(iter(model_wrapper.model.parameters())).dtype

    # placeholder to save the layer inputs
    inputs = torch.zeros((num_samples, seqlen, hidden_size), dtype=dtype, device=device)
    cache = {"i": 0, "attention_mask": None}
    # additional inputs to the layers
    additional_input = additional_inputs.get(model_wrapper.model_type, [])
    for input_name in additional_input:
        cache[input_name] = None

    # get layers
    layers = model_wrapper.get_layers(False)

    class FirstLayer(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inputs, **kwargs):
            # handle batch dimension
            for batch in range(inputs.shape[0]):
                if cache["i"] >= num_samples:
                    break
                inputs[cache["i"]] = inputs[batch]
                cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask")
            for input_name in additional_input:
                cache[input_name] = kwargs.get(input_name)
            raise ValueError("Stop forward propagation")

    # put all modules until the first layer on the device
    for module in model_wrapper.get_embeds(False):
        module.to(device)

    # wrap the first layer
    layers[0] = FirstLayer(layers[0])

    # run the model on the data
    for data, _ in dataloader:
        input_ids = data["input_ids"].to(device)
        try:
            model_wrapper.model(input_ids)
        except ValueError:
            pass
        # stop if we have enough samples
        if cache["i"] >= num_samples:
            break

    # unwrap the first layer
    layers[0] = layers[0].module

    # put all modules until the first layer back on the CPU
    for module in model_wrapper.get_embeds(False):
        module.to("cpu")

    if "cuda" in str(device):
        torch.cuda.empty_cache()

    extras = {}
    for input_name in additional_input:
        extras[input_name] = cache[input_name]

    return inputs, cache["attention_mask"], extras


class SparseGPTModule:
    def __init__(self, layer):
        self.layer = layer
        self.device = self.layer.weight.device

        # get weights
        W = self.get_W()
        # store shape of W
        self.rows = W.shape[0]
        self.columns = W.shape[1]

        # Hessian
        self.H = torch.zeros((self.columns, self.columns), device=self.device)

        # number of samples
        self.num_samples = 0

    def get_W(self):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, torch.nn.Conv2d):
            # convert to 2D
            W = W.flatten(1)
        elif isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        return W.float()

    def add_batch(self, batch_input):
        # add batch dim if needed
        if batch_input.ndim == 2:
            batch_input = batch_input.unsqueeze(0)
        # get number of samples
        num_samples = batch_input.shape[0]
        # prepare input for linear layer
        if isinstance(self.layer, (torch.nn.Linear, transformers.Conv1D)):
            if batch_input.ndim == 3:
                # flatten the batch and sequence dimensions
                batch_input = batch_input.reshape(-1, batch_input.shape[-1])
            batch_input = batch_input.t()

        # renormalize H
        self.H *= self.num_samples / (self.num_samples + num_samples)
        # add new samples
        self.num_samples += num_samples
        batch_input = math.sqrt(2 / self.num_samples) * batch_input.float()
        self.H += batch_input.matmul(batch_input.t())

    def prune(self, mode, sparsity=None, n=None, m=None, blocksize=128, percdamp=0.01):
        # pylint: disable=not-callable
        W = self.get_W()
        H = self.H
        del self.H

        # set zero on the diagonal to 1
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        # set corresponding column of W to 0 to ignore it
        W[:, dead] = 0

        # dampen the Hessian
        assert 0 <= percdamp <= 1
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.device)
        H[diag, diag] += damp
        # use Cholesky decomposition to get the inverse
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        # placeholder for losses per row
        Losses = torch.zeros(self.rows, device=self.device)

        # loop over blocks of columns
        for start in range(0, self.columns, blocksize):
            end = min(start + blocksize, self.columns)
            num_cols = end - start

            # get submatrices
            W1 = W[:, start:end].clone()  # weights for the block
            Q1 = torch.zeros_like(W1)  # matrix to store new weights, updated column by column
            Err1 = torch.zeros_like(W1)  # block error
            Losses1 = torch.zeros_like(W1)  # losses
            Hinv1 = Hinv[start:end, start:end]  # Hessian inverse

            if mode == "unstructured":
                # chose the bottom sparsity% weights to prune (True)
                # lower magnitude = higher importance
                magnitude = W1**2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                threshold = torch.sort(magnitude.flatten())[0][int(magnitude.numel() * sparsity)]
                mask1 = magnitude <= threshold
            else:
                # placeholder for mask
                mask1 = torch.zeros_like(W1) == 1

            # loop over columns in the block
            for col in range(num_cols):
                w = W1[:, col]
                hinv = Hinv1[col, col]

                if mode == "structured" and col % m == 0:
                    # every mth column, set bottom n weights to True (prune)
                    magnitude = (
                        W1[:, col : (col + m)] ** 2  # noqa: E203, RUF100
                        / (torch.diag(Hinv1)[col : (col + m)].reshape((1, -1))) ** 2  # noqa: E203, RUF100
                    )
                    mask1.scatter_(1, col + torch.topk(magnitude, n, dim=1, largest=False)[1], True)

                # freeze weights in current column
                q = w.clone()
                q[mask1[:, col]] = 0
                Q1[:, col] = q

                # save losses
                Losses1[:, col] = (w - q) ** 2 / hinv**2

                # pruning error
                err1 = (w - q) / hinv
                # update weights for remaining columns in block
                W1[:, col:] -= err1.unsqueeze(1).matmul(Hinv1[col, col:].unsqueeze(0))
                # save error for lazy update
                Err1[:, col] = err1

            # copy new weights for current block
            W[:, start:end] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            # lazy update rest of the weights
            W[:, end:] -= Err1.matmul(Hinv[start:end, end:])

        if "cuda" in str(self.device):
            torch.cuda.synchronize()

        # set new weights
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        return torch.sum(Losses).item()

    def free(self):
        self.H = None
        if "cuda" in str(self.device):
            torch.cuda.empty_cache()
