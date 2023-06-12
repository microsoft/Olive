# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from olive.snpe import SNPEProcessedDataLoader


def create_quant_dataloader(data_dir):
    return SNPEProcessedDataLoader(data_dir)


def create_eval_dataloader(data_dir, batch_size):
    return SNPEProcessedDataLoader(data_dir, annotations_file="labels.npy", batch_size=batch_size)


def post_process(output):
    return output["results"]["InceptionV3/Predictions/Reshape_1"].squeeze().argmax(axis=1)
