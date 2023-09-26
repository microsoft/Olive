# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy


def create_ddp_strategy(cluster, accelerator):
    return DDPStrategy(find_unused_parameters=True, cluster_environment=cluster, accelerator=accelerator)


def create_trainer(
    logger,
    callbacks=None,
    max_epochs=None,
    max_steps=None,
    val_check_interval=None,
    log_every_n_steps=50,
    precision=32,
    default_root_dir=None,
    **kwargs,
):
    return pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=max_epochs,
        max_steps=max_steps,
        val_check_interval=val_check_interval,
        log_every_n_steps=log_every_n_steps,
        precision=precision,
        default_root_dir=default_root_dir,
        **kwargs,
    )
