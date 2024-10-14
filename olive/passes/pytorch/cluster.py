# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import os
from abc import ABC, abstractmethod
from typing import Dict

from pytorch_lightning.plugins.environments import ClusterEnvironment

logger = logging.getLogger(__name__)


class BaseClusterEnvironment(ClusterEnvironment, ABC):
    def __init__(self, master_port: int = 6105):
        self.master_port = master_port
        self._is_initialized = False

        # Will get actual values in init_process_group.
        # We need to use these instead of using os.environ directly
        # because these get called in pytorch plugins after the teardown.
        self._main_addr = ""
        self._world_size = -1
        self._local_rank = -1
        self._node_rank = -1
        self._global_rank = -1

        # needed for teardown
        self._original_env_vars: Dict = {}
        self._overrides: Dict = {}

    def init_process_group(self) -> None:
        import torch

        if not self._is_initialized:
            overrides = self._environment_variable_overrides(self.master_port)
            self._override_environment_variables(overrides)
            self._set_internal_variables_from_environment()

            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend="nccl")
            self._is_initialized = True

    @abstractmethod
    def _environment_variable_overrides(self, port) -> Dict[str, str]:
        pass

    def _override_environment_variables(self, overrides: Dict[str, str]) -> None:
        self._overrides = overrides
        for variable, value in overrides.items():
            self._store_value(variable)
            os.environ[variable] = value

    def _set_internal_variables_from_environment(self) -> None:
        self._main_addr = os.environ["MASTER_ADDR"]
        self._world_size = int(os.environ["WORLD_SIZE"])
        self._local_rank = int(os.environ["LOCAL_RANK"])
        self._node_rank = int(os.environ["NODE_RANK"])
        self._global_rank = int(os.environ["RANK"])

    def _store_value(self, variable):
        original_value = os.environ.get(variable, None)
        self._original_env_vars[variable] = original_value

    def is_initialized(self) -> bool:
        return self._is_initialized

    @abstractmethod
    def gpus(self) -> int:
        pass

    @abstractmethod
    def num_nodes(self) -> int:
        pass

    @property
    def creates_processes_externally(self) -> bool:
        return True

    @property
    def main_address(self) -> str:
        return self._main_addr

    @property
    def main_port(self) -> int:
        return self.master_port

    @staticmethod
    def detect() -> bool:
        return True

    def world_size(self) -> int:
        return self._world_size

    def set_world_size(self, size: int) -> None:
        pass

    def global_rank(self) -> int:
        return self._global_rank

    def set_global_rank(self, rank: int) -> None:
        pass

    def local_rank(self) -> int:
        return self._local_rank

    def node_rank(self) -> int:
        return self._node_rank

    def teardown(self) -> None:
        """Clean up any state set after execution finishes."""
        logger.info("Cleaning up environment variables")
        logger.info("self._original_env_vars: %s", self._original_env_vars)
        for variable, original_value in self._original_env_vars.items():
            if original_value is None and variable in os.environ:
                # delete any new variables we might have created
                os.environ.pop(variable)
            elif original_value is not None and variable in os.environ:
                # if no need to delete, just set back the original value
                os.environ[variable] = original_value
            else:
                logger.info("original_value: %s", original_value)
                logger.info("variable: %s", variable)
                if variable in os.environ:
                    logger.info("os.environ[variable]: %s", os.environ[variable])


class AzureMLPerProcessCluster(BaseClusterEnvironment):
    def _environment_variable_overrides(self, port: int = 6105) -> Dict[str, str]:
        """Set the MPI environment variables required for multinode distributed training.

        Args:
            port (int): Used to set MASTER_PORT environment variable if its not present.

        """
        overrides = {}

        overrides["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
        overrides["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]

        single_node = int(os.environ["OMPI_COMM_WORLD_LOCAL_SIZE"]) == int(overrides["WORLD_SIZE"])
        if not single_node:
            master_node_params = os.environ["AZ_BATCH_MASTER_NODE"].split(":")
            overrides["MASTER_ADDR"] = master_node_params[0]

            # Do not overwrite master port with that defined in AZ_BATCH_MASTER_NODE
            if "MASTER_PORT" not in os.environ:
                overrides["MASTER_PORT"] = str(port)
        else:
            overrides["MASTER_ADDR"] = os.environ["AZ_BATCHAI_MPI_MASTER_NODE"]
            overrides["MASTER_PORT"] = "54965"

        overrides["NCCL_SOCKET_IFNAME"] = "^docker0,lo"

        # set local rank, compute node rank
        overrides["LOCAL_RANK"] = os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "-1")
        world_size = os.environ.get("OMPI_COMM_WORLD_SIZE", "1")
        if int(world_size) == 1:
            overrides["NODE_RANK"] = "0"
        else:
            local_size = int(os.environ["OMPI_COMM_WORLD_LOCAL_SIZE"])
            node_rank = (int(overrides["RANK"]) - int(overrides["LOCAL_RANK"])) // local_size
            overrides["NODE_RANK"] = str(node_rank)

        return overrides

    def gpus(self) -> int:
        world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))
        if world_size == 1:
            return 1
        else:
            return int(os.environ["OMPI_COMM_WORLD_LOCAL_SIZE"])

    def num_nodes(self) -> int:
        world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))
        if world_size == 1:
            return 1
        else:
            return world_size // self.gpus()


def create_cluster():
    try:
        cluster_environment = AzureMLPerProcessCluster()
        cluster_environment.init_process_group()
    except KeyError:
        cluster_environment = None
        logger.warning("Couldn't initialize cluster environment - this is expected if not running in AzureML")
    return cluster_environment


def get_rank() -> int:
    import torch

    if not torch.cuda.is_available():
        return 0
    if not torch.distributed.is_initialized():
        return 0
    else:
        return torch.distributed.get_rank()


def is_master_proc():
    return get_rank() == 0


def barrier():
    import torch

    if torch.cuda.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
