# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
from pathlib import Path
from typing import Any, Callable, Dict

from pydantic import validator

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import DistributedOnnxModel, PyTorchModel
from olive.passes import Pass
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.common import get_external_data_config
from olive.passes.onnx.conversion import DeviceSpecificOnnxConversion
from olive.passes.onnx.optimum_merging import OptimumMerging
from olive.passes.pass_config import PassConfigParam

logger = logging.getLogger(__name__)


class Sharding(Pass):
    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {
            "world_size": PassConfigParam(
                type_=int,
                default=2,
                required=True,
                description=("Number of GPU nodes to shard for. Must be greater than 1."),
            ),
            "target_opset": PassConfigParam(
                type_=int, default_value=17, description="The version of the default (ai.onnx) opset to target."
            ),
        }
        config.update(get_external_data_config())
        return config

    @staticmethod
    def _validate_world_size(v):
        if int(v) < 2:
            raise ValueError("world_size should be >= 2")

        return v

    @staticmethod
    def _validators() -> Dict[str, Callable]:
        return {"validate_distributor_config": validator("world_size", allow_reuse=True)(Sharding._validate_world_size)}

    @staticmethod
    def _generate_ranked_model(params):
        model_config, pass_config, accelerator_spec, world_size, local_rank, output_dirpath = params
        accelerator_spec = AcceleratorSpec(**accelerator_spec)

        import os

        os.environ["OMPI_COMM_WORLD_RANK"] = str(local_rank)
        os.environ["OMPI_COMM_WORLD_SIZE"] = str(world_size)
        os.environ["OMPI_COMM_WORLD_LOCAL_RANK"] = str(local_rank)
        os.environ["MIOPEN_FIND_MODE"] = "1"
        os.environ["OMPI_MCA_btl"] = "^openib"  # noqa: SIM112
        os.environ["OMPI_MCA_btl_openib_warn_no_device_params_found"] = "0"  # noqa: SIM112
        os.environ["OMPI_MCA_pml"] = "ob1"  # noqa: SIM112
        os.environ["OMPI_MCA_btl_tcp_if_include"] = "eth0"  # noqa: SIM112
        os.environ["OMPI_MCA_hwloc_base_binding_policy"] = "numa"  # noqa: SIM112
        os.environ["OMPI_MCA_ess"] = "^singleton"  # noqa: SIM112
        os.environ["OMPI_MCA_ess_base_vpid"] = "0"  # noqa: SIM112
        os.environ["OMPI_MCA_orte_tag_output"] = "1"  # noqa: SIM112
        os.environ["OMPI_MCA_pmix"] = "^s1,s2,cray,isolated"  # noqa: SIM112
        os.environ["OMPI_MCA_rmaps_ppr_n_pernode"] = "1"  # noqa: SIM112
        os.environ["NCCL_DEBUG"] = "WARN"

        import torch.distributed
        from mpi4py import MPI

        world_size = MPI.COMM_WORLD.Get_size()
        local_rank = MPI.COMM_WORLD.Get_rank()

        torch.distributed.init_process_group(
            "nccl", init_method="tcp://127.0.0.1:9876", world_size=world_size, rank=local_rank
        )

        # Create decoder & decoder_with_past models and export each as onnx model
        # Merge the generated model
        # NOTE: The intermediate PyTorchModel are never written to disk.

        input_model = PyTorchModel(**model_config)

        conversion_pass_config = {
            "target_opset": pass_config["target_opset"],
            "save_as_external_data": pass_config["save_as_external_data"],
            "all_tensors_to_one_file": pass_config["all_tensors_to_one_file"],
        }
        conversion_pass = create_pass_from_dict(
            DeviceSpecificOnnxConversion,
            config=conversion_pass_config,
            disable_search=True,
            accelerator_spec=accelerator_spec,
        )
        MPI.COMM_WORLD.Barrier()
        intermediate_model = conversion_pass.run(
            input_model, None, str(Path(output_dirpath) / f"ranked_models_{local_rank:02d}"), None
        )
        MPI.COMM_WORLD.Barrier()

        merging_pass_config = {
            "save_as_external_data": pass_config["save_as_external_data"],
            "all_tensors_to_one_file": pass_config["all_tensors_to_one_file"],
            "output_model_name": f"model_{local_rank:02d}.onnx",
        }
        merging_pass = create_pass_from_dict(
            OptimumMerging, config=merging_pass_config, disable_search=True, accelerator_spec=accelerator_spec
        )
        output_model = merging_pass.run(intermediate_model, None, output_dirpath, None)

        return output_model.model_path

    def _run_for_config(
        self, model: PyTorchModel, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> DistributedOnnxModel:
        from mpi4py.futures import MPIPoolExecutor

        model_config = model.to_json()["config"]
        pass_config = config
        acc_spec = self.accelerator_spec.to_json()
        world_size = config["world_size"]

        params = [
            (
                model_config,
                pass_config,
                acc_spec,
                world_size,
                rank,
                output_model_path,
            )
            for rank in range(world_size)
        ]

        with MPIPoolExecutor(max_workers=world_size) as executor:
            output_filepaths = executor.map(Sharding._generate_ranked_model, params)
            executor.shutdown()

        return DistributedOnnxModel(sorted(output_filepaths))
