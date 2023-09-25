# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import argparse
import os
import pprint
import sys

import numpy as np
import onnxruntime


def _run_non_distributed(filepath):
    session = onnxruntime.InferenceSession(filepath, providers=["CPUExecutionProvider", "CUDAExecutionProvider"])
    input_name = session.get_inputs()[0].name
    data = np.full((40, 40), 2, dtype=np.int64)
    return session.run(None, {input_name: data})[0]


def _mpipool_worker(args):
    local_rank, world_size, filepath = args

    os.environ["OMPI_COMM_WORLD_RANK"] = str(local_rank)
    os.environ["OMPI_COMM_WORLD_SIZE"] = str(world_size)

    from mpi4py import MPI

    local_rank = MPI.COMM_WORLD.Get_rank()
    MPI.COMM_WORLD.barrier()

    print(f"rank: {local_rank}, filepath: {filepath}")

    session = onnxruntime.InferenceSession(
        filepath,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        provider_options=[{"device_id": str(local_rank)}, {}],
    )
    input_name = session.get_inputs()[0].name
    data = np.full((40, 40), 2, dtype=np.int64)
    return session.run(None, {input_name: data})[0]


def _run_using_mpipool(filepath, world_size):
    from mpi4py.futures import MPIPoolExecutor
    from mpi4py.MPI import COMM_WORLD

    args = [(rank, world_size, filepath.format(rank)) for rank in range(world_size)]
    with MPIPoolExecutor(max_workers=world_size) as executor:
        outputs = executor.map(_mpipool_worker, args)
        executor.shutdown()

    COMM_WORLD.barrier()

    return list(outputs)


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", dest="filepath", type=str, help="Onnx model file to load for non-distributed run")
    parser.add_argument(
        "--filename-pattern",
        dest="filename_pattern",
        type=str,
        help="Onnx model file name pattern to use for distributed run",
    )
    parser.add_argument("--world-size", dest="world_size", type=int, help="World size for distributed run")
    parser.add_argument(
        "--compare",
        dest="compare",
        action="store_true",
        default=False,
        help="Compare results from distributed session to non-distributed session",
    )
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug output")
    args = parser.parse_args()

    distributed_session_outputs = None
    if args.filename_pattern and (args.world_size > 1):
        distributed_session_outputs = _run_using_mpipool(args.filename_pattern, args.world_size)

    # For some unidentified reason, running a non-distributed session before a distributed one doesn't work.
    # The distributed session fails on MPI initialization.
    non_distributed_session_outputs = None
    if args.filepath:
        non_distributed_session_outputs = _run_non_distributed(args.filepath)

    if args.debug:
        pprint.pprint(
            {
                "non_distributed_session_outputs": non_distributed_session_outputs,
                "distributed_session_outputs": distributed_session_outputs,
            }
        )

    if args.compare and (non_distributed_session_outputs is not None) and (distributed_session_outputs is not None):
        atol = 1e-04
        results = {}
        for i in range(args.world_size):
            results[f"nondist vs. dist_{i:02d}"] = np.allclose(
                non_distributed_session_outputs, distributed_session_outputs[i], atol=atol
            )

            if i > 0:
                results[f"dist_00 vs. dist_{i:02d}"] = np.allclose(
                    distributed_session_outputs[0], distributed_session_outputs[i], atol=atol
                )

        if not np.all(list(results.values())):
            pprint.pprint(results)
            raise Exception("Inference tests failed!")

    return 0


if __name__ == "__main__":
    sys.exit(_main())


# python3 inference.py \
#   --filepath model.onnx
#
# python3 inference.py \
#   --filename-pattern model_{:02d}.onnx \
#   --world-size 2
#
# python3 inference.py \
#   --filepath model.onnx \
#   --filename-pattern model_{:02d}.onnx \
#   --world-size 2 \
#   --compare
