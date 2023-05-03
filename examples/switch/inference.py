# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import argparse
import os
import pprint
import sys

import numpy
import onnxruntime


def _run_non_distributed(filepath):
    session = onnxruntime.InferenceSession(filepath, providers=["CPUExecutionProvider", "CUDAExecutionProvider"])
    input_name = session.get_inputs()[0].name
    data = numpy.full((40, 40), 5, dtype=numpy.int64)
    output = session.run(None, {input_name: data})[0]
    pprint.pprint(output)

    return 0


def _run_using_mpirun(filepath):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    local_rank = comm.Get_rank()
    filepath = filepath.format(local_rank)

    session = onnxruntime.InferenceSession(
        filepath,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        provider_options=[{"device_id": str(local_rank)}, {}],
    )
    input_name = session.get_inputs()[0].name
    data = numpy.full((40, 40), 10, dtype=numpy.int64)
    output = session.run(None, {input_name: data})[0]
    pprint.pprint(output)

    return 0


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
    data = numpy.full((40, 40), 10, dtype=numpy.int64)
    output = session.run(None, {input_name: data})[0]

    return output


def _run_using_mpipool(filepath, world_size):
    from mpi4py.futures import MPIPoolExecutor
    from mpi4py.MPI import COMM_WORLD

    args = [(rank, world_size, filepath.format(rank)) for rank in range(0, 2)]
    with MPIPoolExecutor(max_workers=world_size) as executor:
        results = executor.map(_mpipool_worker, args)
        executor.shutdown()

    COMM_WORLD.barrier()
    for result in results:
        pprint.pprint(result)

    return 0


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", dest="filepath", required=True, type=str, help="Onnx model file to load")
    parser.add_argument("--use_mpirun", action="store_true", default=False, help="Run using mpirun")
    parser.add_argument("--use_mpipool", action="store_true", default=False, help="Run using mpipool")
    parser.add_argument("--world_size", type=str, help="World size for distributed run")
    args = parser.parse_args()

    if args.use_mpipool:
        return _run_using_mpipool(args.filepath, args.world_size)
    elif args.use_mpirun:
        return _run_using_mpirun(args.filepath)
    else:
        return _run_non_distributed(args.filepath)


if __name__ == "__main__":
    sys.exit(_main())


# python3 examples/switch/inference.py \
#   --filepath examples/switch/model_4n_2l_8e.onnx
#
# python3 examples/switch/inference.py \
#   --filepath examples/switch/model_4n_2l_8e_{:02d}.onnx \
#   --use_mpipool \
#   --world_size 2
#
# mpirun -n 2 python3 examples/switch/inference.py \
#   --filepath examples/switch/model_4n_2l_8e_{:02d}.onnx \
#   --use_mpirun
