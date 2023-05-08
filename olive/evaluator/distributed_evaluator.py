from olive.model import DistributedOnnxModel
import os
import onnxruntime
import pprint
import numpy
import time


def _mpipool_worker(args):
    model, local_rank, world_size = args
    shape = (1, 128)

    os.environ["OMPI_COMM_WORLD_RANK"] = str(local_rank)
    os.environ["OMPI_COMM_WORLD_SIZE"] = str(world_size)

    from mpi4py import MPI

    local_rank = MPI.COMM_WORLD.Get_rank()
    MPI.COMM_WORLD.barrier()

    session = model.prepare_session(rank=local_rank)
    input_name = session.get_inputs()[0].name
    data = numpy.full(shape, 10.0, dtype=numpy.int64)

    input_name_2 = session.get_inputs()[1].name
    data_2 = numpy.full(shape, 10.0, dtype=numpy.float32)

    start_time = time.time()
    output = session.run(None, {input_name: data, input_name_2: data_2})[0]

    return time.time() - start_time


def eval_onnx_distributed_latency(model: DistributedOnnxModel):
    from mpi4py.futures import MPIPoolExecutor
    from mpi4py.MPI import COMM_WORLD

    args = [(model, rank, 2) for rank in range(0, 2)]
    with MPIPoolExecutor(max_workers=2) as executor:
        results = executor.map(_mpipool_worker, args)
        executor.shutdown()

    COMM_WORLD.barrier()
    results = list(results)
    for result in results:
        pprint.pprint(result)

    return results[0]
