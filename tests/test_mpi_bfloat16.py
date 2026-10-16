"""BF16 regression tests for externally launched MPI ranks.

Run this file under srun or mpirun with at least two ranks. Each launched
process is an MPI participant; these tests do not spawn Python workers.
The MCR-DL extension must be built against an MPI library with native BF16
support. CUDA cases additionally require CUDA-aware MPI.
"""

import os

import pytest
import torch


@pytest.fixture(scope="module")
def mpi_backend():
    # Bind the device before initializing CUDA-aware MPI, including when
    # Slurm exposes all of the node's GPUs to every rank.
    if torch.cuda.is_available():
        local_rank = next(
            (
                int(os.environ[name])
                for name in (
                    "LOCAL_RANK",
                    "SLURM_LOCALID",
                    "OMPI_COMM_WORLD_LOCAL_RANK",
                    "MV2_COMM_WORLD_LOCAL_RANK",
                    "MPI_LOCALRANKID",
                )
                if name in os.environ
            ),
            0,
        )
        torch.cuda.set_device(0 if torch.cuda.device_count() == 1 else local_rank)

    from mcr_dl.mpi import MPIBackend

    backend = MPIBackend()
    try:
        if backend.get_world_size() < 2:
            pytest.skip("Launch with srun or mpirun and at least two MPI ranks")
        assert not torch.distributed.is_initialized(), "Expected direct MCR-DL MPI"
        yield backend
    finally:
        backend.mpi_comm_op.finalize()


@pytest.fixture(params=["cpu", "cuda"])
def device(request, mpi_backend):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    return torch.device(request.param)


@pytest.mark.parametrize("numel", [1, 257])
def test_bfloat16_all_reduce(mpi_backend, device, numel):
    # Include negative and fractional values to detect byte/integer reduction.
    pattern = (torch.arange(numel, dtype=torch.float32) % 7 - 3) / 4
    rank = mpi_backend.get_rank()
    size = mpi_backend.get_world_size()
    tensor = (pattern * (rank + 1)).to(device=device, dtype=torch.bfloat16)
    expected = (pattern * (size * (size + 1) // 2)).to(torch.bfloat16)

    # The current MCR-DL MPI wrapper does not order CUDA producer work.
    # Make inputs ready explicitly so this test isolates datatype support.
    if tensor.is_cuda:
        torch.cuda.synchronize()
    mpi_backend.all_reduce(tensor)
    if tensor.is_cuda:
        torch.cuda.synchronize()

    torch.testing.assert_close(tensor.cpu(), expected)


@pytest.mark.parametrize("numel", [1, 257])
def test_bfloat16_broadcast(mpi_backend, device, numel):
    # An odd element count also catches treating two-byte BF16 as MPI_BYTE.
    expected = ((torch.arange(numel, dtype=torch.float32) % 7 - 3) / 4).to(
        torch.bfloat16
    )
    tensor = torch.zeros(numel, dtype=torch.bfloat16, device=device)
    if mpi_backend.get_rank() == 0:
        tensor.copy_(expected)

    if tensor.is_cuda:
        torch.cuda.synchronize()
    mpi_backend.broadcast(tensor, src=0)
    if tensor.is_cuda:
        torch.cuda.synchronize()

    torch.testing.assert_close(tensor.cpu(), expected, rtol=0, atol=0)
