"""CPU API tests; run with pytest tests/test_all_reduce_out.py."""

from datetime import timedelta
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import mcr_dl
import mcr_dl.comm as comm
from mcr_dl.mpi import MPIBackend
from mcr_dl.nccl import NCCLBackend
from mcr_dl.reduce_op import ReduceOp
from mcr_dl.torch import TorchBackend
from mcr_dl.utils.comms_logging import calc_bw_log
from mcr_dl.utils.dist import get_msg_size_from_args


def _torch_worker(rank, rendezvous):
    comm.cdb = TorchBackend(backend="gloo", init_method=rendezvous, rank=rank,
                            world_size=2, timeout=timedelta(seconds=30))
    try:
        # Different shapes with equal counts are valid contiguous buffers.
        for op, value in ((ReduceOp.SUM, 3), (ReduceOp.PRODUCT, 2),
                          (ReduceOp.MIN, 1), (ReduceOp.MAX, 2)):
            for async_op in (False, True):
                input_tensor = torch.full((2, 3), rank + 1, dtype=torch.float32)
                original = input_tensor.clone()
                output_tensor = torch.full((6,), -99.0)
                work = mcr_dl.all_reduce_out(output_tensor, input_tensor, op=op,
                                            async_op=async_op)
                if async_op:
                    assert work is not None
                    work.wait()
                else:
                    assert work is None
                torch.testing.assert_close(output_tensor, torch.full_like(output_tensor, value))
                torch.testing.assert_close(input_tensor, original)

                mcr_dl.all_reduce(input_tensor, op=op)
                torch.testing.assert_close(input_tensor, torch.full_like(input_tensor, value))

        group = dist.new_group([0])
        input_tensor = torch.ones(3)
        output_tensor = torch.full((3,), -1.0)
        mcr_dl.all_reduce_out(output_tensor, input_tensor, group=group)
        torch.testing.assert_close(output_tensor, torch.full_like(output_tensor, 1 if rank == 0 else -1))
    finally:
        dist.destroy_process_group()


def test_torch_all_reduce_out(tmp_path):
    mp.spawn(_torch_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True)


@pytest.mark.parametrize("invalid", ["size", "dtype", "noncontiguous", "alias", "overlap"])
def test_invalid_output_does_not_modify_input(invalid):
    backend = TorchBackend.__new__(TorchBackend)
    input_tensor = torch.arange(6, dtype=torch.float32)
    if invalid == "size":
        output_tensor = torch.empty(5)
    elif invalid == "dtype":
        output_tensor = torch.empty(6, dtype=torch.int32)
    elif invalid == "noncontiguous":
        output_tensor = torch.empty(3, 2).t()
    elif invalid == "alias":
        output_tensor = input_tensor
    else:
        storage = torch.arange(7, dtype=torch.float32)
        input_tensor, output_tensor = storage[:-1], storage[1:]
    original = input_tensor.clone()
    with pytest.raises(ValueError):
        backend.all_reduce_out(output_tensor, input_tensor)
    torch.testing.assert_close(input_tensor, original)


@pytest.mark.parametrize("backend_type", [MPIBackend, NCCLBackend])
def test_public_native_dispatch(monkeypatch, backend_type):
    backend = backend_type.__new__(backend_type)
    native = Mock()
    if backend_type is MPIBackend:
        backend.initialized = True
        backend.world_group = native
        backend._group_ranks = {native: (0, 1)}
    else:
        backend.nccl_comm_op = native
    monkeypatch.setattr(comm, "cdb", backend)
    input_tensor = torch.ones(6)
    output_tensor = torch.empty_like(input_tensor)
    mcr_dl.all_reduce_out(output_tensor, input_tensor, ReduceOp.MAX, group=native)
    if backend_type is MPIBackend:
        native.all_reduce_out.assert_called_once_with(output_tensor, input_tensor, 3, False, stream=None)
        native.reset_mock()
        stream = Mock(spec=torch.cuda.Stream)
        stream.cuda_stream = 123
        stream.device = torch.device("cuda", 0)
        mcr_dl.all_reduce_out(output_tensor, input_tensor, stream=stream)
        native.all_reduce_out.assert_called_once_with(output_tensor, input_tensor, 0, False,
                                                     stream=(123, 0))
        with pytest.raises(NotImplementedError, match="async_op"):
            mcr_dl.all_reduce_out(output_tensor, input_tensor, async_op=True)
    else:
        native.all_reduce_out.assert_called_once_with(output_tensor, input_tensor, ReduceOp.MAX,
                                                     False, native, False)


def test_all_reduce_out_logging(monkeypatch):
    # Exercise positional, keyword and mixed calls without needing CUDA timers.
    def operation(output_tensor, input_tensor):
        pass

    input_tensor = torch.ones(6)
    output_tensor = torch.empty_like(input_tensor)
    for args, kwargs in (((output_tensor, input_tensor), {}),
                         ((), dict(output_tensor=output_tensor, input_tensor=input_tensor)),
                         ((output_tensor,), dict(input_tensor=input_tensor))):
        assert get_msg_size_from_args(operation, *args, **kwargs) == 24
    monkeypatch.setattr(comm, "get_world_size", lambda: 2)
    assert calc_bw_log("all_reduce_out", 24, 1) == calc_bw_log("all_reduce", 24, 1)
