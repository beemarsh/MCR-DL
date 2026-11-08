"""Check benchmark modes and buffer handling without a CUDA allocation."""

from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import mcr_dl
from mcr_dl.reduce_op import ReduceOp


@pytest.fixture
def benchmarks(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    import all_reduce
    import compare_torch
    import utils
    return SimpleNamespace(all_reduce=all_reduce, compare=compare_torch, utils=utils)


@pytest.mark.parametrize("in_place", [False, True])
def test_mode_flag(benchmarks, monkeypatch, tmp_path, in_place):
    flags = ["--in-place"] if in_place else []
    assert benchmarks.utils.benchmark_parser().parse_args(flags).in_place is in_place
    monkeypatch.setattr(sys, "argv", ["compare_torch.py", "--backend", "mpi",
                                      "--output", str(tmp_path / "report"), *flags])
    assert benchmarks.compare.parse_args().in_place is in_place


@pytest.mark.parametrize("in_place", [False, True])
@pytest.mark.parametrize("engine", ["mcr_dl", "torch"])
def test_timed_all_reduce_preserves_input_only_out_of_place(benchmarks, monkeypatch, in_place, engine):
    benchmark = benchmarks.all_reduce
    args = benchmarks.utils.benchmark_parser().parse_args([])
    args.in_place, args.dist = in_place, engine
    args.warmups, args.trials, args.async_op = 1, 2, True
    work = Mock()
    dist = Mock()
    dist.get_world_size.return_value = 2

    def reduce(tensor, **kwargs):
        tensor.mul_(2)
        return work

    def reduce_out(output, input, **kwargs):
        output.copy_(input * 2)
        return work

    dist.all_reduce.side_effect = reduce
    dist.all_reduce_out.side_effect = reduce_out
    monkeypatch.setattr(mcr_dl, "get_distributed_engine", lambda: dist)
    monkeypatch.setattr(benchmark, "sync_all", lambda: None)
    monkeypatch.setattr(benchmark, "print_rank_0", lambda message: None)
    event = Mock()
    event.elapsed_time.return_value = 1.0
    input = torch.arange(1, 5, dtype=torch.float32)
    original = input.clone()
    output = input if in_place else torch.empty_like(input)
    benchmark.timed_all_reduce(input, output, event, event, args)
    torch.testing.assert_close(output, original * (8 if in_place else 2))
    if not in_place:
        torch.testing.assert_close(input, original)
        assert work.wait.call_count == args.warmups + args.trials
    assert dist.all_reduce_out.call_count == (3 if engine == "mcr_dl" and not in_place else 0)


@pytest.mark.parametrize("in_place", [False, True])
@pytest.mark.parametrize("scan", [False, True])
def test_all_reduce_allocates_and_budgets_output(benchmarks, monkeypatch, in_place, scan):
    benchmark = benchmarks.all_reduce
    args = benchmarks.utils.benchmark_parser().parse_args([])
    args.in_place, args.scan, args.maxsize = in_place, scan, 3
    dist = Mock()
    dist.get_world_size.return_value = 2
    dist.get_rank.return_value = 1
    accelerator = Mock()
    accelerator.device_name.return_value = "cpu"
    accelerator.total_memory.return_value = 320
    monkeypatch.setattr(mcr_dl, "get_distributed_engine", lambda: dist)
    monkeypatch.setattr(benchmark, "get_accelerator", lambda: accelerator)
    monkeypatch.setattr(benchmarks.utils, "get_accelerator", lambda: accelerator)
    monkeypatch.setattr(benchmark, "sync_all", lambda: None)
    monkeypatch.setattr(benchmark, "print_header", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "Event", Mock())
    sizes = []

    def timed(input, output, *args):
        sizes.append(input.numel())
        assert (input.data_ptr() == output.data_ptr()) is in_place
        assert output.shape == input.shape

    monkeypatch.setattr(benchmark, "timed_all_reduce", timed)
    benchmark.run_all_reduce(0, args)
    assert sizes == ([4, 8] if scan else [48 if in_place else 24])


@pytest.mark.parametrize("in_place", [False, True])
@pytest.mark.parametrize("implementation", ["torch", "mpi", "nccl"])
def test_comparison_dispatch_and_input_validation(benchmarks, in_place, implementation):
    comparison = benchmarks.compare.Comparison(SimpleNamespace(
        backend=implementation, in_place=in_place, rtol=0, atol=0))
    comparison.torch = torch
    comparison.sum = ReduceOp.SUM
    comparison.gather = lambda checks: [checks]
    native = Mock()
    comparison.dist = comparison.native_group = comparison.native = native
    native.all_reduce.side_effect = lambda data, *args: data.mul_(2)
    native.all_reduce_out.side_effect = lambda output, data, *args: output.copy_(data * 2)
    input = torch.arange(1, 5, dtype=torch.float32)
    original = input.clone()
    output = input if in_place else torch.full_like(input, float("nan"))
    comparison.invoke("torch" if implementation == "torch" else "mcr", "all_reduce", input, output)
    torch.testing.assert_close(output, original * 2)
    assert native.all_reduce_out.call_count == (0 if in_place or implementation == "torch" else 1)
    buffers = {name: (input, output) for name in ("mcr", "torch")}
    checks = comparison.validate("all_reduce", buffers, original * 2, original)[0]
    assert checks["mcr"]["pass"] and checks["torch"]["pass"] and checks["pair"]["pass"]
    if not in_place:
        assert checks["mcr_input"]["pass"] and checks["torch_input"]["pass"]
        input.add_(1)
        checks = comparison.validate("all_reduce", buffers, original * 2, original)[0]
        assert not checks["mcr_input"]["pass"]
        assert not checks["torch_input"]["pass"]
