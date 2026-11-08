"""Validate and time the legacy C++ bindings against the matching torch backend.

Launch with srun/mpirun, one process per GPU. See docs/benchmark_comparison.md.
This measures Python-to-C++ completed-call latency, including device completion.
"""

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time
import traceback


REPO = Path(__file__).resolve().parents[1]
OPS = ("all_reduce", "broadcast", "all_to_all_single")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=("mpi", "nccl"))
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32", "float64"))
    parser.add_argument("--ops", nargs="+", choices=OPS, default=list(OPS))
    parser.add_argument("--in-place", action="store_true",
                        help="Run all_reduce in place (default: use a separate output tensor)")
    parser.add_argument("--numels", nargs="+", type=int,
                        default=[1, 257, 1024, 16384, 262144, 4194304],
                        help="Elements per rank; for all_to_all_single, elements per peer")
    parser.add_argument("--warmups", type=int, default=50)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--root", type=int, default=0, help="Broadcast source rank")
    parser.add_argument("--rtol", type=float, default=None, help="SUM validation tolerance")
    parser.add_argument("--atol", type=float, default=None, help="SUM validation tolerance")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--output", type=Path, required=True, help="Output prefix, without extension")
    args = parser.parse_args()
    if min(args.numels) < 1 or args.warmups < 0 or args.trials < 1 or args.timeout_seconds < 1:
        parser.error("Sizes, trials and timeout must be positive; warmups must be nonnegative")
    for tolerance in (args.rtol, args.atol):
        if tolerance is not None and (not math.isfinite(tolerance) or tolerance < 0):
            parser.error("Tolerances must be finite and nonnegative")
    defaults = {"bfloat16": (0.016, 1e-5), "float32": (1.3e-6, 1e-5), "float64": (1e-7, 1e-7)}
    args.rtol = defaults[args.dtype][0] if args.rtol is None else args.rtol
    args.atol = defaults[args.dtype][1] if args.atol is None else args.atol
    return args


def env_rank(names, default=0):
    return next((int(os.environ[name]) for name in names if name in os.environ), default)


def percentile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    low, high = math.floor(index), math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)


def latency_summary(per_rank):
    # Each sample represents the slowest participating rank for that iteration.
    samples = [max(sample) for sample in zip(*per_rank)]
    return {"mean_us": statistics.mean(samples), "p50_us": statistics.median(samples),
            "p95_us": percentile(samples, 0.95), "min_us": min(samples)}


class Comparison:
    def __init__(self, args):
        self.args = args
        self.mpi = self.native = None
        self.rank = env_rank(("SLURM_PROCID", "OMPI_COMM_WORLD_RANK", "PMI_RANK", "RANK"))
        self.rows = []
        self.details = []
        self.metadata = {}

    def initialize(self):
        import torch
        import torch.distributed as dist

        self.torch, self.dist = torch, dist
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU is required. Run inside a Perlmutter GPU allocation.")
        if not dist.is_available() or not getattr(dist, f"is_{self.args.backend}_available")():
            raise RuntimeError(f"This PyTorch build lacks the {self.args.backend} backend. No fallback is used.")
        local_rank = env_rank(("LOCAL_RANK", "SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK",
                               "MV2_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID"))
        self.device = 0 if torch.cuda.device_count() == 1 else local_rank
        torch.cuda.set_device(self.device)
        sys.path.insert(0, str(REPO))
        from mcr_dl.ops.op_builder import MPICommBuilder, NCCLCommBuilder
        from mcr_dl.reduce_op import ReduceOp

        self.sum = ReduceOp.SUM
        # Let torch own MPI lifetime when using ProcessGroupMPI. Both extensions
        # must link the same MPI implementation as this PyTorch installation.
        if self.args.backend == "mpi":
            dist.init_process_group("mpi", timeout=timedelta(seconds=self.args.timeout_seconds))
        self.mpi = MPICommBuilder().load(verbose=False)
        self.mpi.initialize()
        self.rank, self.size = self.mpi.get_rank(0), self.mpi.get_world_size(0)
        if self.size < 2:
            raise RuntimeError("Launch at least two MPI ranks with srun or mpirun, not torchrun.")
        if not 0 <= self.args.root < self.size:
            raise ValueError("--root must be a participating rank")
        if self.args.backend == "nccl":
            if "MASTER_ADDR" not in os.environ or "MASTER_PORT" not in os.environ:
                raise RuntimeError("Set MASTER_ADDR to the first compute node and MASTER_PORT to a free port")
            # nccl.cpp selects rank % visible GPUs internally. Refuse a mismatch.
            if self.rank % torch.cuda.device_count() != self.device:
                raise RuntimeError("Use one visible GPU per rank (--gpus-per-task=1 --gpu-bind=single:1)")
            dist.init_process_group("nccl", rank=self.rank, world_size=self.size,
                                    timeout=timedelta(seconds=self.args.timeout_seconds))
            self.native = NCCLCommBuilder().load(verbose=False)
            self.native.initialize(self.rank, self.size)
        else:
            self.native = self.mpi
            self.native_group = self.mpi.NativeGroup([])
        if (dist.get_rank(), dist.get_world_size()) != (self.rank, self.size):
            raise RuntimeError("torch.distributed and the native extension disagree on MPI rank/world size")
        torch.cuda.set_device(self.device)
        self.metadata = self.get_metadata()
        self.write_results()

    def gather(self, value):
        values = [None] * self.size
        self.dist.all_gather_object(values, value)
        return values

    def synchronize(self):
        self.torch.cuda.synchronize(self.device)

    def get_metadata(self):
        torch = self.torch
        maps_file = Path("/proc/self/maps")
        libraries = sorted({line.split()[-1] for line in maps_file.read_text().splitlines()
                            if any(name in line for name in ("libmpi", "libnccl", "libcudart"))})
        rank_info = self.gather({"rank": self.rank, "host": socket.gethostname(),
                                 "device_index": self.device,
                                 "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                 "gpu": torch.cuda.get_device_name(self.device),
                                 "mpi_extension": self.mpi.__file__,
                                 "native_extension": self.native.__file__,
                                 "loaded_libraries": libraries})
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                  capture_output=True, text=True, check=False).stdout.strip()
        sources = {name: hashlib.sha256((REPO / "mcr_dl/ops/csrc/comm" / name).read_bytes()).hexdigest()
                   for name in ("mpi.cpp", "nccl.cpp")}
        return {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "git_commit": revision,
                "source_sha256": sources, "torch": torch.__version__, "cuda": torch.version.cuda,
                "torch_nccl_version": torch.cuda.nccl.version() if self.dist.is_nccl_available() else None,
                "backend": self.args.backend, "dtype": self.args.dtype, "world_size": self.size,
                "all_reduce_in_place": self.args.in_place,
                "torch_all_reduce_includes_copy": not self.args.in_place,
                "warmups": self.args.warmups, "trials": self.args.trials, "root": self.args.root,
                "sum_rtol": self.args.rtol, "sum_atol": self.args.atol,
                "timing": "perf_counter_ns around one Python binding call plus CUDA device completion; "
                          "reset, producer synchronization and MPI barrier excluded; slowest rank per trial",
                "rank_info": rank_info,
                "environment": {key: os.environ[key] for key in (
                    "SLURM_JOB_ID", "SLURM_JOB_NUM_NODES", "SLURM_NTASKS", "OMP_NUM_THREADS",
                    "NCCL_ALGO", "NCCL_PROTO", "NCCL_DEBUG", "NCCL_NET", "NCCL_SOCKET_IFNAME",
                    "MPICH_GPU_SUPPORT_ENABLED", "MV2_USE_CUDA") if key in os.environ}}

    def values(self, count, rank, offset=0):
        # Exact eighths, including negatives, with rank and position dependence.
        torch = self.torch
        indices = torch.arange(offset, offset + count, dtype=torch.int64)
        return (((indices + rank * 3) % 17 - 8).double() * (rank + 1) / 8).to(
            getattr(torch, self.args.dtype))

    def expected(self, op, numel):
        torch = self.torch
        if op == "all_reduce":
            expected = torch.zeros(numel, dtype=torch.float64)
            for rank in range(self.size):
                expected.add_(self.values(numel, rank).double())
            return expected.to(getattr(torch, self.args.dtype))
        if op == "broadcast":
            return self.values(numel, self.args.root)
        return torch.cat([self.values(numel, rank, self.rank * numel) for rank in range(self.size)])

    def invoke(self, implementation, op, data, output):
        if implementation == "torch":
            if op == "all_reduce":
                if not self.args.in_place:
                    # Include the copy in the torch baseline's measured latency.
                    output.copy_(data)
                self.dist.all_reduce(data if self.args.in_place else output)
            elif op == "broadcast":
                self.dist.broadcast(data, src=self.args.root)
            else:
                self.dist.all_to_all_single(output, data)
        elif self.args.backend == "mpi":
            if op == "all_reduce":
                if self.args.in_place:
                    self.native_group.all_reduce(data, self.sum.value)
                else:
                    self.native_group.all_reduce_out(output, data, self.sum.value)
            elif op == "broadcast":
                self.native_group.broadcast(data, self.args.root)
            else:
                self.native_group.all_to_all(output, data)
        elif op == "all_reduce":
            if self.args.in_place:
                self.native.all_reduce(data, self.sum, False, None, False)
            else:
                self.native.all_reduce_out(output, data, self.sum, False, None, False)
        elif op == "broadcast":
            self.native.broadcast(data, self.args.root, False, None, False)
        else:
            self.native.all_to_all_single(output, data, False, None, False)

    def check(self, actual, expected, op):
        torch = self.torch
        actual, expected = actual.cpu().double(), expected.cpu().double()
        rtol, atol = (self.args.rtol, self.args.atol) if op == "all_reduce" else (0, 0)
        finite = torch.isfinite(actual) & torch.isfinite(expected)
        close = torch.isclose(actual, expected, rtol=rtol, atol=atol) & finite
        errors = (actual - expected).abs()
        bad = (~close).nonzero().flatten()
        first_bad = None
        if bad.numel():
            index = int(bad[0])
            first_bad = {"index": index, "actual": str(float(actual[index])),
                         "expected": str(float(expected[index]))}
        return {"pass": bool(close.all()), "mismatched_elements": int(bad.numel()),
                "nonfinite_elements": int((~finite).sum()),
                "max_abs_error": float(errors[finite].max()) if bool(finite.any()) else None,
                "first_mismatch": first_bad}

    def validate(self, op, buffers, expected, template):
        checks = {"rank": self.rank,
                  "mcr": self.check(buffers["mcr"][1], expected, op),
                  "torch": self.check(buffers["torch"][1], expected, op),
                  "pair": self.check(buffers["mcr"][1], buffers["torch"][1], op)}
        if op == "all_reduce" and not self.args.in_place:
            for name in ("mcr", "torch"):
                checks[f"{name}_input"] = self.check(buffers[name][0], template, "input")
        return self.gather(checks)

    def run_case(self, op, numel):
        torch = self.torch
        count = numel * self.size if op == "all_to_all_single" else numel
        template = self.values(count, self.rank).to(device=f"cuda:{self.device}")
        expected = self.expected(op, numel)
        separate_output = op == "all_to_all_single" or (op == "all_reduce" and not self.args.in_place)
        validation_keys = ("mcr", "torch", "pair")
        if op == "all_reduce" and not self.args.in_place:
            validation_keys += ("mcr_input", "torch_input")
        buffers = {}
        for name in ("mcr", "torch"):
            data = torch.empty_like(template)
            buffers[name] = (data, torch.empty_like(data) if separate_output else data)

        def execute(name):
            data, output = buffers[name]
            data.copy_(template)
            if op == "broadcast" and self.rank != self.args.root:
                # A partial BF16 write must fail even for a one-element message.
                data.fill_(float("nan"))
            if separate_output:
                output.fill_(float("nan"))
            self.synchronize()
            self.mpi.barrier()
            started = time.perf_counter_ns()
            self.invoke(name, op, data, output)
            self.synchronize()
            return (time.perf_counter_ns() - started) / 1000

        for name in buffers:
            execute(name)
        validation = self.validate(op, buffers, expected, template)
        passed = all(item[key]["pass"] for item in validation for key in validation_keys)
        row = {"backend": self.args.backend, "dtype": self.args.dtype, "world_size": self.size,
               "op": op, "numel_per_rank": count,
               "in_place": self.args.in_place if op == "all_reduce" else None,
               "numel_per_peer": numel if op == "all_to_all_single" else None,
               "input_bytes_per_rank": count * template.element_size(), "status": "FAIL",
               "speedup": None}
        for name in ("mcr", "torch"):
            for metric in ("mean_us", "p50_us", "p95_us", "min_us"):
                row[f"{name}_{metric}"] = None
        samples = {"mcr": [], "torch": []}
        gathered_samples = None
        if passed:
            for iteration in range(self.args.warmups + self.args.trials):
                # Alternate which implementation runs first to limit ordering bias.
                order = ("mcr", "torch") if iteration % 2 == 0 else ("torch", "mcr")
                for name in order:
                    elapsed = execute(name)
                    if iteration >= self.args.warmups:
                        samples[name].append(elapsed)
            final_validation = self.validate(op, buffers, expected, template)
            validation.extend(final_validation)
            passed = all(item[key]["pass"] for item in validation for key in validation_keys)
            gathered_samples = self.gather(samples)
            if passed:
                row["status"] = "PASS"
                for name in samples:
                    summary = latency_summary([item[name] for item in gathered_samples])
                    row.update({f"{name}_{key}": value for key, value in summary.items()})
                row["speedup"] = row["torch_p50_us"] / row["mcr_p50_us"]
        for name in validation_keys:
            checks = [item[name] for item in validation]
            row[f"{name}_valid"] = all(item["pass"] for item in checks)
            errors = [item["max_abs_error"] for item in checks if item["max_abs_error"] is not None]
            row[f"{name}_max_abs_error"] = max(errors) if errors else None
            row[f"{name}_nonfinite_elements"] = sum(item["nonfinite_elements"] for item in checks)
        self.rows.append(row)
        self.details.append({"op": op, "numel": numel, "validation": validation,
                             "in_place": row["in_place"],
                             "validation_order": "initial ranks, then final ranks if timed",
                             "raw_samples_by_rank_us": gathered_samples})
        self.write_results()
        if self.rank == 0:
            latency = (f"MCR {row['mcr_p50_us']:.3f} us, torch {row['torch_p50_us']:.3f} us, "
                       f"speedup {row['speedup']:.3f}x") if passed else "validation failed; latency withheld"
            label = op
            if op == "all_reduce":
                label += " (in-place)" if self.args.in_place else " (out-of-place)"
            print(f"{label:28s} {row['input_bytes_per_rank']:10d} B  {row['status']:4s}  {latency}", flush=True)
        return passed

    def write_results(self):
        if self.rank != 0:
            return
        prefix = self.args.output
        prefix.parent.mkdir(parents=True, exist_ok=True)
        report = {"metadata": self.metadata, "rows": self.rows, "details": self.details}
        Path(f"{prefix}.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if self.rows:
            columns = list(dict.fromkeys(key for row in self.rows for key in row))
            with Path(f"{prefix}.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerows(self.rows)

    def close(self):
        self.synchronize()
        self.mpi.barrier()
        if self.args.backend == "nccl":
            self.native.finalize()
        self.dist.destroy_process_group()
        self.mpi.finalize()


def main():
    comparison = Comparison(parse_args())
    try:
        comparison.initialize()
        passed = True
        for op in comparison.args.ops:
            for numel in comparison.args.numels:
                passed = comparison.run_case(op, numel) and passed
        comparison.close()
        return 0 if passed else 1
    except Exception:
        error = traceback.format_exc()
        print(error, file=sys.stderr, flush=True)
        prefix = comparison.args.output
        prefix.parent.mkdir(parents=True, exist_ok=True)
        Path(f"{prefix}.rank{comparison.rank}.error.txt").write_text(error)
        if comparison.mpi is not None:
            comparison.mpi.abort(2)
        return 2


if __name__ == "__main__":
    sys.exit(main())
