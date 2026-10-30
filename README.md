# MCR-DL

MCR-DL is a [HiDL](https://hidl.cse.ohio-state.edu/) project. We encourage you to visit the [HiDL website](https://hidl.cse.ohio-state.edu/) for additional information, the latest performance numbers, and similar projects on high-performance machine and deep learning. For the latest announcements on HiDL projects, [register for the HiDL mailing list](https://hidl.cse.ohio-state.edu/mailinglists/).

# News
**[12/24/2023]** The initial release of MCR-DL! For now, we only support basic single-backend communication without going through the PyTorch distributed module. For a full list of new and existing features, please see [the MCR-DL feature page](http://hidl.cse.ohio-state.edu/features/#MCR-DL)

# MCR-DL v0.1

The initial release of MCR-DL doesn't allow for mixed-backend optimizations. It will still allow users to decouple communication backends from PyTorch's distributed module. This enables much faster small-message performance, allows non-NCCL backends to be used with torch without messy source builds, enables communication logging, torch communication benchmarks, and greatly simplifies communication optimizations such as compression.


## Installation

### Prerequisites
- Python 3.8 or later (for Linux, Python 3.8.1+ is needed).
- Any MPI library (we recommend MVAPICH2-GDR), NCCL, or both </br>
Refer [MVAPICH2-GDR user guide](https://mvapich.cse.ohio-state.edu/userguide/gdr/) to install MVAPICH2-GDR.
- PyTorch 1.12.1 or later </br>
Refer [PyTorch installation guide](/docs/installation/PYTORCH_INSTALLATION_GUIDE.md) to install PyTorch from source and configure MVAPICH2-GDR support.

*Note:
We used the following versions during implementation and testing.
Python=3.9.16, cuda=11.7, gcc=10.3.0, cmake=3.22.2, PyTorch=2.0.1, MVAPICH2-GDR=2.3.7*

### Install MCR-DL
```bash
cd MCR-DL
python setup.py install
```

### Update Configurations
Update mpi, cuda, and nccl paths appropriately in [mcr_dl/config.yml](/mcr_dl/build_config.yml)
Example is available at: [mcr_dl/build.config.example.yaml](mcr_dl/build.config.example.yaml)

## Usage: MPI all-reduce

Run one MPI process per GPU with a CUDA-aware MPI library. This example uses `MPIBackend` directly, which also makes the native MPI backend choice explicit when integrating with an application that already initializes `torch.distributed`. Save it as `mpi_all_reduce.py`:

```python
import os

import torch
from mcr_dl.mpi import MPIBackend
from mcr_dl.ops.comm.mpi import finalize

# Select this process's GPU before initializing CUDA-aware MPI.
local_rank = int(os.environ["SLURM_LOCALID"])
torch.cuda.set_device(0 if torch.cuda.device_count() == 1 else local_rank)

comm = MPIBackend()
rank = comm.get_rank()
world_size = comm.get_world_size()

# Tensor creation and the readiness event use the same current CUDA stream.
stream = torch.cuda.current_stream()
tensor = torch.full((1024,), rank + 1, device="cuda", dtype=torch.bfloat16)
comm.all_reduce(tensor, stream=stream)  # Blocking, in-place SUM across all ranks.

# The result is ready when all_reduce returns.
expected = world_size * (world_size + 1) // 2
assert torch.all(tensor == expected).item()
if rank == 0:
    print(f"All-reduce passed on {world_size} ranks; each element is {expected}.")

comm.destroy_process_group()
finalize()
```

### CUDA stream

The native MPI tensor API accepts `stream=`, a `torch.cuda.Stream` on the tensor's device. Omitting it uses the current stream on that device. MCR-DL records a reusable CUDA event on the selected stream, waits for it on the CPU, and then calls MPI.
**The caller must establish the buffer's stream dependencies before calling MCR-DL.** All earlier conflicting reads and writes of input and output buffers must be ordered before that event. If another stream produced the tensor, make the selected stream wait for that producer before the all-reduce:

```python
stream.wait_stream(producer_stream)  # After submitting the producer's work.
comm.all_reduce(tensor, stream=stream)
```

MPI collectives are blocking; `async_op=True` and CUDA graph capture
are unsupported.

## Benchmarking Suite

The intent of these benchmarks is to measure communication latency/bw of MCR-DL and/or pytorch distributed communication operations at the Python layer. These benchmarks are complementary to C-level comms benchmarks like [OSU Micro-Benchmarks](https://mvapich.cse.ohio-state.edu/benchmarks/) and [NCCL Tests](https://github.com/NVIDIA/nccl-tests) in that users can:
- Easily debug which layer of the communication software stack hangs or performance degradations originate from.
- Measure the expected communication performance of either MCR-DL comms or pure PyTorch distributed

To run the existing benchmarks, there are two options:

1. Run a single communication operation:

<pre>
srun -N1 -n4 python all_reduce.py --scan --dist mcr_dl --backend mpi --dtype bfloat16 --warmups 50 --trials 100
</pre>

Use `--help` to find more commands.

Like the individual benchmarks, `run_all.py` supports scanning arguments for the max message size, bw-unit, etc. Simply pass the desired arguments to `run_all.py` and they'll be propagated to each comm op.


# Adding Communication Benchmarks

To add new communication benchmarks, follow this general procedure:

1. Copy a similar benchmark file (e.g. to add `reduce_scatter`, copy `all_reduce.py` as a template)
2. Add a new bw formula in `utils.get_bw`, a new maximum tensor element formula in `utils.max_numel`, and a new arg in `utils.benchmark_parser`
3. Replace comm op calls in new file with find-replace
4. Find a good default `mem_factor` for use in `run_<collective>_single()` function
5. Add new comm op to `run_all.py`


## References
1. Quentin Anthony, Ammar Ahmad Awan, Jeff Rasley, Yuxiong He, Aamir Shafi, Mustafa Abduljabbar, Hari Subramoni, Dhabaleswar Panda. (2023) MCR-DL: Mix-and-Match Communication Runtime for Deep Learning [arXiv:2303.08374](https://arxiv.org/abs/2303.08374) and will appear at IPDPS 2023.
