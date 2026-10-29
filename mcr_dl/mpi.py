# Copyright 2023, The Ohio State University. All rights reserved.
# The MVAPICH software package is developed by the team members of
# The Ohio State University's Network-Based Computing Laboratory (NBCL),
# headed by Professor Dhabaleswar K. (DK) Panda.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Adapt the public MCR-DL API to NativeGroup's contiguous tensor operations."""

from contextlib import nullcontext

import torch
import torch.distributed as dist

from mcr_dl.ops.comm.mpi import build_mpi_op, initialize
from mcr_dl.utils import logger

from .backend import Backend
from .reduce_op import ReduceOp


def _stream(stream):
    # Keep native bindings independent of PyTorch's version-specific stream types.
    if stream is None:
        return None
    if not isinstance(stream, torch.cuda.Stream):
        raise TypeError("stream must be a torch.cuda.Stream or None")
    return stream.cuda_stream, stream.device.index


def _stream_context(tensor, stream):
    if stream is None:
        return nullcontext()
    _stream(stream)
    if stream.device != tensor.device:
        raise ValueError("MCR-DL MPI stream and tensor must be on the same CUDA device")
    # Packing, allocation, and unpacking must follow the same stream as MPI readiness.
    return torch.cuda.stream(stream)


class MPIBackend(Backend):
    """Caller-serialized MPI. CUDA buffers must be ordered on the supplied stream.

    stream=None uses the tensor device's current stream. Native operations block;
    list-output unpacking is queued on that stream. See docs/mpi_streams.md.
    """
    def __init__(self, name='mpi', rank=0, size=1, mpu=None):
        super().__init__(name='mpi')
        self.mpi_comm_op = build_mpi_op()
        self.has_allgather_base = True
        self.has_all_reduce_coalesced = False
        self.has_coalescing_manager = False
        self.enable_onebit = False
        self.using_mpi = True
        self.mpu = mpu
        self.init_process_group()

    def init_process_group(self):
        if self.initialized:
            return
        self.world_group = initialize()
        self.rank = self.world_rank = self.world_group.rank()
        self.size = self.world_size = self.world_group.size()
        self.single_gpu_mode = self.size == 1
        # Keep world ranks in communicator order for source/destination translation.
        self._group_ranks = {self.world_group: tuple(range(self.size))}
        self.initialized = True
        logger.info(f"Initialized MCR-DL MPI with rank {self.rank} and world size {self.size}")

    def new_group(self, ranks=None):
        self._group()
        ranks = tuple(range(self.size) if ranks is None else sorted(ranks))
        if (not ranks or len(set(ranks)) != len(ranks)
                or any(not isinstance(rank, int) or rank < 0 or rank >= self.size for rank in ranks)):
            raise ValueError("Group ranks must be distinct integers within the MPI world")
        # None means the world group, so nonmembers need a different sentinel.
        if self.rank not in ranks:
            return dist.GroupMember.NON_GROUP_MEMBER
        # Members must construct overlapping groups in the same order.
        group = self.mpi_comm_op.NativeGroup(list(ranks))
        self._group_ranks[group] = ranks
        return group

    def destroy_process_group(self, group=None):
        if group == dist.GroupMember.NON_GROUP_MEMBER:
            return
        if group is None or group is self.world_group:
            self._group_ranks.clear()
            self.world_group = None
            self.initialized = False
        else:
            del self._group_ranks[self._group(group)]
        # MPI and its communicators live until explicit finalize(), since NCCL
        # may still use the shared runtime after this Python backend is destroyed.

    def _group(self, group=None, async_op=False):
        if not self.initialized:
            raise RuntimeError("The MPI backend is not initialized")
        if group == dist.GroupMember.NON_GROUP_MEMBER:
            return None
        # Native collectives block. Only isend/irecv return NativeRequest objects.
        if async_op:
            raise NotImplementedError("Native MPI collectives do not support async_op=True")
        group = self.world_group if group is None else group
        if group not in self._group_ranks:
            raise ValueError("Use a group created by this MPI backend")
        return group

    def _group_rank(self, group, world_rank):
        if world_rank is None:
            raise NotImplementedError("Native MPI requires an explicit source rank")
        try:
            return self._group_ranks[group].index(world_rank)
        except ValueError:
            raise ValueError(f"World rank {world_rank} is not in the group") from None

    def get_world_group(self):
        return self._group()

    def get_rank(self, group=None):
        native = self._group(group)
        return -1 if native is None else native.rank()

    def get_world_size(self, group=None):
        native = self._group(group)
        return -1 if native is None else native.size()

    def get_global_rank(self, group, group_rank):
        native = self._group(group)
        if native is None or not 0 <= group_rank < native.size():
            raise RuntimeError("Group rank is out of range")
        return self._group_ranks[native][group_rank]

    @staticmethod
    def _reduce_op(op):
        # PyTorch's enum values differ from MCR-DL's; match names, not integers.
        enum = ReduceOp if isinstance(op, ReduceOp) else dist.ReduceOp
        for code, name in enumerate(('SUM', 'PRODUCT', 'MIN', 'MAX')):
            if op == getattr(enum, name):
                return code
        raise ValueError("Native MPI supports SUM, PRODUCT, MIN and MAX reductions")

    def barrier(self, group=None, async_op=False):
        native = self._group(group, async_op)
        if native is not None:
            native.barrier()

    def broadcast(self, tensor, src, op=ReduceOp.SUM, group=None, async_op=False, *, stream=None):
        # Keep the public op argument for compatibility; broadcast does not reduce.
        native = self._group(group, async_op)
        if native is not None:
            native.broadcast(tensor, self._group_rank(native, src), stream=_stream(stream))

    def all_reduce(self, tensor, op=ReduceOp.SUM, group=None, async_op=False, is_prof=False, *, stream=None):
        native = self._group(group, async_op)
        if native is not None:
            native.all_reduce(tensor, self._reduce_op(op), is_prof, stream=_stream(stream))

    inference_all_reduce = all_reduce

    def reduce(self, tensor, dst, op=ReduceOp.SUM, group=None, async_op=False, *, stream=None):
        native = self._group(group, async_op)
        if native is not None:
            native.reduce(tensor, self._group_rank(native, dst), self._reduce_op(op), stream=_stream(stream))

    def send(self, tensor, dst, group=None, tag=0, *, stream=None):
        native = self._group(group)
        if native is not None:
            native.send(tensor, self._group_rank(native, dst), tag, stream=_stream(stream))

    def recv(self, tensor, src=None, group=None, tag=0, *, stream=None):
        native = self._group(group)
        if native is None:
            return -1
        native.recv(tensor, self._group_rank(native, src), tag, stream=_stream(stream))
        return src

    def isend(self, tensor, dst, group=None, tag=0, *, stream=None):
        native = self._group(group)
        if native is not None:
            return native.isend(tensor, self._group_rank(native, dst), tag, stream=_stream(stream))

    def irecv(self, tensor, src=None, group=None, tag=0, *, stream=None):
        native = self._group(group)
        if native is not None:
            return native.irecv(tensor, self._group_rank(native, src), tag, stream=_stream(stream))

    def has_all_gather_into_tensor(self):
        return True

    def has_reduce_scatter_tensor(self):
        return True

    def all_gather_into_tensor(self, output_tensor, input_tensor, group=None, async_op=False, *, stream=None):
        native = self._group(group, async_op)
        if native is not None:
            native.all_gather(output_tensor, input_tensor, stream=_stream(stream))

    all_gather_base = all_gather_into_tensor

    def reduce_scatter_tensor(self, output_tensor, input_tensor, op=ReduceOp.SUM,
                              group=None, async_op=False, *, stream=None):
        native = self._group(group, async_op)
        if native is not None:
            native.reduce_scatter(output_tensor, input_tensor, self._reduce_op(op), stream=_stream(stream))

    @staticmethod
    def _check_list(tensors, reference, size):
        if not isinstance(tensors, (list, tuple)) or len(tensors) != size:
            raise ValueError("Provide one tensor per group rank")
        if not isinstance(reference, torch.Tensor):
            raise TypeError("Expected a tensor")
        for tensor in tensors:
            if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
                    or tensor.numel() != reference.numel() or tensor.dtype != reference.dtype
                    or tensor.device != reference.device):
                raise ValueError("Tensor lists must have equal element counts, dtype and device")

    @staticmethod
    def _pack(tensors):
        # The native API takes one contiguous buffer, ordered by group rank.
        with torch.no_grad():
            return torch.cat([tensor.reshape(-1) for tensor in tensors])

    @staticmethod
    def _unpack(packed, tensors):
        # Copy into the caller's tensors so list entries keep their identity.
        with torch.no_grad():
            for rank, tensor in enumerate(tensors):
                part = packed.narrow(0, rank * tensor.numel(), tensor.numel())
                tensor.copy_(part.reshape(tensor.shape))

    def all_gather(self, tensor_list, tensor, group=None, async_op=False, *, stream=None):
        native = self._group(group, async_op)
        if native is not None:
            with _stream_context(tensor, stream):
                self._check_list(tensor_list, tensor, native.size())
                packed = tensor.new_empty((native.size() * tensor.numel(),))
                native.all_gather(packed, tensor, stream=_stream(stream))
                self._unpack(packed, tensor_list)

    def reduce_scatter(self, output, input_list, op=ReduceOp.SUM, group=None, async_op=False, *, stream=None):
        native = self._group(group, async_op)
        if native is not None:
            with _stream_context(output, stream):
                self._check_list(input_list, output, native.size())
                native.reduce_scatter(output, self._pack(input_list), self._reduce_op(op), stream=_stream(stream))

    def gather(self, tensor, gather_list=None, dst=0, group=None, async_op=False, *, stream=None):
        native = self._group(group, async_op)
        if native is None:
            return
        with _stream_context(tensor, stream):
            root = self._group_rank(native, dst)
            packed = None
            if native.rank() == root:
                self._check_list(gather_list, tensor, native.size())
                packed = tensor.new_empty((native.size() * tensor.numel(),))
            elif gather_list:
                raise ValueError("Only the destination rank supplies gather_list")
            native.gather(packed, tensor, root, stream=_stream(stream))
            if packed is not None:
                self._unpack(packed, gather_list)

    def scatter(self, tensor, scatter_list=None, src=0, group=None, async_op=False, *, stream=None):
        native = self._group(group, async_op)
        if native is None:
            return
        with _stream_context(tensor, stream):
            root = self._group_rank(native, src)
            packed = None
            if native.rank() == root:
                self._check_list(scatter_list, tensor, native.size())
                packed = self._pack(scatter_list)
            elif scatter_list:
                raise ValueError("Only the source rank supplies scatter_list")
            native.scatter(tensor, packed, root, stream=_stream(stream))

    def all_to_all_single(self, output, input, output_split_sizes=None, input_split_sizes=None,
                          op=ReduceOp.SUM, group=None, async_op=False, is_prof=False, *, stream=None):
        native = self._group(group, async_op)
        if native is None:
            return
        # Public split sizes count rows; NativeGroup exchanges equal flat chunks.
        for tensor, splits in ((input, input_split_sizes), (output, output_split_sizes)):
            if tensor.dim() == 0 or tensor.shape[0] % native.size() != 0:
                raise ValueError("Equal-split all_to_all requires rows divisible by group size")
            if splits is not None and len(splits) != 0:
                if (len(splits) != native.size()
                        or any(split != tensor.shape[0] // native.size() for split in splits)):
                    raise NotImplementedError("Native MPI all_to_all supports only equal splits")
        native.all_to_all(output, input, is_prof, stream=_stream(stream))

    def all_to_all(self, output_tensor_list, input_tensor_list, group=None, async_op=False, *, stream=None):
        native = self._group(group, async_op)
        if native is not None:
            if not input_tensor_list:
                raise ValueError("Provide one input tensor per group rank")
            reference = input_tensor_list[0]
            with _stream_context(reference, stream):
                self._check_list(input_tensor_list, reference, native.size())
                self._check_list(output_tensor_list, reference, native.size())
                packed_input = self._pack(input_tensor_list)
                packed_output = torch.empty_like(packed_input)
                native.all_to_all(packed_output, packed_input, stream=_stream(stream))
                self._unpack(packed_output, output_tensor_list)
