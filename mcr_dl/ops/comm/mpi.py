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

"""Load MPI and share one default NativeGroup between Python backend instances."""

from mcr_dl.ops import op_builder

__all__ = ['build_mpi_op', 'initialize', 'finalize']

mpi_cpp_module = None
_world_group = None


def build_mpi_op():
    global mpi_cpp_module
    if mpi_cpp_module is None:
        builder = op_builder.MPICommBuilder()
        try:
            mpi_cpp_module = builder.load()
        except Exception as inst:
            raise RuntimeError(f"Failed to build {builder.absolute_name()}") from inst
    return mpi_cpp_module


def initialize():
    """Call collectively from one thread per rank; return the default group."""
    global _world_group
    module = build_mpi_op()
    module.initialize()
    if _world_group is None:
        # An empty rank list selects every rank in MPI_COMM_WORLD.
        _world_group = module.NativeGroup([])
    return _world_group


def finalize():
    """Call after all requests finish and all users of the MPI runtime stop."""
    global _world_group
    if mpi_cpp_module is not None:
        mpi_cpp_module.finalize()
        _world_group = None


# Tensor operations live on NativeGroup. The public MPIBackend handles tensor
# lists and world-rank translation instead of duplicating communication here.
