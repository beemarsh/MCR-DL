/*
* The MVAPICH software package is developed by the team members of
* The Ohio State University's Network-Based Computing Laboratory (NBCL),
* headed by Professor Dhabaleswar K. (DK) Panda.
*
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at

*     http://www.apache.org/licenses/LICENSE-2.0

* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <mpi.h>
#if defined(OPEN_MPI) && OPEN_MPI
#include <mpi-ext.h>
#endif
#include <nccl.h>
#include <pybind11/embed.h>
#include <torch/extension.h>
#include <chrono>
namespace py = pybind11;

#include <c10/util/irange.h>
#include <c10/core/DeviceGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <string>

#include <comm.h>

// TODO: remove
#include <stdio.h>

#define MPICHECK(cmd)                                                        \
    do {                                                                     \
        int e = cmd;                                                         \
        if (e != MPI_SUCCESS) {                                              \
            printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e); \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

#define CUDACHECK(cmd)                                                                            \
    do {                                                                                          \
        cudaError_t e = cmd;                                                                      \
        if (e != cudaSuccess) {                                                                   \
            printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(EXIT_FAILURE);                                                                   \
        }                                                                                         \
    } while (0)

namespace mpi {

int counter = 0;
ncclComm_t ncclcomm;

// py::module_ dist = py::module_::import("mcr_dl");

std::vector<MPI_Comm> global_mpi_comms;
std::mutex mpi_mutex;
bool owns_mpi = false;

void create_comms(int number = 1)
{
    int size = global_mpi_comms.size();
    global_mpi_comms.resize(size + number);

    for (int i = 0; i < number; ++i) {
        MPICHECK(MPI_Comm_dup(MPI_COMM_WORLD, &global_mpi_comms[size + i]));
    }
}

int get_rank(int group = 0)
{
    int world_rank;
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    return world_rank;
}

int get_world_size(int group = 0)
{
    int world_size;
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    return world_size;
}

void print_comm_number() { std::cout << "Number of Comms:" << global_mpi_comms.size() << "\n"; }

void increase_counter() { counter++; }

void decrease_counter() { counter--; }

void print_counter() { std::cout << "Counter is:" << counter << "\n"; }

void initialize()
{
    std::lock_guard<std::mutex> lock(mpi_mutex);
    int finalized;
    MPICHECK(MPI_Finalized(&finalized));
    TORCH_CHECK(!finalized, "MCR-DL cannot reinitialize MPI after finalization");
    int flag;
    MPICHECK(MPI_Initialized(&flag));
    int provided;
    if (!flag) {
        MPICHECK(MPI_Init_thread(NULL, NULL, MPI_THREAD_SERIALIZED, &provided));
        owns_mpi = true;
    } else {
        MPICHECK(MPI_Query_thread(&provided));
    }
    TORCH_CHECK(provided >= MPI_THREAD_SERIALIZED,
                "MCR-DL requires MPI_THREAD_SERIALIZED or MPI_THREAD_MULTIPLE");
    if (global_mpi_comms.empty()) create_comms();
}

void finalize()
{
    std::lock_guard<std::mutex> lock(mpi_mutex);
    int finalized;
    MPICHECK(MPI_Finalized(&finalized));
    if (owns_mpi && !finalized) MPICHECK(MPI_Finalize());
}

void abort_world(int code) { MPI_Abort(MPI_COMM_WORLD, code); }

MPI_Datatype get_mpi_datatype(c10::ScalarType type)
{
    MPI_Datatype mpi_type;
    switch (type) {
        case c10::ScalarType::Byte: mpi_type = MPI_UNSIGNED_CHAR; break;

        case c10::ScalarType::Char: mpi_type = MPI_SIGNED_CHAR; break;

        case c10::ScalarType::Short: mpi_type = MPI_SHORT; break;

        case c10::ScalarType::Bool: mpi_type = MPI_C_BOOL; break;

        case c10::ScalarType::Int: mpi_type = MPI_INT; break;

        case c10::ScalarType::Long: mpi_type = MPI_LONG; break;

        case c10::ScalarType::Float: mpi_type = MPI_FLOAT; break;

        case c10::ScalarType::Double: mpi_type = MPI_DOUBLE; break;

        case c10::ScalarType::BFloat16:
            // Match OSU-Nowlab/PyTorch's native MPI BF16 datatype selection.
#if defined(MPIX_C_BF16)
            mpi_type = MPIX_C_BF16;
#elif defined(MPIX_BFLOAT16)
            mpi_type = MPIX_BFLOAT16;
#else
            TORCH_CHECK(false,
                        "MCR-DL MPI bfloat16 support requires MPIX_C_BF16 or "
                        "MPIX_BFLOAT16. Rebuild MCR-DL with a BF16-capable MPI library.");
#endif
            break;

        case c10::ScalarType::Half:
#if defined(MPIX_C_FLOAT16)
            mpi_type = MPIX_C_FLOAT16;
#else
            TORCH_CHECK(false, "MCR-DL MPI float16 support requires MPIX_C_FLOAT16");
#endif
            break;

        default: TORCH_CHECK(false, "Unsupported MCR-DL MPI tensor dtype: ", type);
    }

    return mpi_type;
}

// Blocking, group-aware API used by inference runtimes. Unlike the legacy
// wrappers, communicator membership and reduction operations are separate.
class NativeGroup {
public:
    explicit NativeGroup(std::vector<int> ranks)
    {
        std::lock_guard<std::mutex> lock(mpi_mutex);
        int world_size, world_rank;
        MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
        MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
        if (ranks.empty()) {
            ranks.resize(world_size);
            std::iota(ranks.begin(), ranks.end(), 0);
        }
        TORCH_CHECK(std::find(ranks.begin(), ranks.end(), world_rank) != ranks.end(),
                    "Only members may construct an MCR-DL NativeGroup");
        MPI_Group world_group, subgroup;
        MPICHECK(MPI_Comm_group(MPI_COMM_WORLD, &world_group));
        MPICHECK(MPI_Group_incl(world_group, ranks.size(), ranks.data(), &subgroup));
        // This is collective only over members, matching torch.new_group's
        // custom-backend contract. Groups must be constructed in the same order.
        MPICHECK(MPI_Comm_create_group(MPI_COMM_WORLD, subgroup, 0, &comm_));
        MPICHECK(MPI_Group_free(&subgroup));
        MPICHECK(MPI_Group_free(&world_group));
        MPICHECK(MPI_Comm_rank(comm_, &rank_));
        MPICHECK(MPI_Comm_size(comm_, &size_));
    }

    // MPI_Finalize owns communicator cleanup. Python object destruction is not
    // ordered across ranks, so it must not perform collective MPI_Comm_free.
    int rank() const { return rank_; }
    int size() const { return size_; }

    void all_reduce(torch::Tensor tensor, int op)
    {
        std::lock_guard<std::mutex> lock(mpi_mutex);
        c10::OptionalDeviceGuard guard;
        prepare(tensor, guard);
        MPI_Op reduction;
        switch (op) {
            case 0: reduction = MPI_SUM; break;
            case 1: reduction = MPI_PROD; break;
            case 2: reduction = MPI_MIN; break;
            case 3: reduction = MPI_MAX; break;
            default: TORCH_CHECK(false, "Unsupported MCR-DL reduction operation");
        }
        MPICHECK(MPI_Allreduce(MPI_IN_PLACE, tensor.data_ptr(), count(tensor),
                               get_mpi_datatype(tensor.scalar_type()), reduction, comm_));
        finish(tensor);
    }

    void all_gather(torch::Tensor output, torch::Tensor input)
    {
        std::lock_guard<std::mutex> lock(mpi_mutex);
        TORCH_CHECK(output.scalar_type() == input.scalar_type() &&
                    output.device() == input.device() && output.is_contiguous() &&
                    output.numel() == input.numel() * size_,
                    "Invalid MCR-DL all_gather output");
        c10::OptionalDeviceGuard guard;
        prepare(input, guard);
        MPICHECK(MPI_Allgather(input.data_ptr(), count(input),
                               get_mpi_datatype(input.scalar_type()), output.data_ptr(),
                               count(input), get_mpi_datatype(input.scalar_type()), comm_));
        finish(output);
    }

    void broadcast(torch::Tensor tensor, int root)
    {
        std::lock_guard<std::mutex> lock(mpi_mutex);
        check_rank(root);
        c10::OptionalDeviceGuard guard;
        prepare(tensor, guard);
        MPICHECK(MPI_Bcast(tensor.data_ptr(), count(tensor),
                           get_mpi_datatype(tensor.scalar_type()), root, comm_));
        finish(tensor);
    }

    void send(torch::Tensor tensor, int destination, int tag)
    {
        std::lock_guard<std::mutex> lock(mpi_mutex);
        check_rank(destination);
        c10::OptionalDeviceGuard guard;
        prepare(tensor, guard);
        MPICHECK(MPI_Send(tensor.data_ptr(), count(tensor),
                          get_mpi_datatype(tensor.scalar_type()), destination, tag, comm_));
        finish(tensor);
    }

    void recv(torch::Tensor tensor, int source, int tag)
    {
        std::lock_guard<std::mutex> lock(mpi_mutex);
        check_rank(source);
        c10::OptionalDeviceGuard guard;
        prepare(tensor, guard);
        MPICHECK(MPI_Recv(tensor.data_ptr(), count(tensor),
                          get_mpi_datatype(tensor.scalar_type()), source, tag, comm_,
                          MPI_STATUS_IGNORE));
        finish(tensor);
    }

    void barrier()
    {
        std::lock_guard<std::mutex> lock(mpi_mutex);
        MPICHECK(MPI_Barrier(comm_));
    }

private:
    MPI_Comm comm_;
    int rank_, size_;

    void check_rank(int rank) const
    {
        TORCH_CHECK(rank >= 0 && rank < size_, "MCR-DL group rank out of range");
    }

    static int count(const torch::Tensor& tensor)
    {
        TORCH_CHECK(tensor.numel() <= std::numeric_limits<int>::max(),
                    "MCR-DL MPI tensor exceeds the MPI count limit");
        return static_cast<int>(tensor.numel());
    }

    static void prepare(const torch::Tensor& tensor, c10::OptionalDeviceGuard& guard)
    {
        TORCH_CHECK(tensor.is_contiguous() && !tensor.is_sparse(),
                    "MCR-DL MPI requires a dense contiguous tensor");
        TORCH_CHECK(tensor.is_cpu() || tensor.is_cuda(),
                    "MCR-DL MPI supports CPU and CUDA tensors");
        if (tensor.is_cuda()) {
            guard.reset_device(tensor.device());
            cudaStreamCaptureStatus status;
            CUDACHECK(cudaStreamIsCapturing(
                c10::cuda::getCurrentCUDAStream(tensor.get_device()).stream(), &status));
            TORCH_CHECK(status == cudaStreamCaptureStatusNone,
                        "MCR-DL MPI collectives cannot execute inside CUDA graph capture");
            // Correctness first: complete producer work before host-side MPI.
            CUDACHECK(cudaDeviceSynchronize());
        }
    }

    static void finish(const torch::Tensor& tensor)
    {
        if (tensor.is_cuda()) CUDACHECK(cudaDeviceSynchronize());
    }
};

void send(torch::Tensor data, int rank, int tag)
{
    MPICHECK(MPI_Send(data.data_ptr(),
                      data.numel(),
                      get_mpi_datatype(data.scalar_type()),
                      rank,
                      tag,
                      MPI_COMM_WORLD));
}

void barrier() { MPICHECK(MPI_Barrier(MPI_COMM_WORLD)); }

void recv(torch::Tensor data, int rank, int tag)
{
    MPICHECK(MPI_Recv(data.data_ptr(),
                      data.numel(),
                      get_mpi_datatype(data.scalar_type()),
                      rank,
                      tag,
                      MPI_COMM_WORLD,
                      MPI_STATUS_IGNORE));
}

size_t isend(torch::Tensor data, int rank, int tag, int comm = 0)
{
    MPI_Request req;
    MPICHECK(MPI_Isend(data.data_ptr(),
                       data.numel(),
                       get_mpi_datatype(data.scalar_type()),
                       rank,
                       tag,
                       global_mpi_comms[comm],
                       &req));
    size_t casted_req = size_t(req);
    return casted_req;
}

size_t irecv(torch::Tensor data, int rank, int tag, int comm = 0)
{
    MPI_Request req;
    MPICHECK(MPI_Irecv(data.data_ptr(),
                       data.numel(),
                       get_mpi_datatype(data.scalar_type()),
                       rank,
                       tag,
                       global_mpi_comms[comm],
                       &req));
    size_t casted_req = size_t(req);
    return casted_req;
}

void allreduce(torch::Tensor data, py::object comm, bool is_prof)
{
    std::chrono::steady_clock::time_point begin, end;
    torch::Tensor recvbuf = torch::empty_like(data);
    int comm_val = py::int_(comm.attr("value"));

    if (is_prof) { begin = std::chrono::steady_clock::now(); }
    MPICHECK(MPI_Allreduce(MPI_IN_PLACE,
                           data.data_ptr(),
                           data.numel(),
                           get_mpi_datatype(data.scalar_type()),
                           MPI_SUM,
                           global_mpi_comms[comm_val]));
    if (is_prof) {
        end = std::chrono::steady_clock::now();
        if (get_rank() == 0) {
            std::cout << "MPI allreduce time = "
                      << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
                      << "us, Size = " << data.numel() * data.element_size() << " B"
                      << "\n";
        }
    }
}

void allgather(torch::Tensor outputTensor, torch::Tensor inputTensor, int comm = 0)
{
    MPICHECK(MPI_Allgather(inputTensor.data_ptr(),
                           inputTensor.numel() / get_world_size(0),
                           get_mpi_datatype(inputTensor.scalar_type()),
                           outputTensor.data_ptr(),
                           outputTensor.numel() / get_world_size(0),
                           get_mpi_datatype(outputTensor.scalar_type()),
                           global_mpi_comms[comm]));
}

void gather(torch::Tensor data, int root_rank, int comm = 0)
{
    torch::Tensor recvbuf = torch::empty_like(data);
    MPICHECK(MPI_Gather(data.data_ptr(),
                        data.numel(),
                        get_mpi_datatype(data.scalar_type()),
                        recvbuf.data_ptr(),
                        data.numel(),
                        get_mpi_datatype(data.scalar_type()),
                        root_rank,
                        global_mpi_comms[comm]));
}

void scatter(torch::Tensor data, int root_rank, int comm = 0)
{
    torch::Tensor recvbuf = torch::empty_like(data);
    MPICHECK(MPI_Scatter(data.data_ptr(),
                         data.numel(),
                         get_mpi_datatype(data.scalar_type()),
                         recvbuf.data_ptr(),
                         data.numel(),
                         get_mpi_datatype(data.scalar_type()),
                         root_rank,
                         global_mpi_comms[comm]));
}
/*
void reduce_scatter(torch::Tensor data, int comm=0)
{
  torch::Tensor recvbuf = torch::empty_like(data);
  MPI_Reduce_scatter( data.data_ptr(),
            recvbuf.data_ptr(),
            data.numel(),
            get_mpi_datatype(data.scalar_type()),
            MPI_SUM,
            global_mpi_comms[comm]);
}
*/
void reduce(torch::Tensor data, int root_rank, int comm = 0)
{
    torch::Tensor recvbuf = torch::empty_like(data);
    MPICHECK(MPI_Reduce(data.data_ptr(),
                        recvbuf.data_ptr(),
                        data.numel(),
                        get_mpi_datatype(data.scalar_type()),
                        MPI_SUM,
                        root_rank,
                        global_mpi_comms[comm]));
}

void bcast(torch::Tensor data, int root_rank, py::object comm)
{
    int comm_val = py::int_(comm.attr("value"));

    MPICHECK(MPI_Bcast(data.data_ptr(),
                       data.numel(),
                       get_mpi_datatype(data.scalar_type()),
                       root_rank,
                       global_mpi_comms[comm_val]));
}

void alltoall(torch::Tensor outputTensor, torch::Tensor inputTensor, py::object comm, bool is_prof)
{
    std::chrono::steady_clock::time_point begin, end;
    int comm_val = py::int_(comm.attr("value"));

    if (is_prof) { begin = std::chrono::steady_clock::now(); }
    MPICHECK(MPI_Alltoall(inputTensor.data_ptr(),
                          inputTensor.numel() / get_world_size(0),
                          get_mpi_datatype(inputTensor.scalar_type()),
                          outputTensor.data_ptr(),
                          outputTensor.numel() / get_world_size(0),
                          get_mpi_datatype(outputTensor.scalar_type()),
                          global_mpi_comms[comm_val]));
    if (is_prof) {
        end = std::chrono::steady_clock::now();
        if (get_rank(0) == 0) {
            std::cout << "MPI alltoall time = "
                      << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
                      << "us, Size = " << inputTensor.numel() * inputTensor.element_size() << " B"
                      << "\n";
        }
    }
}

void alltoall_list(std::vector<torch::Tensor>& outputTensors,
                   std::vector<torch::Tensor>& inputTensors,
                   int comm = 0)
{
    for (int t = 0; t < inputTensors.size(); t++) {
        torch::Tensor& input = inputTensors[t];
        torch::Tensor& output = outputTensors[t];
        MPICHECK(MPI_Alltoall(input.data_ptr(),
                              input.numel(),
                              get_mpi_datatype(input.scalar_type()),
                              output.data_ptr(),
                              output.numel(),
                              get_mpi_datatype(output.scalar_type()),
                              global_mpi_comms[comm]));
    }
}

void wait(size_t casted_req)
{
    MPI_Request req = (MPI_Request)casted_req;
    MPI_Status status;

    MPICHECK(MPI_Wait(&req, &status));
}

void device_sync() { CUDACHECK(cudaDeviceSynchronize()); }

torch::Tensor wait_multi(size_t casted_req, torch::Tensor data)
{
    MPI_Request req = (MPI_Request)casted_req;
    MPI_Status status;

    MPICHECK(MPI_Wait(&req, &status));
    return data;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<NativeGroup>(m, "NativeGroup")
        .def(py::init<std::vector<int>>(), py::call_guard<py::gil_scoped_release>())
        .def("rank", &NativeGroup::rank)
        .def("size", &NativeGroup::size)
        .def("all_reduce", &NativeGroup::all_reduce, py::call_guard<py::gil_scoped_release>())
        .def("all_gather", &NativeGroup::all_gather, py::call_guard<py::gil_scoped_release>())
        .def("broadcast", &NativeGroup::broadcast, py::call_guard<py::gil_scoped_release>())
        .def("send", &NativeGroup::send, py::call_guard<py::gil_scoped_release>())
        .def("recv", &NativeGroup::recv, py::call_guard<py::gil_scoped_release>())
        .def("barrier", &NativeGroup::barrier, py::call_guard<py::gil_scoped_release>());
    m.def("abort", &abort_world, py::call_guard<py::gil_scoped_release>());
    m.def("send", &send, "mpi send");
    m.def("recv", &recv, "mpi recv");
    m.def("isend", &isend, "mpi isend");
    m.def("irecv", &irecv, "mpi irecv");
    m.def("allreduce", &allreduce, "mpi allreduce");
    m.def("allgather", &allgather, "mpi allgather");
    m.def("gather", &gather, "mpi gather");
    m.def("scatter", &scatter, "mpi scatter");
    // m.def("reduce_scatter", &reduce_scatter, "mpi reduce-scatter");
    m.def("reduce", &reduce, "mpi reduce");
    m.def("bcast", &bcast, "mpi bcast");
    m.def("alltoall", &alltoall, "mpi alltoall");
    m.def("alltoall_list", &alltoall_list, "mpi alltoall list");
    m.def("wait", &wait, "mpi wait");
    m.def("device_sync", &device_sync, "mpi device sync");
    m.def("wait_multi", &wait_multi, "mpi wait multi");
    m.def("initialize", &initialize, "mpi initialize", py::call_guard<py::gil_scoped_release>());
    m.def("finalize", &finalize, "mpi finalize", py::call_guard<py::gil_scoped_release>());
    m.def("get_rank", &get_rank, "get rank");
    m.def("barrier", &barrier, "barrier");
    m.def("get_world_size", &get_world_size, "get world size");
    m.def("increase_counter", &increase_counter, "mpi increase counter");
    m.def("decrease_counter", &decrease_counter, "mpi decrease counter");
    m.def("print_counter", &print_counter, "mpi print counter");
    m.def("create_comms", &create_comms, "mpi create comms");
    m.def("print_comm_number", &print_comm_number, "mpi print comm number");
}

}  // namespace mpi
