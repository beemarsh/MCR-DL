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

#include <cuda_runtime_api.h>
#include <mpi.h>
#if defined(OPEN_MPI) && OPEN_MPI
#include <mpi-ext.h>
#endif
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <chrono>
namespace py = pybind11;

#include <ATen/cuda/CUDAEvent.h>
#include <c10/core/DeviceGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <map>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include <stdio.h>

#define MPICHECK(cmd)                                                        \
    do                                                                       \
    {                                                                        \
        int e = cmd;                                                         \
        if (e != MPI_SUCCESS)                                                \
        {                                                                    \
            printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e); \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

#define CUDACHECK(cmd)                                                                            \
    do                                                                                            \
    {                                                                                             \
        cudaError_t e = cmd;                                                                      \
        if (e != cudaSuccess)                                                                     \
        {                                                                                         \
            printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(EXIT_FAILURE);                                                                   \
        }                                                                                         \
    } while (0)

namespace mpi
{

    // The caller serializes whole MCR-DL operations (including event waits),
    // requests, and other MPI calls in this process.
    using Stream = std::optional<std::pair<std::uintptr_t, int>>; // CUDA handle, device
    bool owns_mpi = false;

    // These world helpers are also used to bootstrap the NCCL backend.
    int get_rank(int group = 0)
    {
        TORCH_CHECK(group == 0, "Use NativeGroup.rank() for subgroup queries");
        int world_rank;
        MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
        return world_rank;
    }

    int get_world_size(int group = 0)
    {
        TORCH_CHECK(group == 0, "Use NativeGroup.size() for subgroup queries");
        int world_size;
        MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
        return world_size;
    }

    void barrier()
    {
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    }

    void initialize()
    {
        int finalized;
        MPICHECK(MPI_Finalized(&finalized));
        TORCH_CHECK(!finalized, "MCR-DL cannot reinitialize MPI after finalization");
        int flag;
        MPICHECK(MPI_Initialized(&flag));
        int provided;
        if (!flag)
        {
            MPICHECK(MPI_Init_thread(NULL, NULL, MPI_THREAD_SERIALIZED, &provided));
            owns_mpi = true;
        }
        else
        {
            MPICHECK(MPI_Query_thread(&provided));
        }
        TORCH_CHECK(provided >= MPI_THREAD_SERIALIZED,
                    "MCR-DL requires MPI_THREAD_SERIALIZED or MPI_THREAD_MULTIPLE");
    }

    void finalize()
    {
        int finalized;
        MPICHECK(MPI_Finalized(&finalized));
        if (owns_mpi && !finalized)
            MPICHECK(MPI_Finalize());
    }

    void abort_world(int code) { MPI_Abort(MPI_COMM_WORLD, code); }

    MPI_Datatype get_mpi_datatype(c10::ScalarType type)
    {
        MPI_Datatype mpi_type;
        switch (type)
        {
        case c10::ScalarType::Byte:
            mpi_type = MPI_UNSIGNED_CHAR;
            break;

        case c10::ScalarType::Char:
            mpi_type = MPI_SIGNED_CHAR;
            break;

        case c10::ScalarType::Short:
            mpi_type = MPI_SHORT;
            break;

        case c10::ScalarType::Bool:
            mpi_type = MPI_C_BOOL;
            break;

        case c10::ScalarType::Int:
            mpi_type = MPI_INT;
            break;

        case c10::ScalarType::Long:
            mpi_type = MPI_LONG;
            break;

        case c10::ScalarType::Float:
            mpi_type = MPI_FLOAT;
            break;

        case c10::ScalarType::Double:
            mpi_type = MPI_DOUBLE;
            break;

        case c10::ScalarType::BFloat16:
            // Match native MPI BF16 datatype selection.
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

        default:
            TORCH_CHECK(false, "Unsupported MCR-DL MPI tensor dtype: ", type);
        }

        return mpi_type;
    }

    // Represents a native MPI request, ensuring the associated tensor remains alive until the request completes.
    // Currently this is only for non-blocking send and recv.
    class NativeRequest
    {
    public:
        explicit NativeRequest(torch::Tensor tensor) : tensor_(std::move(tensor)) {}
        NativeRequest(const NativeRequest &) = delete;
        NativeRequest &operator=(const NativeRequest &) = delete;

        ~NativeRequest()
        {
            // Dropping an unfinished request waits rather than freeing storage
            // that MPI may still use. The caller must also serialize destruction.
            if (Py_IsInitialized() && PyGILState_Check())
            {
                py::gil_scoped_release release;
                complete(true, true);
            }
            else
            {
                complete(true, true);
            }
        }

        bool wait()
        {
            return complete(true);
        }

        bool is_completed()
        {
            return complete(false);
        }

    private:
        friend class NativeGroup;
        torch::Tensor tensor_;
        MPI_Request request_ = MPI_REQUEST_NULL;
        bool completed_ = false;

        bool complete(bool wait, bool destroying = false)
        {
            if (completed_)
                return true;
            int finalized;
            MPICHECK(MPI_Finalized(&finalized));
            if (finalized)
            {
                // Requests must be completed before MPI_Finalize. Destruction
                // after finalization must never use a stale request handle.
                TORCH_CHECK(destroying, "Complete MCR-DL requests before finalizing MPI");
                return true;
            }
            c10::OptionalDeviceGuard guard;
            if (tensor_.is_cuda())
                guard.reset_device(tensor_.device());
            int done = 1;
            if (wait)
                MPICHECK(MPI_Wait(&request_, MPI_STATUS_IGNORE));
            else
                MPICHECK(MPI_Test(&request_, &done, MPI_STATUS_IGNORE));
            if (done)
            {
                completed_ = true;
                tensor_ = torch::Tensor();
            }
            return done != 0;
        }
    };

    // One communicator for a set of existing MPI ranks. Tensor operations use
    // this group; it does not create processes or worker threads.
    class NativeGroup
    {
    public:
        NativeGroup(const NativeGroup &) = delete;
        NativeGroup &operator=(const NativeGroup &) = delete;

        explicit NativeGroup(std::vector<int> ranks)
        {
            int world_size, world_rank;
            MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
            MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
            if (ranks.empty())
            {
                ranks.resize(world_size);
                std::iota(ranks.begin(), ranks.end(), 0);
            }
            TORCH_CHECK(std::find(ranks.begin(), ranks.end(), world_rank) != ranks.end(),
                        "Only members may construct an MCR-DL NativeGroup");
            MPI_Group world_group, subgroup;
            MPICHECK(MPI_Comm_group(MPI_COMM_WORLD, &world_group));
            MPICHECK(MPI_Group_incl(world_group, ranks.size(), ranks.data(), &subgroup));
            // Members must create overlapping groups in the same order.
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

        void all_reduce(torch::Tensor tensor, int op, bool is_prof = false, Stream stream = std::nullopt)
        {
            c10::OptionalDeviceGuard guard;
            prepare(tensor, guard, stream);
            const auto reduction = reduction_op(op);
            const auto datatype = get_mpi_datatype(tensor.scalar_type());
            const auto elements = count(tensor);
            const auto begin = is_prof ? Clock::now() : Clock::time_point{};
            MPICHECK(MPI_Allreduce(MPI_IN_PLACE, tensor.data_ptr(), elements,
                                   datatype, reduction, comm_));
            if (is_prof)
                print_elapsed("all_reduce", begin, tensor);
        }

        void all_gather(torch::Tensor output, torch::Tensor input, Stream stream = std::nullopt)
        {
            check_buffers(output, input);
            TORCH_CHECK(output.numel() % size_ == 0 &&
                            output.numel() / size_ == input.numel(),
                        "Invalid MCR-DL all_gather output");
            c10::OptionalDeviceGuard guard;
            prepare(input, guard, stream);
            MPICHECK(MPI_Allgather(input.data_ptr(), count(input),
                                   get_mpi_datatype(input.scalar_type()), output.data_ptr(),
                                   count(input), get_mpi_datatype(input.scalar_type()), comm_));
        }

        void all_to_all(torch::Tensor output, torch::Tensor input, bool is_prof = false, Stream stream = std::nullopt)
        {
            check_buffers(output, input);
            TORCH_CHECK(output.numel() == input.numel() && input.numel() % size_ == 0,
                        "MCR-DL all_to_all requires equally sized tensors divisible by group size");
            const auto elements = count(input.numel() / size_);
            const auto datatype = get_mpi_datatype(input.scalar_type());
            c10::OptionalDeviceGuard guard;
            prepare(input, guard, stream);
            const auto begin = is_prof ? Clock::now() : Clock::time_point{};
            MPICHECK(MPI_Alltoall(input.data_ptr(), elements, datatype,
                                 output.data_ptr(), elements, datatype, comm_));
            if (is_prof)
                print_elapsed("all_to_all", begin, input);
        }

        // The root's tensor receives the reduction; other ranks keep their input.
        void reduce(torch::Tensor tensor, int root, int op, Stream stream = std::nullopt)
        {
            check_rank(root);
            c10::OptionalDeviceGuard guard;
            prepare(tensor, guard, stream);
            const auto reduction = reduction_op(op);
            MPICHECK(MPI_Reduce(rank_ == root ? MPI_IN_PLACE : tensor.data_ptr(),
                               rank_ == root ? tensor.data_ptr() : nullptr, count(tensor),
                               get_mpi_datatype(tensor.scalar_type()), reduction, root, comm_));
        }

        // output is a concatenation in group-rank order, supplied only at root.
        void gather(std::optional<torch::Tensor> output, torch::Tensor input, int root, Stream stream = std::nullopt)
        {
            check_rank(root);
            if (rank_ == root)
            {
                TORCH_CHECK(output.has_value(), "MCR-DL gather requires an output at root");
                check_buffers(*output, input);
                TORCH_CHECK(output->numel() % size_ == 0 &&
                                output->numel() / size_ == input.numel(),
                            "MCR-DL gather output must hold one input per group rank");
            }
            c10::OptionalDeviceGuard guard;
            prepare(input, guard, stream);
            const auto datatype = get_mpi_datatype(input.scalar_type());
            const auto elements = count(input);
            MPICHECK(MPI_Gather(input.data_ptr(), elements, datatype,
                               rank_ == root ? output->data_ptr() : nullptr,
                               elements, datatype, root, comm_));
        }

        // input is a concatenation in group-rank order, supplied only at root.
        void scatter(torch::Tensor output, std::optional<torch::Tensor> input, int root, Stream stream = std::nullopt)
        {
            check_rank(root);
            if (rank_ == root)
            {
                TORCH_CHECK(input.has_value(), "MCR-DL scatter requires an input at root");
                check_buffers(output, *input);
                TORCH_CHECK(input->numel() % size_ == 0 &&
                                input->numel() / size_ == output.numel(),
                            "MCR-DL scatter input must hold one output per group rank");
            }
            c10::OptionalDeviceGuard guard;
            prepare(output, guard, stream);
            const auto datatype = get_mpi_datatype(output.scalar_type());
            const auto elements = count(output);
            MPICHECK(MPI_Scatter(rank_ == root ? input->data_ptr() : nullptr,
                                elements, datatype, output.data_ptr(), elements,
                                datatype, root, comm_));
        }

        void reduce_scatter(torch::Tensor output, torch::Tensor input, int op, Stream stream = std::nullopt)
        {
            check_buffers(output, input);
            TORCH_CHECK(input.numel() % size_ == 0 &&
                            input.numel() / size_ == output.numel(),
                        "MCR-DL reduce_scatter input must hold one output per group rank");
            const auto reduction = reduction_op(op);
            c10::OptionalDeviceGuard guard;
            prepare(input, guard, stream);
            MPICHECK(MPI_Reduce_scatter_block(input.data_ptr(), output.data_ptr(), count(output),
                                            get_mpi_datatype(input.scalar_type()), reduction, comm_));
        }

        void broadcast(torch::Tensor tensor, int root, Stream stream = std::nullopt)
        {
            check_rank(root);
            c10::OptionalDeviceGuard guard;
            prepare(tensor, guard, stream);
            MPICHECK(MPI_Bcast(tensor.data_ptr(), count(tensor),
                               get_mpi_datatype(tensor.scalar_type()), root, comm_));
        }

        void send(torch::Tensor tensor, int destination, int tag, Stream stream = std::nullopt)
        {
            check_rank(destination);
            c10::OptionalDeviceGuard guard;
            prepare(tensor, guard, stream);
            MPICHECK(MPI_Send(tensor.data_ptr(), count(tensor),
                              get_mpi_datatype(tensor.scalar_type()), destination, tag, comm_));
        }

        void recv(torch::Tensor tensor, int source, int tag, Stream stream = std::nullopt)
        {
            check_rank(source);
            c10::OptionalDeviceGuard guard;
            prepare(tensor, guard, stream);
            MPICHECK(MPI_Recv(tensor.data_ptr(), count(tensor),
                              get_mpi_datatype(tensor.scalar_type()), source, tag, comm_,
                              MPI_STATUS_IGNORE));
        }

        std::shared_ptr<NativeRequest> isend(torch::Tensor tensor, int destination, int tag, Stream stream = std::nullopt)
        {
            check_rank(destination);
            check_tag(tag);
            c10::OptionalDeviceGuard guard;
            prepare(tensor, guard, stream);
            const auto elements = count(tensor);
            const auto datatype = get_mpi_datatype(tensor.scalar_type());
            auto request = std::make_shared<NativeRequest>(tensor);
            MPICHECK(MPI_Isend(tensor.data_ptr(), elements, datatype,
                              destination, tag, comm_, &request->request_));
            return request;
        }

        std::shared_ptr<NativeRequest> irecv(torch::Tensor tensor, int source, int tag, Stream stream = std::nullopt)
        {
            check_rank(source);
            check_tag(tag);
            c10::OptionalDeviceGuard guard;
            prepare(tensor, guard, stream);
            const auto elements = count(tensor);
            const auto datatype = get_mpi_datatype(tensor.scalar_type());
            auto request = std::make_shared<NativeRequest>(tensor);
            MPICHECK(MPI_Irecv(tensor.data_ptr(), elements, datatype,
                              source, tag, comm_, &request->request_));
            return request;
        }

        void barrier()
        {
            MPICHECK(MPI_Barrier(comm_));
        }

    private:
        using Clock = std::chrono::steady_clock;
        MPI_Comm comm_;
        int rank_, size_;
        // Created on first use; PyTorch's event wrapper handles device-aware cleanup.
        std::map<int, at::cuda::CUDAEvent> ready_events_;

        static MPI_Op reduction_op(int op)
        {
            switch (op)
            {
            case 0:
                return MPI_SUM;
            case 1:
                return MPI_PROD;
            case 2:
                return MPI_MIN;
            case 3:
                return MPI_MAX;
            default:
                TORCH_CHECK(false, "Unsupported MCR-DL reduction operation");
            }
        }

        static void check_buffers(const torch::Tensor &output, const torch::Tensor &input)
        {
            TORCH_CHECK(output.defined() && input.defined() &&
                            output.is_contiguous() && input.is_contiguous() &&
                            !output.is_sparse() && !input.is_sparse(),
                        "MCR-DL MPI requires dense contiguous input and output tensors");
            TORCH_CHECK(output.scalar_type() == input.scalar_type() &&
                            output.device() == input.device(),
                        "MCR-DL MPI input and output must have the same dtype and device");
            TORCH_CHECK(input.is_cpu() || input.is_cuda(),
                        "MCR-DL MPI supports CPU and CUDA tensors");
            if (output.numel() == 0 || input.numel() == 0)
                return;
            const auto out = reinterpret_cast<std::uintptr_t>(output.data_ptr());
            const auto in = reinterpret_cast<std::uintptr_t>(input.data_ptr());
            TORCH_CHECK(out <= in ? in - out >= output.numel() * output.element_size()
                                  : out - in >= input.numel() * input.element_size(),
                        "MCR-DL MPI input and output buffers must not overlap");
        }

        void print_elapsed(const char *operation, Clock::time_point begin,
                           const torch::Tensor &tensor) const
        {
            // Measure only the MPI call; CUDA synchronization is outside this
            // interval. Only group rank zero prints.
            const auto end = Clock::now();
            if (rank_ == 0)
                std::cout << "MCR-DL NativeGroup " << operation << " MPI time = "
                          << std::chrono::duration<double, std::micro>(end - begin).count()
                          << " us, Size = " << tensor.numel() * tensor.element_size() << " B\n";
        }

        void check_rank(int rank) const
        {
            TORCH_CHECK(rank >= 0 && rank < size_, "MCR-DL group rank out of range");
        }

        void check_tag(int tag) const
        {
            int *upper_bound = nullptr;
            int present;
            MPICHECK(MPI_Comm_get_attr(comm_, MPI_TAG_UB, &upper_bound, &present));
            // Predefined attributes are guaranteed on MPI_COMM_WORLD; some
            // implementations do not attach them to newly created groups.
            if (!present)
                MPICHECK(MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &upper_bound, &present));
            TORCH_CHECK(present && upper_bound != nullptr && tag >= 0 && tag <= *upper_bound,
                        "MCR-DL MPI message tag out of range");
        }

        static int count(const torch::Tensor &tensor)
        {
            return count(tensor.numel());
        }

        static int count(int64_t elements)
        {
            TORCH_CHECK(elements <= std::numeric_limits<int>::max(),
                        "MCR-DL MPI tensor exceeds the MPI count limit");
            return static_cast<int>(elements);
        }

        void prepare(const torch::Tensor &tensor, c10::OptionalDeviceGuard &guard,
                     const Stream &stream)
        {
            TORCH_CHECK(tensor.is_contiguous() && !tensor.is_sparse(),
                        "MCR-DL MPI requires a dense contiguous tensor");
            TORCH_CHECK(tensor.is_cpu() || tensor.is_cuda(),
                        "MCR-DL MPI supports CPU and CUDA tensors");
            if (tensor.is_cpu())
            {
                TORCH_CHECK(!stream, "CPU tensors do not accept a CUDA stream");
                return;
            }
            TORCH_CHECK(!stream || stream->second == tensor.get_device(),
                        "MCR-DL MPI stream and tensor must be on the same device");
            guard.reset_device(tensor.device());
            const auto cuda_stream = stream
                ? c10::cuda::getStreamFromExternal(
                      reinterpret_cast<cudaStream_t>(stream->first), stream->second)
                : c10::cuda::getCurrentCUDAStream(tensor.get_device());
            cudaStreamCaptureStatus status;
            CUDACHECK(cudaStreamIsCapturing(cuda_stream.stream(), &status));
            TORCH_CHECK(status == cudaStreamCaptureStatusNone,
                        "MCR-DL MPI operations cannot execute inside CUDA graph capture");
            // The caller orders all buffer accesses before this event. Wait for
            // that stream's work before MPI touches the buffers, not the whole GPU.
            auto &ready = ready_events_[tensor.get_device()]; // Timing disabled by default.
            ready.record(cuda_stream);
            ready.synchronize();
            // CUDA-aware MPI completes buffer access before a blocking call (or
            // a request's completion) returns, so no post-MPI CUDA wait is needed.
        }
    };

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
    {
        py::class_<NativeRequest, std::shared_ptr<NativeRequest>>(m, "NativeRequest",
            "Owns an asynchronous MPI buffer. Caller serializes all MPI calls. Complete before MPI finalization; "
            "destroying an unfinished request waits for completion.")
            .def("wait", &NativeRequest::wait,
                 "Wait for communication and CUDA completion; returns True.",
                 py::call_guard<py::gil_scoped_release>())
            .def("is_completed", &NativeRequest::is_completed,
                 "Poll MPI; a completed request's buffer is ready for use.",
                 py::call_guard<py::gil_scoped_release>());
        py::class_<NativeGroup>(m, "NativeGroup",
            "Caller-serialized MPI communicator. stream=None uses the tensor device's current CUDA stream; "
            "otherwise pass (CUDA stream handle, device index). Order all buffer accesses before this stream. "
            "Blocking operations return with MPI buffer access complete; CUDA graph capture is unsupported.")
            .def(py::init<std::vector<int>>(), py::call_guard<py::gil_scoped_release>())
            .def("rank", &NativeGroup::rank)
            .def("size", &NativeGroup::size)
            .def("all_reduce", &NativeGroup::all_reduce,
                 "In-place reduction; is_prof prints MPI-only latency on group rank zero.",
                 py::arg("tensor"), py::arg("op"), py::arg("is_prof") = false,
                 py::kw_only(), py::arg("stream") = py::none(),
                 py::call_guard<py::gil_scoped_release>())
            .def("all_gather", &NativeGroup::all_gather,
                 py::arg("output"), py::arg("input"),
                 py::kw_only(), py::arg("stream") = py::none(),
                 py::call_guard<py::gil_scoped_release>())
            .def("all_to_all", &NativeGroup::all_to_all,
                 "Exchange equal contiguous chunks in group-rank order. Buffers must not overlap; "
                 "is_prof prints MPI-only latency on group rank zero.",
                 py::arg("output"), py::arg("input"), py::arg("is_prof") = false,
                 py::kw_only(), py::arg("stream") = py::none(),
                 py::call_guard<py::gil_scoped_release>())
            .def("reduce", &NativeGroup::reduce,
                 "Reduce into tensor at the group-local root; op: 0=SUM, 1=PROD, 2=MIN, 3=MAX.",
                 py::arg("tensor"), py::arg("root"), py::arg("op") = 0,
                 py::kw_only(), py::arg("stream") = py::none(),
                 py::call_guard<py::gil_scoped_release>())
            .def("gather", &NativeGroup::gather,
                 "Gather equal inputs into a distinct concatenated output at the group-local root. "
                 "Pass output=None on other ranks.",
                 py::arg("output"), py::arg("input"), py::arg("root"),
                 py::kw_only(), py::arg("stream") = py::none(),
                 py::call_guard<py::gil_scoped_release>())
            .def("scatter", &NativeGroup::scatter,
                 "Scatter equal chunks of the root's concatenated input into distinct outputs. "
                 "Root is group-local; pass input=None on other ranks.",
                 py::arg("output"), py::arg("input"), py::arg("root"),
                 py::kw_only(), py::arg("stream") = py::none(),
                 py::call_guard<py::gil_scoped_release>())
            .def("reduce_scatter", &NativeGroup::reduce_scatter,
                 "Reduce concatenated inputs and scatter equal chunks in group-rank order. "
                 "Buffers must not overlap; op: 0=SUM, 1=PROD, 2=MIN, 3=MAX.",
                 py::arg("output"), py::arg("input"), py::arg("op") = 0,
                 py::kw_only(), py::arg("stream") = py::none(),
                 py::call_guard<py::gil_scoped_release>())
            .def("broadcast", &NativeGroup::broadcast,
                 py::arg("tensor"), py::arg("root"),
                 py::kw_only(), py::arg("stream") = py::none(),
                 py::call_guard<py::gil_scoped_release>())
            .def("send", &NativeGroup::send,
                 py::arg("tensor"), py::arg("destination"), py::arg("tag"),
                 py::kw_only(), py::arg("stream") = py::none(),
                 py::call_guard<py::gil_scoped_release>())
            .def("recv", &NativeGroup::recv,
                 py::arg("tensor"), py::arg("source"), py::arg("tag"),
                 py::kw_only(), py::arg("stream") = py::none(),
                 py::call_guard<py::gil_scoped_release>())
            .def("isend", &NativeGroup::isend,
                 "Send to a group-local rank; retain the request and do not modify tensor until completion.",
                 py::arg("tensor"), py::arg("destination"), py::arg("tag") = 0,
                 py::kw_only(), py::arg("stream") = py::none(),
                 py::call_guard<py::gil_scoped_release>())
            .def("irecv", &NativeGroup::irecv,
                 "Receive from a group-local rank; retain the request and do not access tensor until completion.",
                 py::arg("tensor"), py::arg("source"), py::arg("tag") = 0,
                 py::kw_only(), py::arg("stream") = py::none(),
                 py::call_guard<py::gil_scoped_release>())
            .def("barrier", &NativeGroup::barrier, py::call_guard<py::gil_scoped_release>());
        m.def("abort", &abort_world, py::call_guard<py::gil_scoped_release>());
        m.def("initialize", &initialize, "mpi initialize", py::call_guard<py::gil_scoped_release>());
        m.def("finalize", &finalize, "mpi finalize", py::call_guard<py::gil_scoped_release>());
        m.def("get_rank", &get_rank, py::arg("group") = 0,
              py::call_guard<py::gil_scoped_release>());
        m.def("barrier", &barrier, py::call_guard<py::gil_scoped_release>());
        m.def("get_world_size", &get_world_size, py::arg("group") = 0,
              py::call_guard<py::gil_scoped_release>());
    }

} // namespace mpi
