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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <chrono>
namespace py = pybind11;

#include <ATen/cuda/CUDAEvent.h>
#include <c10/core/DeviceGuard.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <deque>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <stdio.h>

// Every MPI call takes mpi::MpiCallLock (see GraphProxy).
#define MPICHECK(cmd)                                                        \
    do                                                                       \
    {                                                                        \
        int e;                                                               \
        {                                                                    \
            mpi::MpiCallLock mpi_call_lock_;                                 \
            e = cmd;                                                         \
        }                                                                    \
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

#define CUCHECK(cmd)                                                                   \
    do                                                                                 \
    {                                                                                  \
        CUresult r = cmd;                                                              \
        if (r != CUDA_SUCCESS)                                                         \
        {                                                                              \
            const char *msg = nullptr;                                                 \
            cuGetErrorString(r, &msg);                                                 \
            TORCH_CHECK(false, "MCR-DL CUDA driver error ", msg ? msg : "", " in ", #cmd); \
        }                                                                              \
    } while (0)

namespace mpi
{

    // Set while a GraphProxy thread runs. MPI is then called from two threads
    // (MPI_THREAD_SERIALIZED), so every MPI call holds mpi_mutex().
    inline std::atomic<bool> &graph_proxy_active()
    {
        static std::atomic<bool> active{false};
        return active;
    }

    inline std::mutex &mpi_mutex()
    {
        static std::mutex mutex;
        return mutex;
    }

    struct MpiCallLock
    {
        std::unique_lock<std::mutex> lock;
        MpiCallLock()
        {
            if (graph_proxy_active().load(std::memory_order_acquire))
                lock = std::unique_lock<std::mutex>(mpi_mutex());
        }
    };

    // With a GraphProxy running, wait for the device before a caller's MPI
    // call: a replayed graph may still need the proxy for its all-reduces, and
    // a caller blocked in MPI would hold mpi_mutex() while peers wait on them.
    inline void sync_device_for_proxy()
    {
        if (graph_proxy_active().load(std::memory_order_acquire))
            CUDACHECK(cudaDeviceSynchronize());
    }

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
        sync_device_for_proxy();
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

    void stop_graph_proxies();

    void finalize()
    {
        stop_graph_proxies();
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

    // One thread: make prior writes visible system-wide, ring the doorbell with
    // the slot id, spin until the proxy writes the same id to "done", reset it.
    // Both words are pinned host memory mapped into the device address space.
    // Loaded with cuModuleLoadData, so building MCR-DL needs no nvcc.
    constexpr const char *kSignalWaitPtx = R"PTX(
.version 7.0
.target sm_70
.address_size 64

.visible .entry mcr_dl_proxy_signal_wait(
    .param .u64 doorbell,
    .param .u64 done,
    .param .u32 id
)
{
    .reg .u64 %rd<3>;
    .reg .u32 %r<3>;
    .reg .pred %p<2>;
    ld.param.u64 %rd1, [doorbell];
    ld.param.u64 %rd2, [done];
    ld.param.u32 %r1, [id];
    fence.sc.sys;
    st.release.sys.global.u32 [%rd1], %r1;
WAIT:
    ld.acquire.sys.global.u32 %r2, [%rd2];
    setp.ne.u32 %p1, %r2, %r1;
    @%p1 bra WAIT;
    st.relaxed.sys.global.u32 [%rd2], 0;
    ret;
}
)PTX";

    // Runs the MPI all-reduces recorded in CUDA graphs, so a graph can contain
    // blocking MPI. At each recorded all-reduce the graph runs a one-thread
    // kernel (kSignalWaitPtx) that writes the call's slot id to a pinned host
    // doorbell, waits until a pinned host "done" word holds the same id, and
    // resets it.
    // This thread polls the doorbell, runs MPI on the slot's buffers, which keep
    // their CUDA graph pool addresses on every replay, and then writes "done".
    // The kernel runs after the producing kernels and fences before the
    // doorbell, so the buffers are ready for MPI when it lands. One proxy per device, never destroyed: its thread stops
    // in finalize().
    class GraphProxy
    {
    public:
        struct Slot
        {
            MPI_Comm comm;
            const void *send; // MPI_IN_PLACE for in-place all-reduces
            void *recv;
            int count;
            MPI_Datatype type;
            MPI_Op op;
        };

        // Starts the proxy; call outside CUDA graph capture.
        static GraphProxy &get(int device)
        {
            std::lock_guard<std::mutex> guard(registry_mutex());
            auto &proxy = registry()[device];
            if (proxy == nullptr)
                proxy = new GraphProxy(device);
            return *proxy;
        }

        static GraphProxy *find(int device)
        {
            std::lock_guard<std::mutex> guard(registry_mutex());
            auto it = registry().find(device);
            return it == registry().end() ? nullptr : it->second;
        }

        static void stop_all()
        {
            std::lock_guard<std::mutex> guard(registry_mutex());
            for (auto &entry : registry())
                entry.second->stop();
            graph_proxy_active().store(false, std::memory_order_release);
        }

        // Ids start at 1; 0 means an empty doorbell.
        uint32_t add(const Slot &slot)
        {
            std::lock_guard<std::mutex> guard(slots_mutex_);
            TORCH_CHECK(slots_.size() < std::numeric_limits<uint32_t>::max() - 1,
                        "MCR-DL graph proxy slot ids exhausted");
            slots_.push_back(slot);
            return static_cast<uint32_t>(slots_.size());
        }

        // Steady-clock ns when each request was seen and when its MPI call
        // returned, for benchmarks; clears the record.
        std::pair<std::vector<int64_t>, std::vector<int64_t>> take_times()
        {
            std::lock_guard<std::mutex> guard(times_mutex_);
            auto times = std::make_pair(std::move(seen_ns_), std::move(done_ns_));
            seen_ns_.clear();
            done_ns_.clear();
            return times;
        }

        // Records the all-reduce with this slot id on a (capturing) stream.
        void enqueue(cudaStream_t stream, uint32_t id)
        {
            CUstream s = reinterpret_cast<CUstream>(stream);
            if (use_memops_)
            {
                CUCHECK(cuStreamWriteValue32(s, doorbell_, id, CU_STREAM_WRITE_VALUE_DEFAULT));
                CUCHECK(cuStreamWaitValue32(s, done_, id, CU_STREAM_WAIT_VALUE_EQ));
                CUCHECK(cuStreamWriteValue32(s, done_, 0, CU_STREAM_WRITE_VALUE_DEFAULT));
                return;
            }
            CUdeviceptr doorbell = doorbell_, done = done_;
            void *args[] = {&doorbell, &done, &id};
            CUCHECK(cuLaunchKernel(signal_wait_, 1, 1, 1, 1, 1, 1, 0, s, args, nullptr));
        }

    private:
        using Clock = std::chrono::steady_clock;
        static constexpr int kDoneWord = 32; // 128 bytes after the doorbell

        explicit GraphProxy(int device) : device_(device)
        {
            c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device));
            void *host = nullptr;
            CUDACHECK(cudaHostAlloc(&host, 4096, cudaHostAllocMapped | cudaHostAllocPortable));
            std::memset(host, 0, 4096);
            words_ = static_cast<uint32_t *>(host);
            void *dev = nullptr;
            CUDACHECK(cudaHostGetDevicePointer(&dev, host, 0));
            doorbell_ = reinterpret_cast<CUdeviceptr>(dev);
            done_ = doorbell_ + kDoneWord * sizeof(uint32_t);
            // The signal kernel by default; MCR_DL_GRAPH_PROXY_SIGNAL=memop uses
            // three stream memory operations instead (slower GPU-side wait).
            const char *signal = std::getenv("MCR_DL_GRAPH_PROXY_SIGNAL");
            use_memops_ = signal != nullptr && std::string(signal) == "memop";
            if (!use_memops_)
            {
                CUCHECK(cuModuleLoadData(&module_, kSignalWaitPtx));
                CUCHECK(cuModuleGetFunction(&signal_wait_, module_, "mcr_dl_proxy_signal_wait"));
            }
            // Set before the thread exists: from here on MPI calls are locked.
            graph_proxy_active().store(true, std::memory_order_release);
            thread_ = std::thread([this] { run(); });
            // The thread's CUDA setup must not overlap a later graph capture.
            while (!ready_.load(std::memory_order_acquire))
                std::this_thread::yield();
        }

        static std::map<int, GraphProxy *> &registry()
        {
            static std::map<int, GraphProxy *> proxies;
            return proxies;
        }

        static std::mutex &registry_mutex()
        {
            static std::mutex mutex;
            return mutex;
        }

        void stop()
        {
            stop_.store(true, std::memory_order_relaxed);
            if (thread_.joinable())
                thread_.join();
        }

        void run()
        {
            CUDACHECK(cudaSetDevice(device_));
            ready_.store(true, std::memory_order_release);
            auto last_work = Clock::now();
            unsigned spins = 0;
            while (!stop_.load(std::memory_order_relaxed))
            {
                const uint32_t id = __atomic_load_n(&words_[0], __ATOMIC_ACQUIRE);
                if (id == 0)
                {
                    // Spin while decode steps run back to back; after 20 ms
                    // without work, poll every 50 us instead.
                    if (++spins % 4096 == 0 &&
                        Clock::now() - last_work > std::chrono::milliseconds(20))
                        std::this_thread::sleep_for(std::chrono::microseconds(50));
#if defined(__x86_64__)
                    __builtin_ia32_pause();
#endif
                    continue;
                }
                // The graph posts its next request only after this one's "done".
                __atomic_store_n(&words_[0], 0u, __ATOMIC_RELAXED);
                const int64_t seen = now_ns();
                Slot slot;
                {
                    std::lock_guard<std::mutex> guard(slots_mutex_);
                    if (id > slots_.size())
                    {
                        fprintf(stderr, "MCR-DL graph proxy: unknown slot %u\n", id);
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }
                    slot = slots_[id - 1];
                }
                MPICHECK(MPI_Allreduce(slot.send, slot.recv, slot.count, slot.type,
                                       slot.op, slot.comm));
                __atomic_store_n(&words_[kDoneWord], id, __ATOMIC_RELEASE);
                record(seen, now_ns());
                last_work = Clock::now();
            }
        }

        static int64_t now_ns()
        {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(
                       Clock::now().time_since_epoch())
                .count();
        }

        void record(int64_t seen, int64_t done)
        {
            std::lock_guard<std::mutex> guard(times_mutex_);
            if (seen_ns_.size() < (1u << 20))
            {
                seen_ns_.push_back(seen);
                done_ns_.push_back(done);
            }
        }

        const int device_;
        bool use_memops_ = false;
        CUmodule module_ = nullptr;
        CUfunction signal_wait_ = nullptr;
        std::mutex times_mutex_;
        std::vector<int64_t> seen_ns_, done_ns_;
        uint32_t *words_ = nullptr;
        CUdeviceptr doorbell_ = 0, done_ = 0;
        std::mutex slots_mutex_;
        std::deque<Slot> slots_;
        std::atomic<bool> stop_{false}, ready_{false};
        std::thread thread_;
    };

    void stop_graph_proxies() { GraphProxy::stop_all(); }

    class NativeGroup;

    // Groups by id, for the C++ PyTorch ops below, which take an int group id
    // instead of a Python object. Entries are cleared when a group is destroyed.
    inline std::vector<NativeGroup *> &group_registry()
    {
        static std::vector<NativeGroup *> groups;
        return groups;
    }

    // One communicator for a set of existing MPI ranks. Tensor operations use
    // this group; it does not create processes or worker threads.
    class NativeGroup
    {
    public:
        NativeGroup(const NativeGroup &) = delete;
        NativeGroup &operator=(const NativeGroup &) = delete;

        ~NativeGroup()
        {
            auto &groups = group_registry();
            if (id_ >= 0 && id_ < static_cast<int>(groups.size()))
                groups[id_] = nullptr;
        }

        // Id accepted by the mcr_dl::all_reduce_inplace / all_reduce_out PyTorch ops.
        int id() const { return id_; }

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
            id_ = static_cast<int>(group_registry().size());
            group_registry().push_back(this);
        }

        // MPI_Finalize owns communicator cleanup. Python object destruction is not
        // ordered across ranks, so it must not perform collective MPI_Comm_free.
        int rank() const { return rank_; }
        int size() const { return size_; }

        void all_reduce(torch::Tensor tensor, int op, bool is_prof = false, Stream stream = std::nullopt)
        {
            all_reduce_impl(tensor, tensor, op, is_prof, stream);
        }

        void all_reduce(torch::Tensor output, torch::Tensor input, int op,
                        bool is_prof = false, Stream stream = std::nullopt)
        {
            check_buffers(output, input);
            TORCH_CHECK(output.numel() == input.numel(),
                        "MCR-DL all_reduce requires equally sized input and output tensors");
            all_reduce_impl(output, input, op, is_prof, stream);
        }

        // All-reduce that can be recorded in a CUDA graph on the current stream
        // (output may be input for MPI_IN_PLACE). During capture it records a
        // GraphProxy all-reduce: both buffers must keep their addresses for as
        // long as the graph is replayed, as CUDA graph pool memory does.
        // Otherwise it starts this device's GraphProxy and runs the blocking
        // all-reduce, so call it once before capturing.
        void all_reduce_graph(const torch::Tensor &output, const torch::Tensor &input, int op)
        {
            const bool in_place = output.is_same(input);
            if (in_place)
            {
                TORCH_CHECK(input.defined() && input.is_contiguous() && !input.is_sparse() &&
                                (input.is_cpu() || input.is_cuda()),
                            "MCR-DL MPI requires a dense contiguous CPU or CUDA tensor");
            }
            else
            {
                check_buffers(output, input);
                TORCH_CHECK(output.numel() == input.numel(),
                            "MCR-DL all_reduce_graph requires equally sized input and output tensors");
            }
            if (input.is_cpu())
            {
                all_reduce_impl(output, input, op, false, std::nullopt);
                return;
            }
            const int device = input.get_device();
            c10::cuda::CUDAGuard guard(input.device());
            const auto stream = c10::cuda::getCurrentCUDAStream(device);
            cudaStreamCaptureStatus status;
            CUDACHECK(cudaStreamIsCapturing(stream.stream(), &status));
            if (status == cudaStreamCaptureStatusNone)
            {
                GraphProxy::get(device);
                all_reduce_impl(output, input, op, false, std::nullopt);
                return;
            }
            TORCH_CHECK(status == cudaStreamCaptureStatusActive,
                        "MCR-DL all_reduce_graph: the CUDA graph capture was invalidated");
            GraphProxy *proxy = GraphProxy::find(device);
            TORCH_CHECK(proxy != nullptr,
                        "MCR-DL all_reduce_graph must run once outside CUDA graph capture "
                        "(this starts the graph proxy) before it is captured");
            const uint32_t id = proxy->add({comm_, in_place ? MPI_IN_PLACE : input.data_ptr(),
                                            output.data_ptr(), count(input),
                                            get_mpi_datatype(input.scalar_type()),
                                            reduction_op(op)});
            proxy->enqueue(stream.stream(), id);
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
            sync_device_for_proxy();
            MPICHECK(MPI_Barrier(comm_));
        }

    private:
        using Clock = std::chrono::steady_clock;
        MPI_Comm comm_;
        int rank_, size_;
        int id_ = -1;
        // Created on first use; PyTorch's event wrapper handles device-aware cleanup.
        std::map<int, at::cuda::CUDAEvent> ready_events_;

        void all_reduce_impl(const torch::Tensor &output, const torch::Tensor &input,
                             int op, bool is_prof, const Stream &stream)
        {
            c10::OptionalDeviceGuard guard;
            prepare(input, guard, stream);
            const auto reduction = reduction_op(op);
            const auto datatype = get_mpi_datatype(input.scalar_type());
            const auto elements = count(input);
            const bool in_place = output.is_same(input);
            const auto begin = is_prof ? Clock::now() : Clock::time_point{};
            MPICHECK(MPI_Allreduce(in_place ? MPI_IN_PLACE : input.data_ptr(),
                                   output.data_ptr(), elements, datatype, reduction, comm_));
            if (is_prof)
                print_elapsed(in_place ? "all_reduce" : "all_reduce_out", begin, input);
        }

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
                sync_device_for_proxy();
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
            if (graph_proxy_active().load(std::memory_order_acquire))
            {
                // Also covers this stream (see sync_device_for_proxy).
                CUDACHECK(cudaDeviceSynchronize());
                return;
            }
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
            .def("id", &NativeGroup::id,
                 "Group id for torch.ops.mcr_dl.all_reduce_inplace and torch.ops.mcr_dl.all_reduce_out.")
            .def("rank", &NativeGroup::rank)
            .def("size", &NativeGroup::size)
            .def("all_reduce", py::overload_cast<torch::Tensor, int, bool, Stream>(&NativeGroup::all_reduce),
                 "In-place reduction; is_prof prints MPI-only latency on group rank zero.",
                 py::arg("tensor"), py::arg("op"), py::arg("is_prof") = false,
                 py::kw_only(), py::arg("stream") = py::none(),
                 py::call_guard<py::gil_scoped_release>())
            .def("all_reduce", py::overload_cast<torch::Tensor, torch::Tensor, int, bool, Stream>(&NativeGroup::all_reduce),
                 "Reduce input into a distinct output of equal size, dtype and device.",
                 py::arg("output"), py::arg("input"), py::arg("op"), py::arg("is_prof") = false,
                 py::kw_only(), py::arg("stream") = py::none(),
                 py::call_guard<py::gil_scoped_release>())
            .def("all_reduce_out", py::overload_cast<torch::Tensor, torch::Tensor, int, bool, Stream>(&NativeGroup::all_reduce),
                 "Reduce input into a distinct output of equal size, dtype and device. "
                 "Buffers must not overlap; is_prof prints MPI-only latency on group rank zero.",
                 py::arg("output"), py::arg("input"), py::arg("op"), py::arg("is_prof") = false,
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
        m.def("graph_proxy_take_times", [](int device) {
                  GraphProxy *proxy = GraphProxy::find(device);
                  return proxy ? proxy->take_times()
                               : std::make_pair(std::vector<int64_t>{}, std::vector<int64_t>{});
              },
              "Benchmarks: (seen_ns, mpi_done_ns) per graph-proxy request since the last call.");
        m.def("get_rank", &get_rank, py::arg("group") = 0,
              py::call_guard<py::gil_scoped_release>());
        m.def("barrier", &barrier, py::call_guard<py::gil_scoped_release>());
        m.def("get_world_size", &get_world_size, py::arg("group") = 0,
              py::call_guard<py::gil_scoped_release>());
    }

    // ---- PyTorch operators -------------------------------------------------
    // torch.ops.mcr_dl.all_reduce_inplace / all_reduce_out reach MPI through the
    // dispatcher alone: no Python implementation and no pybind argument parsing
    // per call. They use the tensor device's current CUDA stream, like the
    // NativeGroup methods called with stream=None.

    NativeGroup &lookup_group(int64_t id)
    {
        auto &groups = group_registry();
        TORCH_CHECK(id >= 0 && id < static_cast<int64_t>(groups.size()) && groups[id] != nullptr,
                    "MCR-DL: no NativeGroup with id ", id);
        return *groups[id];
    }

    // Release the GIL while MPI blocks, as the pybind bindings do, so other
    // Python threads keep running. Callers without the GIL run unchanged.
    template <typename F>
    void without_gil(F &&fn)
    {
        if (Py_IsInitialized() && PyGILState_Check())
        {
            py::gil_scoped_release release;
            fn();
        }
        else
        {
            fn();
        }
    }

    void all_reduce_inplace_op(at::Tensor &tensor, int64_t group, int64_t op)
    {
        NativeGroup &g = lookup_group(group);
        without_gil([&] { g.all_reduce(tensor, static_cast<int>(op)); });
    }

    void all_reduce_out_op(const at::Tensor &input, at::Tensor &output, int64_t group, int64_t op)
    {
        NativeGroup &g = lookup_group(group);
        without_gil([&] { g.all_reduce(output, input, static_cast<int>(op)); });
    }

    at::Tensor all_reduce_graph_op(const at::Tensor &input, int64_t group, int64_t op)
    {
        NativeGroup &g = lookup_group(group);
        at::Tensor output = at::empty_like(input, at::MemoryFormat::Contiguous);
        without_gil([&] { g.all_reduce_graph(output, input, static_cast<int>(op)); });
        return output;
    }

    void all_reduce_graph_inplace_op(at::Tensor &tensor, int64_t group, int64_t op)
    {
        NativeGroup &g = lookup_group(group);
        without_gil([&] { g.all_reduce_graph(tensor, tensor, static_cast<int>(op)); });
    }

    at::Tensor all_reduce_graph_meta(const at::Tensor &input, int64_t, int64_t)
    {
        return at::empty_like(input, at::MemoryFormat::Contiguous);
    }

    // Shape-only kernels for torch.compile tracing: the ops only mutate.
    void all_reduce_inplace_meta(at::Tensor &, int64_t, int64_t) {}
    void all_reduce_out_meta(const at::Tensor &, at::Tensor &, int64_t, int64_t) {}
    void all_reduce_graph_inplace_meta(at::Tensor &, int64_t, int64_t) {}

} // namespace mpi

TORCH_LIBRARY(mcr_dl, m)
{
    m.def("all_reduce_inplace(Tensor(a!) tensor, int group, int op=0) -> ()");
    m.def("all_reduce_out(Tensor input, Tensor(a!) output, int group, int op=0) -> ()");
    // Recordable in CUDA graphs (see NativeGroup::all_reduce_graph).
    m.def("all_reduce_graph(Tensor input, int group, int op=0) -> Tensor");
    m.def("all_reduce_graph_inplace(Tensor(a!) tensor, int group, int op=0) -> ()");
}

TORCH_LIBRARY_IMPL(mcr_dl, CUDA, m)
{
    m.impl("all_reduce_inplace", &mpi::all_reduce_inplace_op);
    m.impl("all_reduce_out", &mpi::all_reduce_out_op);
    m.impl("all_reduce_graph", &mpi::all_reduce_graph_op);
    m.impl("all_reduce_graph_inplace", &mpi::all_reduce_graph_inplace_op);
}

TORCH_LIBRARY_IMPL(mcr_dl, CPU, m)
{
    m.impl("all_reduce_inplace", &mpi::all_reduce_inplace_op);
    m.impl("all_reduce_out", &mpi::all_reduce_out_op);
    m.impl("all_reduce_graph", &mpi::all_reduce_graph_op);
    m.impl("all_reduce_graph_inplace", &mpi::all_reduce_graph_inplace_op);
}

TORCH_LIBRARY_IMPL(mcr_dl, Meta, m)
{
    m.impl("all_reduce_inplace", &mpi::all_reduce_inplace_meta);
    m.impl("all_reduce_out", &mpi::all_reduce_out_meta);
    m.impl("all_reduce_graph", &mpi::all_reduce_graph_meta);
    m.impl("all_reduce_graph_inplace", &mpi::all_reduce_graph_inplace_meta);
}
