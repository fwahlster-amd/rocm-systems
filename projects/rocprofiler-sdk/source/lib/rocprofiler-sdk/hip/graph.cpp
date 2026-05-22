// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "lib/rocprofiler-sdk/hip/graph.hpp"

#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/tracing/tracing.hpp"

#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/hip/runtime_api_id.h>  // pulls in <hip/amd_detail/hip_api_trace.hpp>

#include <atomic>
#include <deque>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <utility>

namespace rocprofiler
{
namespace hip
{
namespace graph
{
namespace
{
// Process-global map from hipGraphExec_t handle to a stable monotonic id.
// Reads (one per launch) take the shared lock; writes (one per
// hipGraphInstantiate*/hipGraphExecDestroy) take the exclusive lock.
// Map sizes are small (graphs currently in flight; typically <100).
std::shared_mutex                              g_map_mutex;
std::unordered_map<::hipGraphExec_t, uint64_t> g_exec_to_id;
std::atomic<uint64_t>                          g_next_graph_exec_id{1};  // 0 reserved = "not from a graph"

uint64_t
assign_graph_exec_id(::hipGraphExec_t exec)
{
    if(exec == nullptr) return 0;
    auto id = g_next_graph_exec_id.fetch_add(1, std::memory_order_relaxed);
    std::unique_lock lock{g_map_mutex};
    g_exec_to_id[exec] = id;
    return id;
}

void
forget_graph_exec(::hipGraphExec_t exec)
{
    if(exec == nullptr) return;
    std::unique_lock lock{g_map_mutex};
    g_exec_to_id.erase(exec);
}

// Each instantiation of wrap_instantiate has its own static next_func slot
// because the 3 hipGraphInstantiate* APIs have different signatures (different
// trailing Args...). The lambdas have no captures, so they convert to plain
// function pointers via the unary +.
template <typename RetT, typename... Args>
auto
wrap_instantiate(RetT (*next)(::hipGraphExec_t*, Args...))
{
    static auto next_func = next;
    return +[](::hipGraphExec_t* out, Args... args) -> RetT {
        auto ret = next_func(out, std::forward<Args>(args)...);
        if(ret == hipSuccess && out != nullptr && *out != nullptr)
        {
            assign_graph_exec_id(*out);
        }
        return ret;
    };
}

template <typename RetT>
auto
wrap_destroy(RetT (*next)(::hipGraphExec_t))
{
    static auto next_func = next;
    return +[](::hipGraphExec_t exec) -> RetT {
        // Remove the map entry BEFORE invoking the destroy: HIP destroys the
        // handle even on most error paths, so dropping the mapping first is
        // safer than risking a dangling key.
        forget_graph_exec(exec);
        return next_func(exec);
    };
}

// Per-thread stack of active hipGraphLaunch calls. std::deque (not std::vector)
// because launch_state contains std::atomic<uint64_t> which is non-movable;
// std::deque doesn't move existing elements on growth, so references handed
// out by current_launch_state() remain valid as nested launches push.
thread_local std::deque<launch_state> g_launch_stack;

// Forward decl kept so wrap_launch (defined below) resolves the name via
// unqualified lookup within the same anonymous namespace.
void
emit_graph_launch_record(const launch_state& s, rocprofiler_timestamp_t end_ts);
void
emit_graph_launch_record(const launch_state& s, rocprofiler_timestamp_t end_ts)
{
    // GRAPH_LAUNCH is a buffer-only domain (no callback tracing kind defined
    // for it), so use the single-DomainIdx populate_contexts overload that
    // fills only buffered_contexts + external_correlation_ids.
    auto tracing_data_v = tracing::tracing_data{};
    tracing::populate_contexts(ROCPROFILER_BUFFER_TRACING_GRAPH_LAUNCH,
                               /*operation*/ 0u,
                               tracing_data_v.buffered_contexts,
                               tracing_data_v.external_correlation_ids);

    if(tracing_data_v.buffered_contexts.empty()) return;

    // rocprofiler_async_correlation_id_t has 2 fields (internal, external);
    // execute_buffer_record_emplace overwrites correlation_id from its
    // internal_corr_id + external_corr_ids args, so default-construct it here.
    auto record = rocprofiler_buffer_tracing_graph_launch_record_t{
        sizeof(rocprofiler_buffer_tracing_graph_launch_record_t),
        ROCPROFILER_BUFFER_TRACING_GRAPH_LAUNCH,
        /*operation*/ 0u,
        rocprofiler_async_correlation_id_t{},
        s.thread_id,
        s.start_ts,
        end_ts,
        s.agent_id,
        s.queue_id,
        s.graph_exec_id,
        s.dispatch_count};  // launch_state's counter -> record's kernel_dispatch_count

    tracing::execute_buffer_record_emplace(tracing_data_v.buffered_contexts,
                                           s.thread_id,
                                           s.correlation_id,
                                           tracing_data_v.external_correlation_ids,
                                           /*ancestor_corr_id*/ uint64_t{0},
                                           ROCPROFILER_BUFFER_TRACING_GRAPH_LAUNCH,
                                           /*operation*/ 0u,
                                           record);
}

// hipGraphLaunch and hipGraphLaunch_spt share the SAME signature. A naive
// wrap_launch<RetT> template would collapse them into a single instantiation
// and the two static next_func slots would alias. The LaunchApiTag template
// parameter forces distinct instantiations (and thus distinct next_func
// storage) for each API.
enum class LaunchApiTag
{
    hipGraphLaunch,
    hipGraphLaunch_spt
};

template <LaunchApiTag Tag, typename RetT>
auto
wrap_launch(RetT (*next)(::hipGraphExec_t, ::hipStream_t))
{
    static auto next_func = next;
    return +[](::hipGraphExec_t exec, ::hipStream_t stream) -> RetT {
        g_launch_stack.emplace_back();
        auto& s         = g_launch_stack.back();
        s.graph_exec_id = lookup_graph_exec_id(exec);
        if(s.graph_exec_id == 0)
        {
            // Attach-mid-process fallback: rocprofiler may have attached after
            // hipGraphInstantiate ran, so the map has no entry. Assign now so
            // subsequent dispatches in this launch still get a non-zero ID.
            s.graph_exec_id = assign_graph_exec_id(exec);
        }
        s.thread_id      = common::get_tid();
        // NOTE: correlation_id is left 0 for now. The spec's correlation_id-join
        // contract requires reading the HIP API tracing TLS that is built up by
        // the outer HIP wrapper, but no such TLS accessor exists yet in hip.cpp.
        // A follow-up will add the accessor; the GRAPH_LAUNCH record is still
        // emitted, it just won't join cleanly to HIP API records until then.
        s.correlation_id = 0;
        s.start_ts       = rocprofiler_timestamp_t{common::timestamp_ns()};

        auto ret = next_func(exec, stream);

        // Per spec §4.4: emit summary record only on hipSuccess. Always emit
        // on success (including dispatch_count == 0) per §4.2.
        auto end_ts = rocprofiler_timestamp_t{common::timestamp_ns()};
        if(ret == hipSuccess)
        {
            emit_graph_launch_record(s, end_ts);
        }
        // Pop unconditionally — TLS state must always be cleaned up, even on
        // error paths.
        g_launch_stack.pop_back();
        return ret;
    };
}
}  // namespace

void
init()
{
    // The map is default-constructed at static-init; nothing to do here yet.
    // Future tasks may add lifecycle wiring (Task 8 hooks instantiate/destroy
    // wrappers via update_table; this init() exists as a stable entry point).
}

launch_state*
current_launch_state()
{
    return g_launch_stack.empty() ? nullptr : &g_launch_stack.back();
}

uint64_t
lookup_graph_exec_id(::hipGraphExec_t exec)
{
    if(exec == nullptr) return 0;
    std::shared_lock lock{g_map_mutex};
    auto             it = g_exec_to_id.find(exec);
    return it == g_exec_to_id.end() ? 0 : it->second;
}

// Explicit specialization for the HIP runtime dispatch table. Wraps the four
// graph-lifecycle entry points so that:
//   - successful hipGraphInstantiate* assigns a fresh monotonic graph_exec_id
//   - hipGraphExecDestroy removes the map entry
//
// Each install site is guarded with a null check so older HIP runtimes that
// lack one of these fn slots don't NPE.
template <>
void
update_table(::HipDispatchTable* table)
{
    if(table == nullptr) return;
    if(table->hipGraphInstantiate_fn)
        table->hipGraphInstantiate_fn = wrap_instantiate(table->hipGraphInstantiate_fn);
    if(table->hipGraphInstantiateWithFlags_fn)
        table->hipGraphInstantiateWithFlags_fn =
            wrap_instantiate(table->hipGraphInstantiateWithFlags_fn);
    if(table->hipGraphInstantiateWithParams_fn)
        table->hipGraphInstantiateWithParams_fn =
            wrap_instantiate(table->hipGraphInstantiateWithParams_fn);
    if(table->hipGraphExecDestroy_fn)
        table->hipGraphExecDestroy_fn = wrap_destroy(table->hipGraphExecDestroy_fn);
    if(table->hipGraphLaunch_fn)
        table->hipGraphLaunch_fn =
            wrap_launch<LaunchApiTag::hipGraphLaunch>(table->hipGraphLaunch_fn);
    if(table->hipGraphLaunch_spt_fn)
        table->hipGraphLaunch_spt_fn =
            wrap_launch<LaunchApiTag::hipGraphLaunch_spt>(table->hipGraphLaunch_spt_fn);
}

}  // namespace graph
}  // namespace hip
}  // namespace rocprofiler
