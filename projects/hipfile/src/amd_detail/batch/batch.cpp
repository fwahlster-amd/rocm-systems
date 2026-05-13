/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "batch.h"
#include "buffer.h"
#include "context.h"
#include "file.h"
#include "hipfile.h"
#include "state.h"
#include "thread-pool.h"

#include <chrono>
#include <cstddef>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace hipFile {

namespace {

    bool is_terminal_status(hipFileStatus_t status) noexcept
    {
        switch (status) {
            case hipFileComplete:
            case hipFileFailed:
            case hipFileCanceled:
            case hipFileInvalid:
            case hipFileTimeout:
                return true;
            case hipFileWaiting:
            case hipFilePending:
                return false;
            default:
                return false;
        }
    }

    bool is_zero_timeout(const struct timespec *timeout) noexcept
    {
        return timeout != nullptr && timeout->tv_sec == 0 && timeout->tv_nsec == 0;
    }

    void validate_timeout(const struct timespec *timeout)
    {
        if (timeout == nullptr) {
            return;
        }
        if (timeout->tv_sec < 0 || timeout->tv_nsec < 0 || timeout->tv_nsec >= 1000000000L) {
            throw std::invalid_argument("Invalid batch status timeout");
        }
    }

    std::chrono::steady_clock::time_point timeout_deadline(const struct timespec *timeout)
    {
        return std::chrono::steady_clock::now() + std::chrono::seconds{timeout->tv_sec} +
               std::chrono::nanoseconds{timeout->tv_nsec};
    }

}

BatchOperation::BatchOperation(std::unique_ptr<const hipFileIOParams_t> params,
                               std::shared_ptr<IBuffer> _buffer, std::shared_ptr<IFile> _file)
    : io_params{std::move(params)}, buffer{_buffer}, file{_file}
{
    // Cookie allows the user to track which operation caused the error.
    // It would be ideal if this could be passed as a member within the exception.

    // Check Buffer parameters
    if (io_params->u.batch.devPtr_base != buffer->getBuffer()) {
        throw std::invalid_argument("Buffer does not match buffer specified in io_params.");
    }
    if (io_params->u.batch.devPtr_offset < 0) {
        std::stringstream msg;
        msg << "Negative buffer offset specified. Value: " << io_params->u.batch.devPtr_offset;
        msg << ". Cookie: " << io_params->cookie;
        throw std::invalid_argument(msg.str());
    }
    if (buffer->getLength() <= static_cast<size_t>(io_params->u.batch.devPtr_offset)) {
        std::stringstream msg;
        msg << "Buffer offset exceeds the size of the buffer. Size: " << buffer->getLength();
        msg << ". Value: " << io_params->u.batch.devPtr_offset << ". Cookie: " << io_params->cookie;
        throw std::invalid_argument(msg.str());
    }
    if (buffer->getLength() - static_cast<size_t>(io_params->u.batch.devPtr_offset) <
        io_params->u.batch.size) {
        std::stringstream msg;
        msg << "IO Size exceeds the size of the buffer & offset. Buffer size: " << buffer->getLength();
        msg << ". Buffer offset: " << io_params->u.batch.devPtr_offset
            << ". IO size: " << io_params->u.batch.size;
        msg << ". Cookie: " << io_params->cookie;
        throw std::invalid_argument(msg.str());
    }

    // Check File parameters
    if (io_params->fh != file->handle()) {
        throw std::invalid_argument("File does not match handle specified in io_params.");
    }
    if (io_params->u.batch.file_offset < 0) {
        std::stringstream msg;
        msg << "Negative file offset specified. Value: " << io_params->u.batch.file_offset;
        msg << ". Cookie: " << io_params->cookie;
        throw std::invalid_argument(msg.str());
    }

    // Check OpCode
    if (io_params->opcode != hipFileBatchRead && io_params->opcode != hipFileBatchWrite) {
        std::stringstream msg;
        msg << "Bad opcode specified. Value: " << io_params->opcode;
        msg << ". Cookie: " << io_params->cookie;
        throw std::invalid_argument(msg.str());
    }

    // Check Batch Mode
    if (io_params->mode != hipFileBatch) {
        std::stringstream msg;
        msg << "Invalid Batch mode specified. Value: " << io_params->mode;
        msg << ". Cookie: " << io_params->cookie;
        throw std::invalid_argument(msg.str());
    }
}

void
BatchOperation::mark_pending()
{
    std::lock_guard<std::mutex> lock{state_mutex};

    if (status == hipFileWaiting) {
        status = hipFilePending;
    }
}

void
BatchOperation::cancel()
{
    std::lock_guard<std::mutex> lock{state_mutex};

    if (status == hipFilePending) {
        status = hipFileCanceled;
    }
}

hipFileStatus_t
BatchOperation::get_status() const
{
    std::lock_guard<std::mutex> lock{state_mutex};
    return status;
}

ssize_t
BatchOperation::get_result() const
{
    std::lock_guard<std::mutex> lock{state_mutex};
    return ret;
}

hipFileIOEvents_t
BatchOperation::event() const
{
    std::lock_guard<std::mutex> lock{state_mutex};
    return {io_params->cookie, status, static_cast<size_t>(ret)};
}

#ifdef AIS_TESTING
void
BatchOperation::set_status_for_testing(hipFileStatus_t new_status, ssize_t result)
{
    std::lock_guard<std::mutex> lock{state_mutex};
    status = new_status;
    ret    = result;
}
#endif

void
BatchOperation::run()
{
    {
        std::lock_guard<std::mutex> lock{state_mutex};
        if (status == hipFileCanceled) {
            return;
        }
        if (status == hipFileWaiting) {
            status = hipFilePending;
        }
    }

    ssize_t result = 0;
    if (io_params->opcode == hipFileBatchRead) {
        result = hipFileRead(io_params->fh, io_params->u.batch.devPtr_base, io_params->u.batch.size,
                             io_params->u.batch.file_offset, io_params->u.batch.devPtr_offset);
    }
    else {
        result = hipFileWrite(io_params->fh, io_params->u.batch.devPtr_base, io_params->u.batch.size,
                              io_params->u.batch.file_offset, io_params->u.batch.devPtr_offset);
    }

    std::lock_guard<std::mutex> lock{state_mutex};
    ret = result;
    if (status != hipFileCanceled) {
        status = result >= 0 ? hipFileComplete : hipFileFailed;
    }
}

BatchContext::BatchContext(unsigned _capacity) : capacity{_capacity}
{
    if (_capacity == 0) {
        throw std::invalid_argument("Batch capacity cannot be zero");
    }
    if (_capacity > MAX_SIZE) {
        throw std::invalid_argument("Batch capacity is limited to " + std::to_string(MAX_SIZE));
    }
}

unsigned
BatchContext::get_capacity() const noexcept
{
    return capacity;
}

void
BatchContext::submit_operations(const hipFileIOParams_t *params, unsigned num_params)
{
    std::unique_lock<std::shared_mutex> _ulock{context_mutex};

    // Check num_params first before doing anything else
    if (num_params > capacity - outstanding_ops.size()) {
        std::stringstream msg;
        msg << "Submission exceeds the capacity of this context. Number of ops submitted: ";
        msg << num_params << ". Context capacity: " << capacity << ". Current outstanding ops: ";
        msg << outstanding_ops.size();
        throw std::invalid_argument(msg.str());
    }

    std::vector<std::shared_ptr<BatchOperation>> pending_ops{};

    // It would be more performant to be able to perform multiple lookups
    // rather than waiting to lock the DriverState lock for each lookup.
    for (unsigned i = 0; i < num_params; i++) {
        // Make a copy of the params so another thread cannot modify the operation.
        auto param_copy = std::make_unique<const hipFileIOParams_t>(params[i]);
        // flags currently unused. Ambiguous if flags in hipFileBatchIOSubmit is for buffer or
        // file flags.
        auto [_file, _buffer] =
            Context<DriverState>::get()->getFileAndBuffer(param_copy->fh, param_copy->u.batch.devPtr_base);
        auto op = std::make_shared<BatchOperation>(std::move(param_copy), _buffer, _file);

        pending_ops.push_back(std::move(op));
    }

    // All submitted operations look valid at this point. Accept them.
    for (const auto &op : pending_ops) {
        op->mark_pending();
    }
    outstanding_ops.insert(pending_ops.begin(), pending_ops.end());

    for (const auto &op : pending_ops) {
        Context<IThreadPool>::get()->enqueue([this, op]() {
            op->run();
            status_cv.notify_all();
        });
    }
}

void
BatchContext::get_status(unsigned min_nr, unsigned *nr, hipFileIOEvents_t *iocbp, struct timespec *timeout)
{
    if (nr == nullptr) {
        throw std::invalid_argument("Number of events cannot be null");
    }
    if (*nr > 0 && iocbp == nullptr) {
        throw std::invalid_argument("Event buffer cannot be null");
    }
    if (min_nr > *nr) {
        throw std::invalid_argument("Minimum event count exceeds event buffer capacity");
    }
    validate_timeout(timeout);

    const unsigned event_capacity = *nr;
    *nr                           = 0;

    std::unique_lock<std::shared_mutex> lock{context_mutex};

    auto terminal_count = [this]() {
        unsigned count = 0;
        for (const auto &op : outstanding_ops) {
            if (is_terminal_status(op->get_status())) {
                count++;
            }
        }
        return count;
    };

    auto collect_terminal_events = [this, event_capacity, nr, iocbp]() {
        unsigned copied = 0;
        for (auto op_iter = outstanding_ops.begin();
             op_iter != outstanding_ops.end() && copied < event_capacity;) {
            hipFileIOEvents_t event = (*op_iter)->event();
            if (!is_terminal_status(event.status)) {
                ++op_iter;
                continue;
            }

            iocbp[copied++] = event;
            op_iter         = outstanding_ops.erase(op_iter);
        }
        *nr = copied;
        return copied;
    };

    if (outstanding_ops.empty() || event_capacity == 0) {
        return;
    }

    if (min_nr == 0 || terminal_count() >= min_nr || is_zero_timeout(timeout)) {
        collect_terminal_events();
        return;
    }

    auto ready = [&terminal_count, min_nr, this]() {
        return terminal_count() >= min_nr || outstanding_ops.empty();
    };

    if (timeout == nullptr) {
        status_cv.wait(lock, ready);
    }
    else {
        status_cv.wait_until(lock, timeout_deadline(timeout), ready);
    }

    collect_terminal_events();
}

#ifdef AIS_TESTING
void
BatchContext::add_operation_for_testing(std::shared_ptr<BatchOperation> op)
{
    std::unique_lock<std::shared_mutex> ulock{context_mutex};
    outstanding_ops.insert(std::move(op));
    status_cv.notify_all();
}

size_t
BatchContext::outstanding_count_for_testing() const
{
    std::shared_lock<std::shared_mutex> slock{context_mutex};
    return outstanding_ops.size();
}
#endif

void
BatchContextMap::clear()
{
    std::unique_lock<std::shared_mutex> ulock{batch_mutex};
    active_contexts.clear();
}

hipFileBatchHandle_t
BatchContextMap::createContext(unsigned capacity)
{
    auto                 context = std::shared_ptr<IBatchContext>{new BatchContext{capacity}};
    hipFileBatchHandle_t handle  = context.get();

    // Should not need to worry about duplicate keys unless the application
    // somehow deallocates this handle...

    std::unique_lock<std::shared_mutex> ulock{batch_mutex};
    active_contexts[handle] = std::move(context);
    return handle;
}

void
BatchContextMap::destroyContext(hipFileBatchHandle_t handle)
{
    std::unique_lock<std::shared_mutex> ulock{batch_mutex};

    auto context = active_contexts.find(handle);
    if (context == active_contexts.end()) {
        throw InvalidBatchHandle();
    }
    // TODO: Check for outstanding operations.
    // TODO: Attempt to cancel any outstanding operations.
    // TODO: Determine if we return unconditionally or require
    //       outstanding ops to terminate first.
    active_contexts.erase(handle);
}

std::shared_ptr<IBatchContext>
BatchContextMap::get(hipFileBatchHandle_t handle)
{
    // NOTE: This mutex only protects the map, so we'll
    //       also need to protect the data
    std::shared_lock<std::shared_mutex> slock{batch_mutex};

    auto context = active_contexts.find(handle);
    if (context == active_contexts.end()) {
        throw InvalidBatchHandle();
    }
    return context->second;
}

}
