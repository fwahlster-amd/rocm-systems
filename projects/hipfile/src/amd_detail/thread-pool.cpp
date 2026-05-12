/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "thread-pool.h"

#include <memory>
#include <oneapi/tbb/task_arena.h>
#include <oneapi/tbb/task_group.h>
#include <stdexcept>
#include <utility>

namespace hipFile {

struct ThreadPool::ThreadPoolStorage {
    tbb::task_arena arena;
    tbb::task_group tasks;
};

ThreadPool::ThreadPool() : storage{std::make_unique<ThreadPoolStorage>()}
{
}

ThreadPool::~ThreadPool() noexcept
{
    try {
        wait();
    }
    catch (...) {
        // Explicit wait() preserves task failures. Destructors cannot report them.
    }
}

void
ThreadPool::enqueue(std::function<void()> work)
{
    if (!work) {
        throw std::invalid_argument("Thread pool work item cannot be empty");
    }

    ThreadPoolStorage &pool = *storage;
    pool.arena.execute([&pool, task = std::move(work)]() mutable { pool.tasks.run(std::move(task)); });
}

void
ThreadPool::wait()
{
    ThreadPoolStorage &pool = *storage;
    pool.arena.execute([&pool]() { pool.tasks.wait(); });
}

std::size_t
ThreadPool::threadCount() const noexcept
{
    return static_cast<std::size_t>(storage->arena.max_concurrency());
}

}
