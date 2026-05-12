/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "context.h"
#include "hipfile-warnings.h"
#include "mthread-pool.h"
#include "thread-pool.h"

#include <atomic>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace hipFile;
using ::testing::StrictMock;

// Put tests inside the macros to suppress the global constructor
// warnings
HIPFILE_WARN_NO_GLOBAL_CTOR_OFF

TEST(HipFileThreadPool, ConstructorUsesDefaultArenaConcurrency)
{
    ThreadPool pool{};
    ASSERT_GT(pool.threadCount(), 0);
}

TEST(HipFileThreadPool, ContextDefaultUsesThreadPool)
{
    IThreadPool *thread_pool = Context<IThreadPool>::get();

    ASSERT_NE(dynamic_cast<ThreadPool *>(thread_pool), nullptr);
    ASSERT_GT(thread_pool->threadCount(), 0);
}

TEST(HipFileThreadPool, ContextCanUseMockThreadPool)
{
    StrictMock<MThreadPool> thread_pool;

    EXPECT_CALL(thread_pool, wait);

    Context<IThreadPool>::get()->wait();
}

TEST(HipFileThreadPool, EnqueuedWorkRuns)
{
    ThreadPool       pool{};
    std::atomic<int> completed{0};

    pool.enqueue([&completed]() { completed.fetch_add(1, std::memory_order_relaxed); });
    pool.enqueue([&completed]() { completed.fetch_add(1, std::memory_order_relaxed); });

    pool.wait();

    ASSERT_EQ(completed.load(std::memory_order_relaxed), 2);
}

TEST(HipFileThreadPool, DestructorWaitsForEnqueuedWork)
{
    std::atomic<int> completed{0};

    {
        ThreadPool pool{};
        pool.enqueue([&completed]() { completed.fetch_add(1, std::memory_order_relaxed); });
    }

    ASSERT_EQ(completed.load(std::memory_order_relaxed), 1);
}

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
