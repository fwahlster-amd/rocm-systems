/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <functional>
#include <memory>

namespace hipFile {

class IThreadPool {
public:
    virtual ~IThreadPool() = default;

    virtual void enqueue(std::function<void()> work) = 0;
    virtual void wait()                              = 0;

    virtual std::size_t threadCount() const noexcept = 0;
};

class ThreadPool : public IThreadPool {
public:
    ThreadPool();
    ~ThreadPool() noexcept override;

    ThreadPool(const ThreadPool &)            = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;
    ThreadPool(ThreadPool &&)                 = delete;
    ThreadPool &operator=(ThreadPool &&)      = delete;

    void enqueue(std::function<void()> work) override;
    void wait() override;

    std::size_t threadCount() const noexcept override;

private:
    struct ThreadPoolStorage;
    std::unique_ptr<ThreadPoolStorage> storage;
};

}
