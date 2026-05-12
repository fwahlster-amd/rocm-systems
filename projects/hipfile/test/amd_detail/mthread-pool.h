/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "context.h"
#include "thread-pool.h"

#include <cstddef>
#include <functional>
#include <gmock/gmock.h>

namespace hipFile {

class MThreadPool : public IThreadPool {
public:
    ContextOverride<IThreadPool> co;

    MThreadPool() : co{this}
    {
    }

    MOCK_METHOD(void, enqueue, (std::function<void()> work), (override));
    MOCK_METHOD(void, wait, (), (override));
    MOCK_METHOD(std::size_t, threadCount, (), (const, noexcept, override));
};

}
