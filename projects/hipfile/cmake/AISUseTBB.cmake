# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
include(FetchContent)

# lint_cmake: -readability/wonkycase
FetchContent_Declare(
  tbb
  URL https://github.com/uxlfoundation/oneTBB/archive/refs/tags/v2022.3.0.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP true
  SYSTEM
)
# lint_cmake: +readability/wonkycase

set(TBB_TEST OFF CACHE BOOL "" FORCE)
# lint_cmake: -readability/wonkycase
FetchContent_MakeAvailable(tbb)
# lint_cmake: +readability/wonkycase

if(tbb_SOURCE_DIR)
    message(STATUS "Using fetched TBB")
else()
    message(STATUS "Using system TBB")
endif()
