/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// Temporary glue header extracted during NCCL->RCCL merge.
// Provides the minimal subset of os.h needed by gin_v11.cc / gin_v12.cc
// until os.h is fully ported or replaced.

#ifndef TMP_MERGE_GLUE_H_
#define TMP_MERGE_GLUE_H_

#define NCCL_DESTROY NCCL_INIT

typedef void* ncclOsLibraryHandle;
void* ncclOsDlsym(ncclOsLibraryHandle handle, const char* symbol);

#endif // TMP_MERGE_GLUE_H_
