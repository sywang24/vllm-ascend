// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "kernel_operator.h"

using namespace AscendC;

class DQuantVectorWorkspace {
protected:
    static constexpr uint32_t kTokenTile = 64;
    static constexpr uint32_t kGqaGroup = 6;
    static constexpr uint32_t kPvChannels = 32;
    static constexpr uint32_t kReductionStride = 8;
    static constexpr uint32_t kPvProducts = kTokenTile * kGqaGroup;
    static constexpr uint32_t kPvResultsPerChannel = kGqaGroup * kReductionStride;
    static constexpr uint32_t kByteMaskOffset = 1024;
    static constexpr uint32_t kPvResultsOffset = 64;
    static constexpr uint32_t kPvGatherIndices = 7424;
    static constexpr uint32_t kPvPackedResults = 7680;
    static constexpr uint32_t kQueryElements = 3072;
    static constexpr uint32_t kOutputWorkspaceBytes = 32768;
    static_assert(kPvResultsOffset + kPvChannels * kPvResultsPerChannel <= kQueryElements);
    static_assert(kPvGatherIndices + kGqaGroup * kPvChannels <= kPvPackedResults);
    static_assert(kPvPackedResults + kGqaGroup * kPvChannels <= kOutputWorkspaceBytes / sizeof(float));
    __aicore__ inline void InitBuffers() {
        // All decoded bytes and intermediate attention values stay in UB.
        pipe.InitBuffer(inputBuffer, 64 * 192);
        pipe.InitBuffer(halfBuffer, 64 * 192 * 2);
        pipe.InitBuffer(bytesBuffer, 64 * 192 * 4);
        pipe.InitBuffer(tableBuffer, 256 * 4);
        pipe.InitBuffer(outputBuffer, kOutputWorkspaceBytes);
        pipe.InitBuffer(workBuffer, 8192);
        pipe.InitBuffer(pvBuffer, kPvProducts * sizeof(float));
        raw = inputBuffer.Get<uint8_t>();
        halves = halfBuffer.Get<half>();
        bytes = bytesBuffer.Get<int32_t>();
        entries = tableBuffer.Get<uint32_t>();
        work = workBuffer.Get<int32_t>();
    }

    TPipe pipe;
    TBuf<TPosition::VECCALC> inputBuffer, halfBuffer, bytesBuffer, tableBuffer, outputBuffer, workBuffer;
    TBuf<TPosition::VECCALC> pvBuffer;
    LocalTensor<uint8_t> raw;
    LocalTensor<half> halves;
    LocalTensor<int32_t> bytes, work;
    LocalTensor<uint32_t> entries;
    uint32_t container;
};
