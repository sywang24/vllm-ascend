// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"

class FusedDecoder : public DQuantVectorWorkspace {
public:
    __aicore__ inline void Setup(GM_ADDR query, GM_ADDR original, GM_ADDR keys, GM_ADDR values,
        GM_ADDR blocks, GM_ADDR keyTable, GM_ADDR valueTable, GM_ADDR output, GM_ADDR lses,
        GM_ADDR lengths, uint32_t batch, uint32_t maxChunks, uint32_t tableStride, float scale) {
        InitBuffers();
        rotated.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(query));
        originalQuery.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(original));
        keyRecords.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(keys));
        valueRecords.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(values));
        pageTable.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(blocks));
        keyEntries.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(keyTable));
        valueEntries.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(valueTable));
        partials.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
        logSums.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(lses));
        compressed.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(lengths));
        batchCount = batch; chunks = maxChunks; pages = tableStride; attentionScale = scale;
    }

    __aicore__ inline void ProcessAttention() {
        auto floats = outputBuffer.Get<float>();
        auto q = floats;
        auto scores = floats[3072];
        auto output = floats[4096];
        auto reductions = floats[7168];
        auto logs = floats[7296];
        auto means = work[1536].ReinterpretCast<float>();
        auto temp = work[1152].ReinterpretCast<float>();
        auto scales = work[1216].ReinterpretCast<float>();
        auto offsets = work[1280].ReinterpretCast<float>();
        auto reduceWork = work[1408].ReinterpretCast<float>();
        auto products = pvBuffer.Get<float>();
        // QK consumes q before PV reuses this region; its first entries retain the LSE.
        auto pvResults = q[kPvResultsOffset];
        auto pvIndices = floats[kPvGatherIndices].ReinterpretCast<uint32_t>();
        auto pvPacked = floats[kPvPackedResults];
        for (uint32_t head = 0; head < kGqaGroup; ++head) {
            for (uint32_t lane = 0; lane < kPvChannels; ++lane) {
                pvIndices.SetValue(head * kPvChannels + lane,
                    (lane * kPvResultsPerChannel + head * kReductionStride) * sizeof(float));
            }
        }
        float qsum[12], qmean[12], maxima[12], denominators[12], offsetSum[12];
        uint32_t activeOffset = 0;
        // Flatten only live tiles, assigning each to its global index modulo the core count.
        // Lengths stay on device so captured graphs also rebalance growing or reordered requests.
        for (uint32_t batch = 0; batch < batchCount; ++batch) {
            uint32_t length = compressed.GetValue(batch);
            uint32_t end = (length + kTokenTile - 1) / kTokenTile;
            uint32_t firstChunk = (GetBlockIdx() + GetBlockNum() - activeOffset % GetBlockNum()) % GetBlockNum();
            activeOffset += end;
            for (uint32_t chunk = firstChunk; chunk < end; chunk += GetBlockNum()) {
                uint32_t valid = length - chunk * 64 < 64 ? length - chunk * 64 : 64;
                uint32_t page = pageTable.GetValue(batch * pages + chunk / 2);
                uint32_t first = page * 128 + chunk % 2 * 64;
                DataCopy(q, rotated[batch * 3072], 3072);
                auto originalBf = halves.ReinterpretCast<bfloat16_t>();
                DataCopy(originalBf, originalQuery[batch * 3072], 3072);
                for (uint32_t record = 0; record < 64; ++record) {
                    DataCopyExtParams params{1, 16, 0, 0, 0};
                    DataCopyPadExtParams<uint8_t> padding{false, 0, 0, 0};
                    DataCopyPad(raw[record * 32], keyRecords[(first + record) * 176 + 160], params, padding);
                }
                SetFlag<HardEvent::MTE2_S>(EVENT_ID0); WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
                for (uint32_t channel = 0; channel < 512; ++channel) {
                    originalBf.SetValue(3072 + channel,
                        raw.ReinterpretCast<bfloat16_t>().GetValue(channel / 8 * 16 + channel % 8));
                }
                SetFlag<HardEvent::S_V>(EVENT_ID0); WaitFlag<HardEvent::S_V>(EVENT_ID0);
                auto originalFloat = bytes.ReinterpretCast<float>();
                Cast(originalFloat, originalBf, RoundMode::CAST_NONE, 3072);
                Cast(means, originalBf[3072], RoundMode::CAST_NONE, 512);
                PipeBarrier<PIPE_V>();
                for (uint32_t head = 0; head < 12; ++head) {
                    ReduceSum(reductions[head * 8], q[head * 256], reduceWork, 256);
                    Mul(originalFloat[head * 256], originalFloat[head * 256], means[head / 6 * 256], 256);
                    PipeBarrier<PIPE_V>();
                    ReduceSum(logs[head * 8], originalFloat[head * 256], reduceWork, 256);
                    PipeBarrier<PIPE_V>();
                }
                SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
                for (uint32_t head = 0; head < 12; ++head) {
                    qsum[head] = reductions.GetValue(head * 8);
                    qmean[head] = logs.GetValue(head * 8);
                }
                Prepare(keyRecords, keyEntries, first, valid, 176, 155);
                Duplicate(scores, 0.0f, 12 * 64);
                PipeBarrier<PIPE_V>();
                for (uint32_t channel = 0; channel < 512; ++channel) {
                    Step();
                    for (uint32_t localHead = 0; localHead < 6; ++localHead) {
                        uint32_t head = channel / 256 * 6 + localHead;
                        Axpy(scores[head * 64], work[1344].ReinterpretCast<float>(),
                             q.GetValue(head * 256 + channel % 256), 64);
                    }
                    PipeBarrier<PIPE_V>();
                }
                for (uint32_t head = 0; head < 12; ++head) {
                    Mul(scores[head * 64], scores[head * 64], scales, 64);
                    Muls(temp, offsets, qsum[head], 64);
                    PipeBarrier<PIPE_V>();
                    Add(scores[head * 64], scores[head * 64], temp, 64);
                    PipeBarrier<PIPE_V>();
                    Adds(scores[head * 64], scores[head * 64], qmean[head], 64);
                    PipeBarrier<PIPE_V>();
                    Muls(scores[head * 64], scores[head * 64], attentionScale, 64);
                    PipeBarrier<PIPE_V>();
                }
                SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
                for (uint32_t head = 0; head < 12; ++head) {
                    for (uint32_t row = valid; row < 64; ++row) {
                        scores.SetValue(head * 64 + row, -3.402823466e38f);
                    }
                }
                SetFlag<HardEvent::S_V>(EVENT_ID0); WaitFlag<HardEvent::S_V>(EVENT_ID0);
                for (uint32_t head = 0; head < 12; ++head) {
                    ReduceMax(reductions[head * 8], scores[head * 64], reduceWork, 64);
                    PipeBarrier<PIPE_V>();
                }
                SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
                for (uint32_t head = 0; head < 12; ++head) maxima[head] = reductions.GetValue(head * 8);
                SetFlag<HardEvent::S_V>(EVENT_ID0); WaitFlag<HardEvent::S_V>(EVENT_ID0);
                for (uint32_t head = 0; head < 12; ++head) {
                    Adds(scores[head * 64], scores[head * 64], -maxima[head], 64);
                    PipeBarrier<PIPE_V>();
                    Exp(scores[head * 64], scores[head * 64], 64);
                    PipeBarrier<PIPE_V>();
                    ReduceSum(reductions[head * 8], scores[head * 64], reduceWork, 64);
                    PipeBarrier<PIPE_V>();
                }
                SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
                for (uint32_t head = 0; head < 12; ++head) denominators[head] = reductions.GetValue(head * 8);
                SetFlag<HardEvent::S_V>(EVENT_ID0); WaitFlag<HardEvent::S_V>(EVENT_ID0);
                Ln(logs, reductions, 96);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
                for (uint32_t head = 0; head < 12; ++head) q.SetValue(head, maxima[head] + logs.GetValue(head * 8));
                Prepare(valueRecords, valueEntries, first, valid, 114, 109);
                for (uint32_t head = 0; head < 12; ++head) {
                    Muls(scores[head * 64], scores[head * 64], 1.0f / denominators[head], 64);
                    PipeBarrier<PIPE_V>();
                    Mul(temp, scores[head * 64], offsets, 64);
                    PipeBarrier<PIPE_V>();
                    ReduceSum(reductions[head * 8], temp, reduceWork, 64);
                    Mul(scores[head * 64], scores[head * 64], scales, 64);
                    PipeBarrier<PIPE_V>();
                }
                SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
                for (uint32_t head = 0; head < 12; ++head) offsetSum[head] = reductions.GetValue(head * 8);
                SetFlag<HardEvent::S_V>(EVENT_ID0); WaitFlag<HardEvent::S_V>(EVENT_ID0);
                for (uint32_t channel = 0; channel < 512; ++channel) {
                    Step();
                    for (uint32_t localHead = 0; localHead < kGqaGroup; ++localHead) {
                        uint32_t head = channel / 256 * kGqaGroup + localHead;
                        Mul(products[localHead * kTokenTile], scores[head * kTokenTile],
                            work[1344].ReinterpretCast<float>(), kTokenTile);
                    }
                    PipeBarrier<PIPE_V>();
                    // Collect channels in UB, then pack all six heads without scalar handoff.
                    ReduceRepeat<ReduceType::SUM>(pvResults[(channel % kPvChannels) * kPvResultsPerChannel], products,
                        int32_t(kTokenTile), kGqaGroup, kReductionStride, 1, kTokenTile / kReductionStride);
                    PipeBarrier<PIPE_V>();
                    if (channel % kPvChannels == kPvChannels - 1) {
                        Gather(pvPacked, pvResults, pvIndices, 0, kGqaGroup * kPvChannels);
                        PipeBarrier<PIPE_V>();
                        for (uint32_t localHead = 0; localHead < kGqaGroup; ++localHead) {
                            uint32_t head = channel / 256 * kGqaGroup + localHead;
                            Adds(output[head * 256 + channel % 256 - (kPvChannels - 1)],
                                pvPacked[localHead * kPvChannels], offsetSum[head], kPvChannels);
                        }
                        PipeBarrier<PIPE_V>();
                    }
                }
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                SetFlag<HardEvent::S_MTE3>(EVENT_ID0); WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
                DataCopy(partials[(batch * chunks + chunk) * 3072], output, 3072);
                DataCopyExtParams lseParams{1, 48, 0, 0, 0};
                DataCopyPad(logSums[(batch * chunks + chunk) * 12], q, lseParams);
                SetFlag<HardEvent::MTE3_S>(EVENT_ID0); WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
                SetFlag<HardEvent::MTE3_V>(EVENT_ID0); WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            }
        }
    }

private:
    friend class DQuantDecoderProbe;
    __aicore__ inline void Prepare(GlobalTensor<uint8_t>& records, GlobalTensor<uint32_t>& table,
                                   uint32_t first, uint32_t valid, uint32_t rowBytes, uint32_t containerBytes) {
        container = containerBytes;
        uint32_t metadata = (container + 1) / 2 * 2;
        for (uint32_t row = 0; row < 64; ++row) {
            DataCopyExtParams params{1, metadata + 4, 0, 0, 0};
            DataCopyPadExtParams<uint8_t> padding{false, 0, 0, 0};
            DataCopyPad(raw[row * 192], records[(first + (row < valid ? row : 0)) * rowBytes], params, padding);
        }
        DataCopy(entries, table, 256);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0); WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        for (uint32_t row = 0; row < 64; ++row) {
            work.SetValue(row, raw.ReinterpretCast<uint32_t>().GetValue(row * 48));
            work.SetValue(64 + row, 4);
            work.SetValue(1088 + row, row * 192);
            work[1216].ReinterpretCast<float>().SetValue(row, float(raw.ReinterpretCast<half>().GetValue(row * 96 + metadata / 2)));
            work[1280].ReinterpretCast<float>().SetValue(row, float(raw.ReinterpretCast<half>().GetValue(row * 96 + metadata / 2 + 1)));
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0); WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Cast(halves, raw, RoundMode::CAST_NONE, 64 * 192);
        PipeBarrier<PIPE_V>();
        Cast(bytes, halves, RoundMode::CAST_RINT, 64 * 192);
        Duplicate(work[768].ReinterpretCast<float>(), float(int32_t(container)), 64);
        Duplicate(work[832].ReinterpretCast<float>(), float(1 << 23), 64);
        Duplicate(work[kByteMaskOffset], int32_t(255), kTokenTile);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Step() {
        auto state = work[0]; auto cursor = work[64]; auto slot = work[128]; auto entry = work[192];
        auto freq = work[256]; auto cumulative = work[320]; auto scratch = work[384];
        auto address = work[448]; auto byte = work[512]; auto refilled = work[576]; auto advanced = work[640];
        auto compare = work[704].ReinterpretCast<float>(); auto limit = work[768].ReinterpretCast<float>();
        auto lower = work[832].ReinterpretCast<float>(); auto mask = work[896].ReinterpretCast<uint8_t>();
        auto mask2 = work[960].ReinterpretCast<uint8_t>();
        ShiftRight(scratch, state, int32_t(8), 64);
        And(slot.ReinterpretCast<uint16_t>(), state.ReinterpretCast<uint16_t>(),
            work[kByteMaskOffset].ReinterpretCast<uint16_t>(), kTokenTile * 2);
        PipeBarrier<PIPE_V>();
        ShiftLeft(address, slot, int32_t(2), 64); PipeBarrier<PIPE_V>();
        Gather(entry.ReinterpretCast<uint32_t>(), entries, address.ReinterpretCast<uint32_t>(), 0, 64); PipeBarrier<PIPE_V>();
        ShiftLeft(freq, entry, int32_t(15), 64);
        ShiftRight(cumulative, entry, int32_t(17), 64); PipeBarrier<PIPE_V>();
        ShiftRight(freq.ReinterpretCast<uint32_t>(), freq.ReinterpretCast<uint32_t>(), uint32_t(23), 64); PipeBarrier<PIPE_V>();
        Mul(state, freq, scratch, 64); Sub(slot, slot, cumulative, 64); PipeBarrier<PIPE_V>();
        Add(state, state, slot, 64); Add(address, work[1088], cursor, 64); PipeBarrier<PIPE_V>();
        ShiftLeft(address, address, int32_t(2), 64);
        Cast(compare, state, RoundMode::CAST_NONE, 64); PipeBarrier<PIPE_V>();
        Compare(mask, compare, lower, CMPMODE::LT, 64);
        Gather(byte, bytes, address.ReinterpretCast<uint32_t>(), 0, 64);
        Cast(compare, cursor, RoundMode::CAST_NONE, 64); PipeBarrier<PIPE_V>();
        Compare(mask2, compare, limit, CMPMODE::LT, 64);
        ShiftLeft(refilled, state, int32_t(8), 64); Adds(advanced, cursor, int32_t(1), 64); PipeBarrier<PIPE_V>();
        And(mask.ReinterpretCast<uint16_t>(), mask.ReinterpretCast<uint16_t>(), mask2.ReinterpretCast<uint16_t>(), 16);
        Add(refilled, refilled, byte, 64); PipeBarrier<PIPE_V>();
        Select(state.ReinterpretCast<float>(), mask, refilled.ReinterpretCast<float>(), state.ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, 64);
        Select(cursor.ReinterpretCast<float>(), mask, advanced.ReinterpretCast<float>(), cursor.ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, 64);
        And(scratch.ReinterpretCast<uint16_t>(), entry.ReinterpretCast<uint16_t>(),
            work[kByteMaskOffset].ReinterpretCast<uint16_t>(), kTokenTile * 2);
        PipeBarrier<PIPE_V>();
        Cast(work[1344].ReinterpretCast<float>(), scratch, RoundMode::CAST_NONE, 64); PipeBarrier<PIPE_V>();
    }

    GlobalTensor<float> rotated, partials, logSums;
    GlobalTensor<bfloat16_t> originalQuery;
    GlobalTensor<uint8_t> keyRecords, valueRecords;
    GlobalTensor<int32_t> pageTable, compressed;
    GlobalTensor<uint32_t> keyEntries, valueEntries;
    uint32_t batchCount, chunks, pages;
    float attentionScale;
};

extern "C" __global__ __aicore__ void dquant_fused_attention(
    GM_ADDR query, GM_ADDR original, GM_ADDR keys, GM_ADDR values, GM_ADDR blocks,
    GM_ADDR keyTable, GM_ADDR valueTable, GM_ADDR output, GM_ADDR lses, GM_ADDR lengths,
    uint32_t batch, uint32_t maxChunks, uint32_t tableStride, float scale) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    FusedDecoder decoder;
    decoder.Setup(query, original, keys, values, blocks, keyTable, valueTable, output, lses,
                  lengths, batch, maxChunks, tableStride, scale);
    decoder.ProcessAttention();
}

#ifndef ASCENDC_CPU_DEBUG
extern "C" void launch_dquant_attention(uint32_t blockDim, void* stream, void* query, void* original,
    void* keys, void* values, void* blocks, void* keyTable, void* valueTable, void* output,
    void* lses, void* lengths, uint32_t batch, uint32_t maxChunks, uint32_t tableStride, float scale) {
    dquant_fused_attention<<<blockDim, nullptr, stream>>>(static_cast<uint8_t*>(query),
        static_cast<uint8_t*>(original), static_cast<uint8_t*>(keys), static_cast<uint8_t*>(values),
        static_cast<uint8_t*>(blocks), static_cast<uint8_t*>(keyTable), static_cast<uint8_t*>(valueTable),
        static_cast<uint8_t*>(output), static_cast<uint8_t*>(lses), static_cast<uint8_t*>(lengths),
        batch, maxChunks, tableStride, scale);
}
#endif
