// SPDX-License-Identifier: Apache-2.0
#define dquant_fused_attention dquant_probe_unused_attention
#define launch_dquant_attention launch_probe_unused_attention
#include "workspace.h"
#include "attention.h"

class DQuantDecoderProbe {
public:
    __aicore__ inline void Process(GM_ADDR records, GM_ADDR lut, GM_ADDR symbols,
        GM_ADDR states, uint32_t rows, uint32_t rowBytes, uint32_t containerBytes) {
        FusedDecoder decoder;
        decoder.Setup(nullptr, nullptr, records, nullptr, nullptr, lut, nullptr,
                      symbols, nullptr, nullptr, 0, 0, 0, 0);
        GlobalTensor<int32_t> finalStates;
        finalStates.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(states));
        for (uint32_t tile = GetBlockIdx(); tile < (rows + 63) / 64; tile += GetBlockNum()) {
            uint32_t valid = rows - tile * 64 < 64 ? rows - tile * 64 : 64;
            decoder.Prepare(decoder.keyRecords, decoder.keyEntries, tile * 64,
                            valid, rowBytes, containerBytes);
            for (uint32_t channel = 0; channel < 512; ++channel) {
                decoder.Step();
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                DataCopy(decoder.partials[(tile * 512 + channel) * 64],
                         decoder.work[1344].ReinterpretCast<float>(), 64);
                SetFlag<HardEvent::MTE3_V>(EVENT_ID0); WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            }
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(finalStates[tile * 128], decoder.work, 128);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0); WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0); WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        }
    }
};

extern "C" __global__ __aicore__ void dquant_probe_symbols(GM_ADDR records, GM_ADDR lut,
    GM_ADDR symbols, GM_ADDR states, uint32_t rows, uint32_t rowBytes, uint32_t containerBytes) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    DQuantDecoderProbe probe;
    probe.Process(records, lut, symbols, states, rows, rowBytes, containerBytes);
}

#ifndef ASCENDC_CPU_DEBUG
extern "C" void launch_dquant_probe(uint32_t blockDim, void* stream, void* records,
    void* lut, void* symbols, void* states, uint32_t rows, uint32_t rowBytes, uint32_t containerBytes) {
    dquant_probe_symbols<<<blockDim, nullptr, stream>>>(static_cast<uint8_t*>(records),
        static_cast<uint8_t*>(lut), static_cast<uint8_t*>(symbols), static_cast<uint8_t*>(states),
        rows, rowBytes, containerBytes);
}
#endif
