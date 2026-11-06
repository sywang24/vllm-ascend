// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"

using namespace AscendC;
constexpr uint32_t DIM = 256;
constexpr uint32_t GROUPS = DIM / 4;
constexpr uint32_t TILE = 128;
constexpr uint32_t ELEMENTS = GROUPS * TILE;
constexpr uint32_t MAX_SPLIT = 1024;
constexpr uint32_t MAX_QUERY_GROUPS = 6;

// One task owns a batch-head and split. K/V code tiles are shared by every
// query group, retaining only one decoded tile in local memory.
extern "C" __global__ __aicore__ void nova_attention_shared(
    GM_ADDR lookup, GM_ADDR mean, GM_ADDR key, GM_ADDR value, GM_ADDR scales,
    GM_ADDR table, GM_ADDR starts, GM_ADDR ends, GM_ADDR group_offsets,
    GM_ADDR dim_offsets, GM_ADDR partial, GM_ADDR maxima, GM_ADDR sums,
    uint32_t batch, uint32_t heads, uint32_t groups, uint32_t page_size,
    uint32_t table_stride, uint32_t splits, uint32_t split_tokens,
    uint32_t code_page_stride, uint32_t scale_page_stride,
    uint32_t table_batch_stride, uint32_t profile_mode, float scale) {
    TPipe pipe;
    TBuf<TPosition::VECCALC> book_buf, code_buf, half_buf, index_buf, matrix_buf;
    TBuf<TPosition::VECCALC> group_buf, dim_buf, scale16_buf, scale32_buf, table_buf;
    TBuf<TPosition::VECCALC> score_buf, scratch_buf, accum_buf, scalar_buf, weight_buf;
    TBuf<TPosition::VECCALC> reduce_buf, weight_h_buf, reduce_h_buf;
    TBuf<TPosition::VECCALC> low_group_buf;
    pipe.InitBuffer(book_buf, GROUPS * DIM * 2);
    pipe.InitBuffer(code_buf, ELEMENTS);
    pipe.InitBuffer(half_buf, ELEMENTS * 2);
    pipe.InitBuffer(index_buf, ELEMENTS * 2);
    pipe.InitBuffer(matrix_buf, ELEMENTS * 4);
    pipe.InitBuffer(group_buf, ELEMENTS * 4);
    pipe.InitBuffer(dim_buf, DIM * 4);
    pipe.InitBuffer(table_buf, 2048 * 4);
    pipe.InitBuffer(scale16_buf, 3 * TILE * 2);
    pipe.InitBuffer(scale32_buf, 3 * TILE * 4);
    pipe.InitBuffer(score_buf, MAX_QUERY_GROUPS * MAX_SPLIT * 4);
    pipe.InitBuffer(scratch_buf, MAX_SPLIT * 4);
    pipe.InitBuffer(accum_buf, MAX_QUERY_GROUPS * DIM * 4);
    pipe.InitBuffer(scalar_buf, 8 * 4);
    pipe.InitBuffer(weight_buf, TILE * 4);
    pipe.InitBuffer(reduce_buf, 2 * GROUPS * 4);
    pipe.InitBuffer(weight_h_buf, TILE * 2);
    pipe.InitBuffer(reduce_h_buf, 2 * GROUPS * 2);
    pipe.InitBuffer(low_group_buf, 8 * 4);
    auto book = book_buf.Get<half>();
    auto code = code_buf.Get<uint8_t>();
    auto half_code = half_buf.Get<half>();
    auto index = index_buf.Get<int16_t>();
    auto matrix = matrix_buf.Get<float>();
    auto group_index = group_buf.Get<int32_t>();
    auto dimension_index = dim_buf.Get<uint32_t>();
    auto local_table = table_buf.Get<int32_t>();
    auto scales16 = scale16_buf.Get<bfloat16_t>();
    auto scales32 = scale32_buf.Get<float>();
    auto scores = score_buf.Get<float>();
    auto scratch = scratch_buf.Get<float>();
    auto accum = accum_buf.Get<float>();
    auto scalar = scalar_buf.Get<float>();
    auto weight = weight_buf.Get<float>();
    auto reduced = reduce_buf.Get<float>();
    // Key LUT and gathered key matrix are dead during value processing.
    // Reuse their storage for two value planes and weighted products.
    auto value_h = book_buf.Get<half>();
    auto weight_h = weight_h_buf.Get<half>();
    auto reduced_h = reduce_h_buf.Get<half>();
    auto work_h = matrix_buf.Get<half>();
    auto low_group = low_group_buf.Get<float>();
    GlobalTensor<half> lut_gm;
    GlobalTensor<float> mean_gm, part_gm, max_gm, sum_gm;
    GlobalTensor<uint8_t> key_gm, value_gm;
    GlobalTensor<bfloat16_t> scale_gm;
    GlobalTensor<int32_t> table_gm, starts_gm, ends_gm, group_gm;
    GlobalTensor<uint32_t> dim_gm;
    lut_gm.SetGlobalBuffer((__gm__ half*)lookup);
    mean_gm.SetGlobalBuffer((__gm__ float*)mean);
    key_gm.SetGlobalBuffer((__gm__ uint8_t*)key);
    value_gm.SetGlobalBuffer((__gm__ uint8_t*)value);
    scale_gm.SetGlobalBuffer((__gm__ bfloat16_t*)scales);
    table_gm.SetGlobalBuffer((__gm__ int32_t*)table);
    starts_gm.SetGlobalBuffer((__gm__ int32_t*)starts);
    ends_gm.SetGlobalBuffer((__gm__ int32_t*)ends);
    group_gm.SetGlobalBuffer((__gm__ int32_t*)group_offsets);
    dim_gm.SetGlobalBuffer((__gm__ uint32_t*)dim_offsets);
    part_gm.SetGlobalBuffer((__gm__ float*)partial);
    max_gm.SetGlobalBuffer((__gm__ float*)maxima);
    sum_gm.SetGlobalBuffer((__gm__ float*)sums);
    DataCopy(group_index, group_gm, ELEMENTS);
    DataCopy(dimension_index, dim_gm, DIM);
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);
    for (uint32_t task = GetBlockIdx(); task < batch * heads * splits; task += GetBlockNum()) {
        uint32_t bh = task % (batch * heads);
        uint32_t split = task / (batch * heads);
        uint32_t request = bh / heads;
        uint32_t head = bh % heads;
        int32_t begin = starts_gm.GetValue(request);
        int32_t end = ends_gm.GetValue(request);
        int32_t lo = begin / TILE * TILE + split * split_tokens;
        int32_t hi = lo + split_tokens < end ? lo + split_tokens : end;
        bool active = begin < end && lo < hi;
        if (!active) continue;
        if (active) {
            DataCopyPad(local_table, table_gm[request * table_batch_stride],
                        {1, table_stride * 4, 0, 0, 0}, {false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_S>(0);
            WaitFlag<HardEvent::MTE2_S>(0);
        }
        // K codes and scales are independent of the query group.  The old
        // group-major loop fetched and unpacked the same tile once per group
        // (six times for Qwen3.8), making compressed attention bandwidth-bound.
        // Keep one tile in VE memory and reuse it for every query LUT.
        float local_max[MAX_QUERY_GROUPS];
        float local_sum[MAX_QUERY_GROUPS];
        for (uint32_t group = 0; group < groups; ++group) {
            Duplicate(scores[group * MAX_SPLIT], -1.0e30f, split_tokens);
            local_max[group] = -1.0e30f;
            local_sum[group] = 0.0f;
            low_group.SetValue(group, 0.0f);
            Duplicate(accum[group * DIM], 0.0f, DIM);
        }
        if (active && profile_mode == 3) {
            for (uint32_t group = 0; group < groups; ++group) {
                uint32_t query = bh * groups + group;
                DataCopy(book, lut_gm[query * GROUPS * DIM], GROUPS * DIM);
                SetFlag<HardEvent::MTE2_V>(0);
                WaitFlag<HardEvent::MTE2_V>(0);
            }
        } else if (active && profile_mode != 1 && profile_mode != 4 && profile_mode != 5) {
            for (int32_t pos = lo; pos < hi;) {
                uint32_t page = local_table.GetValue(pos / page_size);
                uint32_t token = pos % page_size;
                uint32_t load_tokens = page_size - token < TILE ? page_size - token : TILE;
                uint32_t active_tokens = hi - pos < load_tokens ? hi - pos : load_tokens;
                uint32_t base = page * code_page_stride + head * GROUPS * page_size + token;
                DataCopy(code, key_gm[base],
                         {GROUPS, (uint16_t)(load_tokens / 32),
                          (uint16_t)((page_size - load_tokens) / 32),
                          (uint16_t)((TILE - load_tokens) / 32)});
                DataCopy(scales16, scale_gm[page * scale_page_stride + head * 3 * page_size + token],
                         {3, (uint16_t)(load_tokens / 16),
                          (uint16_t)((page_size - load_tokens) / 16),
                          (uint16_t)((TILE - load_tokens) / 16)});
                DataCopy(group_index, group_gm, ELEMENTS);
                if (load_tokens < TILE) {
                    uint32_t next_page = pos + load_tokens < hi
                        ? local_table.GetValue((pos + load_tokens) / page_size) : 0;
                    uint32_t remaining = TILE - load_tokens;
                    DataCopy(code[load_tokens], key_gm[next_page * code_page_stride + head * GROUPS * page_size],
                             {GROUPS, (uint16_t)(remaining / 32),
                              (uint16_t)((page_size - remaining) / 32), (uint16_t)(load_tokens / 32)});
                    DataCopy(scales16[load_tokens], scale_gm[next_page * scale_page_stride + head * 3 * page_size],
                             {3, (uint16_t)(remaining / 16),
                              (uint16_t)((page_size - remaining) / 16), (uint16_t)(load_tokens / 16)});
                    load_tokens = TILE;
                    active_tokens = hi - pos < TILE ? hi - pos : TILE;
                }
                SetFlag<HardEvent::MTE2_V>(0);
                WaitFlag<HardEvent::MTE2_V>(0);
                Cast(half_code, code, RoundMode::CAST_NONE, ELEMENTS);
                Cast(matrix, half_code, RoundMode::CAST_NONE, ELEMENTS);
                Muls(matrix, matrix, 2.0f, ELEMENTS);
                Cast(matrix.ReinterpretCast<int32_t>(), matrix,
                     RoundMode::CAST_RINT, ELEMENTS);
                // Cache each K tile's Gather offsets for all query groups.
                Add(group_index, matrix.ReinterpretCast<int32_t>(), group_index, ELEMENTS);
                Cast(scales32, scales16, RoundMode::CAST_NONE, 3 * TILE);
                for (uint32_t group = 0; group < groups; ++group) {
                    uint32_t query = bh * groups + group;
                    auto group_scores = scores[group * MAX_SPLIT];
                    DataCopy(book, lut_gm[query * GROUPS * DIM], GROUPS * DIM);
                    SetFlag<HardEvent::MTE2_V>(0);
                    WaitFlag<HardEvent::MTE2_V>(0);
                    Gather(half_code, book, group_index.ReinterpretCast<uint32_t>(), 0, ELEMENTS);
                    Cast(matrix, half_code, RoundMode::CAST_NONE, ELEMENTS);
                    for (uint32_t stride = GROUPS / 2; stride > 0; stride /= 2) {
                        Add(matrix, matrix, matrix[stride * TILE], stride * TILE);
                    }
                    Mul(matrix, matrix, scales32, TILE);
                    Adds(matrix, matrix, mean_gm.GetValue(query), TILE);
                    Muls(group_scores[pos - lo], matrix, scale, active_tokens);
                    if (pos < begin || pos + active_tokens > hi) {
                        SetFlag<HardEvent::V_S>(0);
                        WaitFlag<HardEvent::V_S>(0);
                        for (uint32_t i = 0; i < active_tokens; ++i) {
                            if (pos + i < begin || pos + i >= hi) {
                                group_scores.SetValue(pos - lo + i, -1.0e30f);
                            }
                        }
                        SetFlag<HardEvent::S_V>(0);
                        WaitFlag<HardEvent::S_V>(0);
                    }
                    // The next query DMA reuses book; Gather must finish first.
                    SetFlag<HardEvent::V_MTE2>(0);
                    WaitFlag<HardEvent::V_MTE2>(0);
                }
                SetFlag<HardEvent::V_MTE2>(0);
                WaitFlag<HardEvent::V_MTE2>(0);
                pos += load_tokens;
            }
        }
        if (active) {
            for (uint32_t group = 0; group < groups; ++group) {
                auto group_scores = scores[group * MAX_SPLIT];
                if (profile_mode == 1) Duplicate(group_scores, 0.0f, split_tokens);
                ReduceMax(scalar, group_scores, scratch, split_tokens);
                SetFlag<HardEvent::V_S>(0);
                WaitFlag<HardEvent::V_S>(0);
                local_max[group] = scalar.GetValue(0);
                Adds(group_scores, group_scores, -local_max[group], split_tokens);
                Exp(group_scores, group_scores, split_tokens);
                ReduceSum(scalar, group_scores, scratch, split_tokens);
                SetFlag<HardEvent::V_S>(0);
                WaitFlag<HardEvent::V_S>(0);
                local_sum[group] = scalar.GetValue(0);
                uint32_t query = bh * groups + group;
                uint32_t output = query * splits + split;
                scalar.SetValue(0, local_max[group]);
                SetFlag<HardEvent::S_MTE3>(0);
                WaitFlag<HardEvent::S_MTE3>(0);
                DataCopy(max_gm[output * 8], scalar, 8);
                SetFlag<HardEvent::MTE3_S>(0);
                WaitFlag<HardEvent::MTE3_S>(0);
                scalar.SetValue(0, local_sum[group]);
                SetFlag<HardEvent::S_MTE3>(0);
                WaitFlag<HardEvent::S_MTE3>(0);
                DataCopy(sum_gm[output * 8], scalar, 8);
                SetFlag<HardEvent::MTE3_S>(0);
                WaitFlag<HardEvent::MTE3_S>(0);
            }
        }
        if (active && profile_mode != 2 && profile_mode != 3) {
            for (int32_t pos = lo; pos < hi;) {
                uint32_t page = local_table.GetValue(pos / page_size);
                uint32_t token = pos % page_size;
                uint32_t load_tokens = page_size - token < TILE ? page_size - token : TILE;
                uint32_t active_tokens = hi - pos < load_tokens ? hi - pos : load_tokens;
                uint32_t base = page * code_page_stride + head * GROUPS * page_size + token;
                DataCopy(code, value_gm[base],
                         {GROUPS, (uint16_t)(load_tokens / 32),
                          (uint16_t)((page_size - load_tokens) / 32),
                          (uint16_t)((TILE - load_tokens) / 32)});
                DataCopy(scales16, scale_gm[page * scale_page_stride + head * 3 * page_size + token],
                         {3, (uint16_t)(load_tokens / 16),
                          (uint16_t)((page_size - load_tokens) / 16),
                          (uint16_t)((TILE - load_tokens) / 16)});
                if (load_tokens < TILE) {
                    uint32_t next_page = pos + load_tokens < hi
                        ? local_table.GetValue((pos + load_tokens) / page_size) : 0;
                    uint32_t remaining = TILE - load_tokens;
                    DataCopy(code[load_tokens], value_gm[next_page * code_page_stride + head * GROUPS * page_size],
                             {GROUPS, (uint16_t)(remaining / 32),
                              (uint16_t)((page_size - remaining) / 32), (uint16_t)(load_tokens / 32)});
                    DataCopy(scales16[load_tokens], scale_gm[next_page * scale_page_stride + head * 3 * page_size],
                             {3, (uint16_t)(remaining / 16),
                              (uint16_t)((page_size - remaining) / 16), (uint16_t)(load_tokens / 16)});
                    load_tokens = TILE;
                    active_tokens = hi - pos < TILE ? hi - pos : TILE;
                }
                SetFlag<HardEvent::MTE2_V>(0);
                WaitFlag<HardEvent::MTE2_V>(0);
                Cast(half_code, code, RoundMode::CAST_NONE, ELEMENTS);
                Cast(index.ReinterpretCast<int16_t>(), half_code, RoundMode::CAST_RINT, ELEMENTS);
                Cast(scales32, scales16, RoundMode::CAST_NONE, 3 * TILE);
                if (pos + TILE > hi) {
                    SetFlag<HardEvent::V_S>(0);
                    WaitFlag<HardEvent::V_S>(0);
                    for (uint32_t i = hi - pos; i < TILE; ++i) {
                        scales32.SetValue(TILE + i, 0.0f);
                        scales32.SetValue(2 * TILE + i, 0.0f);
                    }
                    SetFlag<HardEvent::S_V>(0);
                    WaitFlag<HardEvent::S_V>(0);
                }
                // profile_mode 4 times unpack only, 5 times multiply/reduce only.
                if (profile_mode == 4) {
                    SetFlag<HardEvent::V_MTE2>(0);
                    WaitFlag<HardEvent::V_MTE2>(0);
                    pos += load_tokens;
                    continue;
                }
                // Retain only two decoded planes at once; this keeps the
                // 128-token tile within the same bounded local-memory budget.
                for (uint32_t component = 0; component < 4; component += 2) {
                    for (uint32_t plane = 0; plane < 2; ++plane) {
                        ShiftLeft(half_code.ReinterpretCast<uint16_t>(), index.ReinterpretCast<uint16_t>(),
                                  (uint16_t)(14 - 2 * (component + plane)), ELEMENTS);
                        ShiftRight(half_code.ReinterpretCast<uint16_t>(), half_code.ReinterpretCast<uint16_t>(),
                                   (uint16_t)14, ELEMENTS);
                        Cast(value_h[plane * ELEMENTS], half_code.ReinterpretCast<int16_t>(),
                             RoundMode::CAST_NONE, ELEMENTS);
                        Cast(matrix, value_h[plane * ELEMENTS], RoundMode::CAST_NONE, ELEMENTS);
                        // FP32 vector masks cover at most 64 lanes on 910B.
                        // Apply each half of the token tile independently.
                        for (uint32_t half_tile = 0; half_tile < TILE; half_tile += 64) {
                            Mul(matrix[half_tile], matrix[half_tile], scales32[TILE + half_tile],
                                64, GROUPS, {1, 1, 1, TILE / 8, TILE / 8, 0});
                            Add(matrix[half_tile], matrix[half_tile], scales32[2 * TILE + half_tile],
                                64, GROUPS, {1, 1, 1, TILE / 8, TILE / 8, 0});
                        }
                        Cast(value_h[plane * ELEMENTS], matrix, RoundMode::CAST_NONE, ELEMENTS);
                    }
                    for (uint32_t group = 0; group < groups; ++group) {
                        auto group_scores = scores[group * MAX_SPLIT];
                        uint32_t score_tokens = split_tokens - (pos - lo) < load_tokens
                            ? split_tokens - (pos - lo) : load_tokens;
                        Muls(weight, group_scores[pos - lo], 1.0f, score_tokens);
                        if (score_tokens < TILE) Duplicate(weight[score_tokens], 0.0f, TILE - score_tokens);
                        Cast(weight_h, weight, RoundMode::CAST_NONE, TILE);
                        Mul(work_h, value_h, weight_h, TILE, 2 * GROUPS,
                            {1, 1, 1, TILE / 16, TILE / 16, 0});
                        WholeReduceSum(reduced_h, work_h, TILE, 2 * GROUPS, 1, 1, TILE / 16);
                        Cast(reduced, reduced_h, RoundMode::CAST_NONE, 2 * GROUPS);
                        Add(accum[group * DIM + component * GROUPS],
                            accum[group * DIM + component * GROUPS], reduced, 2 * GROUPS);
                    }
                }
                SetFlag<HardEvent::V_MTE2>(0);
                WaitFlag<HardEvent::V_MTE2>(0);
                pos += load_tokens;
            }
            for (uint32_t group = 0; group < groups; ++group) {
                uint32_t output = (bh * groups + group) * splits + split;
                Adds(accum[group * DIM], accum[group * DIM], low_group.GetValue(group), DIM);
                Gather(matrix, accum[group * DIM], dimension_index, 0, DIM);
                SetFlag<HardEvent::V_MTE3>(0);
                WaitFlag<HardEvent::V_MTE3>(0);
                DataCopy(part_gm[output * DIM], matrix, DIM);
                SetFlag<HardEvent::MTE3_V>(0);
                WaitFlag<HardEvent::MTE3_V>(0);
            }
        } else {
            for (uint32_t group = 0; group < groups; ++group) {
                uint32_t output = (bh * groups + group) * splits + split;
                Duplicate(matrix, 0.0f, DIM);
                SetFlag<HardEvent::V_MTE3>(0);
                WaitFlag<HardEvent::V_MTE3>(0);
                DataCopy(part_gm[output * DIM], matrix, DIM);
                SetFlag<HardEvent::MTE3_V>(0);
                WaitFlag<HardEvent::MTE3_V>(0);
            }
        }
        PipeBarrier<PIPE_ALL>();
    }
}
