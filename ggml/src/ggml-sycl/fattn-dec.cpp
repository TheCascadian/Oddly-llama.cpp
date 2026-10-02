// ARC-LAB decode attention for a q4_0 KV cache (generation and MTP verify batches of 1-4 query tokens).
//
// The TILE path first converts the whole K and V cache to f16 on every call and then reads that copy
// (Bonsai 27B at 16K context: 5.9 + 11.4 ms per generated token over 16 attention layers). Here each
// work-group owns one KV head and a slice of the context, reads the q4_0 blocks directly, and serves all
// G query heads of that KV head (GQA) and all NQ query tokens from the one read (flash-decoding):
//   - the scaled query rows (G * NQ of them) sit in SLM;
//   - per tile of TK keys: thread (c, key) computes the G scores of query token c against its key,
//     straight from the q4_0 nibbles; per-row online softmax with a finite running max (a fully masked
//     tile or slice gets zero weight, never NaN); thread d accumulates P.V for head dim d of every row;
//   - slices are merged by flash_attn_combine_results (same partial layout as the vec kernel).

#include <sycl/sycl.hpp>
#include "dpct/helper.hpp"
#include "common.hpp"
#include "fattn-common.hpp"
#include "fattn-dec.hpp"
#include <algorithm>
#include <cfloat>
#include <cstring>
#include <cstdlib>

// Portable q4_0 KV-cache decode path adapted from Torchit1/llama.cpp arc-b580
// (commit 8160e3b). The separate Intel XMX/DPAS kernels are intentionally not
// included here; this file keeps the standard SYCL path loadable on OpenCL.

namespace {
constexpr int   DEC_D    = 256;       // head size (K and V)
constexpr int   DEC_WG   = 256;       // work-group size = DEC_D: thread d owns output dim d
constexpr float DEC_MINF = -1.0e30f;  // finite "minus infinity" for the running max
}

// v3 kernel, kept for 4-token batches: the pipelined kernel below is slower there (436 vs 508 us per layer at
// 16K; see README) and faster at 1-2 tokens (16K 273 -> 256, 48K 944 -> 742 us).
template <int G, int NQ>
static void fattn_dec_q4_0_v3(const char * Q, const char * K, const char * V, const char * mask, float * dst,
                           float * parts, sycl::float2 * meta, float scale, int ne01, int ne02, int ne11,
                           int nkvh, int ne03, int64_t nb01, int64_t nb02, int64_t nb03, int64_t nb11,
                           int64_t nb12, int64_t nb13, int64_t nb21, int64_t nb22, int64_t nb23, int64_t nb31,
                           int64_t nb33, int ne33, int nsplit, int chunk, dpct::queue_ptr stream) {
    constexpr int R   = G * NQ;             // query rows per KV head
    constexpr int RP  = (R + 3) / 4 * 4;    // padded row stride of the transposed scores
    constexpr int TK  = DEC_WG / NQ;        // keys per tile: thread (c, jj) = (tid / TK, tid % TK)
    constexpr int NB  = DEC_D / QK4_0;
    constexpr int KW  = DEC_D / QK4_0 * sizeof(block_q4_0) / 4;  // 36 dwords per q4_0 K row (D 256)
    constexpr int KWP = KW + 1;                                  // padded SLM row stride (bank conflicts)
    static_assert(TK % 16 == 0 && (DEC_D / QK4_0 * sizeof(block_q4_0)) % 4 == 0, "shape");

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<sycl::float4, 1> sQ(sycl::range<1>(R * DEC_D / 4), cgh);
        sycl::local_accessor<uint32_t, 1>     sK(sycl::range<1>(TK * KWP), cgh);
        sycl::local_accessor<sycl::float4, 1> sS(sycl::range<1>(TK * RP / 4), cgh);  // [key][row], transposed
        sycl::local_accessor<float, 1>        sM(sycl::range<1>(R), cgh);
        sycl::local_accessor<float, 1>        sL(sycl::range<1>(R), cgh);
        sycl::local_accessor<float, 1>        sA(sycl::range<1>(RP), cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(ne03, nkvh, (size_t) nsplit * DEC_WG), sycl::range<3>(1, 1, DEC_WG)),
            [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(16)]] {
                const int tid   = it.get_local_id(2);
                const int split = it.get_group(2);
                const int kvh   = it.get_group(1);
                const int seq   = it.get_group(0);
                auto      sg    = it.get_sub_group();
                const int w     = tid / 16;
                const int lane  = tid % 16;
                float *   sSf   = (float *) &sS[0];

                // query rows r = c * G + g: token c, head kvh * G + g; pre-scaled like the other FA kernels
                for (int e = tid; e < R * DEC_D / 4; e += DEC_WG) {
                    const int r = e / (DEC_D / 4), d4 = e % (DEC_D / 4), c = r / G, g = r % G;
                    sycl::float4 qv(0.0f);
                    if (c < ne01) {
                        qv = *(const sycl::float4 *) (Q + seq * nb03 + c * nb01 + (int64_t) (kvh * G + g) * nb02 + d4 * sizeof(sycl::float4));
                    }
                    sQ[e] = qv * scale;
                }
                if (tid < R) {
                    sM[tid] = DEC_MINF;
                    sL[tid] = 0.0f;
                }
                if (tid < RP) {
                    sA[tid] = 1.0f;
                }

                const int k_begin = split * chunk;
                const int k_end   = sycl::min(k_begin + chunk, ne11);
                const char * Kh = K + seq * nb13 + kvh * nb12;
                const char * Vh = V + seq * nb23 + kvh * nb22;
                const sycl::half * mrow = mask ? (const sycl::half *) (mask + (seq % ne33) * nb33) : nullptr;

                float o[RP];
#pragma unroll
                for (int r = 0; r < RP; ++r) {
                    o[r] = 0.0f;
                }

                const int c  = tid / TK;
                const int jj = tid % TK;
                for (int t0 = k_begin; t0 < k_end; t0 += TK) {
                    // K tile -> SLM: raw q4_0 rows, coalesced aligned dwords
                    for (int e = tid; e < TK * KW; e += DEC_WG) {
                        const int kk = e / KW, wd = e % KW, j = t0 + kk;
                        sK[kk * KWP + wd] = j < k_end ? ((const uint32_t *) (Kh + j * nb11))[wd] : 0u;
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    // scores: this thread's key against the G heads of query token c
                    {
                        const int j = t0 + jj;
                        float     s[G];
#pragma unroll
                        for (int g = 0; g < G; ++g) {
                            s[g] = 0.0f;
                        }
                        if (j < k_end && c < ne01) {
                            const int kr = jj * KWP;
                            const int q0 = c * G * (DEC_D / 4);
#pragma unroll 2
                            for (int b = 0; b < NB; ++b) {
                                // block b = bytes [18b, 18b + 18): scale (2 bytes) then 16 bytes of nibbles
                                auto byte_at = [&](int o) -> uint32_t {
                                    return (sK[kr + o / 4] >> (8 * (o % 4))) & 0xFFu;
                                };
                                const int      o0 = b * (int) sizeof(block_q4_0);
                                const uint16_t hb = (uint16_t) (byte_at(o0) | (byte_at(o0 + 1) << 8));
                                const float    dk = static_cast<float>(sycl::bit_cast<sycl::half>(hb));
#pragma unroll
                                for (int i = 0; i < QK4_0 / 2; i += 4) {
                                    sycl::float4 k0, k1;
#pragma unroll
                                    for (int u = 0; u < 4; ++u) {
                                        const uint32_t by = byte_at(o0 + 2 + i + u);
                                        k0[u] = (float) ((int) (by & 0xF) - 8) * dk;
                                        k1[u] = (float) ((int) (by >> 4) - 8) * dk;
                                    }
                                    const int d4 = (b * QK4_0 + i) / 4;
#pragma unroll
                                    for (int g = 0; g < G; ++g) {
                                        const sycl::float4 qa = sQ[q0 + g * (DEC_D / 4) + d4];
                                        const sycl::float4 qb = sQ[q0 + g * (DEC_D / 4) + d4 + QK4_0 / 8];
                                        s[g] += sycl::dot(qa, k0) + sycl::dot(qb, k1);
                                    }
                                }
                            }
                            const float mv = mrow ? static_cast<float>(mrow[(c * nb31) / (int64_t) sizeof(sycl::half) + j]) : 0.0f;
#pragma unroll
                            for (int g = 0; g < G; ++g) {
                                s[g] += mv;
                            }
                        } else {
#pragma unroll
                            for (int g = 0; g < G; ++g) {
                                s[g] = -INFINITY;
                            }
                        }
#pragma unroll
                        for (int g = 0; g < G; ++g) {
                            sSf[jj * RP + c * G + g] = s[g];
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    // V tile -> SLM (reusing the K tile buffer; scores are done with it)
                    static_assert(sizeof(block_q4_0) * NB % 4 == 0, "V rows are dword multiples");
                    for (int e = tid; e < TK * KW; e += DEC_WG) {
                        const int kk = e / KW, wd = e % KW, j = t0 + kk;
                        sK[kk * KWP + wd] = j < k_end ? ((const uint32_t *) (Vh + j * nb21))[wd] : 0u;
                    }

                    // online softmax, one row per subgroup at a time
                    for (int r = w; r < R; r += DEC_WG / 16) {
                        float mt = -INFINITY;
                        for (int k = lane; k < TK; k += 16) {
                            mt = sycl::fmax(mt, sSf[k * RP + r]);
                        }
                        mt = sycl::reduce_over_group(sg, mt, sycl::maximum<float>());
                        const float m_old = sM[r];
                        const float m_new = sycl::fmax(m_old, mt);  // finite: m_old starts at DEC_MINF
                        float       lt    = 0.0f;
                        for (int k = lane; k < TK; k += 16) {
                            const float p = sycl::native::exp(sSf[k * RP + r] - m_new);  // exp(-inf) = 0
                            sSf[k * RP + r] = p;
                            lt += p;
                        }
                        lt = sycl::reduce_over_group(sg, lt, sycl::plus<float>());
                        if (lane == 0) {
                            const float a = sycl::native::exp(m_old - m_new);
                            sA[r] = a;
                            sL[r] = sL[r] * a + lt;
                            sM[r] = m_new;
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    // P.V for head dim d = tid (padded rows R..RP-1 have P = 0 from the -inf scores)
                    {
                        const int d  = tid;
                        const int b  = d / QK4_0;
                        const int wi = d % QK4_0;
                        const int nk = sycl::min(TK, k_end - t0);
#pragma unroll
                        for (int r = 0; r < RP; ++r) {
                            o[r] *= sA[r];
                        }
                        const int ob = b * (int) sizeof(block_q4_0);          // scale bytes ob, ob + 1
                        const int oq = ob + 2 + wi % (QK4_0 / 2);             // this dim's nibble byte
                        const int sh = wi < QK4_0 / 2 ? 0 : 4;
#pragma unroll 4
                        for (int k = 0; k < nk; ++k) {
                            const int      vr   = k * KWP;
                            const uint32_t sw   = sK[vr + ob / 4] >> (8 * (ob % 4));  // ob even: both scale bytes in one dword
                            const float    dv   = static_cast<float>(sycl::bit_cast<sycl::half>((uint16_t) (sw & 0xFFFFu)));
                            const int      byte = (sK[vr + oq / 4] >> (8 * (oq % 4))) & 0xFF;
                            const float    vv   = (float) (((byte >> sh) & 0xF) - 8) * dv;
#pragma unroll
                            for (int r4 = 0; r4 < RP / 4; ++r4) {
                                const sycl::float4 p = sS[k * (RP / 4) + r4];
                                o[4 * r4 + 0] += p[0] * vv;
                                o[4 * r4 + 1] += p[1] * vv;
                                o[4 * r4 + 2] += p[2] * vv;
                                o[4 * r4 + 3] += p[3] * vv;
                            }
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }

#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const int cr = r / G;
                    if (cr >= ne01) {
                        continue;
                    }
                    const int64_t jdu = ((int64_t) seq * ne01 + cr) * ne02 + kvh * G + r % G;
                    if (nsplit == 1) {
                        const float l = sL[r];
                        dst[jdu * DEC_D + tid] = l > 0.0f ? o[r] / l : 0.0f;
                    } else {
                        parts[(jdu * nsplit + split) * DEC_D + tid] = o[r];
                        if (tid == 0) {
                            meta[jdu * nsplit + split] = sycl::float2(sM[r], sL[r]);
                        }
                    }
                }
            });
    });
}

template <int G, int NQ, int TK, bool PIPE>
static void fattn_dec_q4_0(const char * Q, const char * K, const char * V, const char * mask, float * dst,
                           float * parts, sycl::float2 * meta, float scale, int ne01, int ne02, int ne11,
                           int nkvh, int ne03, int64_t nb01, int64_t nb02, int64_t nb03, int64_t nb11,
                           int64_t nb12, int64_t nb13, int64_t nb21, int64_t nb22, int64_t nb23, int64_t nb31,
                           int64_t nb33, int ne33, int nsplit, int chunk, dpct::queue_ptr stream) {
    constexpr int R   = G * NQ;             // query rows per KV head
    constexpr int RP  = (R + 3) / 4 * 4;    // padded row stride of the transposed scores
    constexpr int NRG = DEC_WG / TK;        // row groups in the score phase: thread (rg, jj) = (tid / TK, tid % TK)
    constexpr int RPT = R / NRG;            // rows per thread in the score phase
    constexpr int NB  = DEC_D / QK4_0;
    constexpr int KW  = DEC_D / QK4_0 * sizeof(block_q4_0) / 4;  // 36 dwords per q4_0 row (D 256)
    constexpr int KWP = KW + 1;                                  // padded SLM row stride (bank conflicts)
    constexpr int PKW = TK * KW / DEC_WG;                        // dwords of a K or V tile per thread
    static_assert(TK % 16 == 0 && R % NRG == 0 && (TK * KW) % DEC_WG == 0 && G % RPT == 0, "shape");  // a thread's rows share one token

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<sycl::float4, 1> sQ(sycl::range<1>(R * DEC_D / 4), cgh);
        sycl::local_accessor<uint32_t, 1>     sK(sycl::range<1>(TK * KWP), cgh);   // K tile, then V tile
        sycl::local_accessor<sycl::float4, 1> sS(sycl::range<1>(TK * RP / 4), cgh);  // [key][row], transposed
        sycl::local_accessor<float, 1>        sM(sycl::range<1>(R), cgh);
        sycl::local_accessor<float, 1>        sL(sycl::range<1>(R), cgh);
        sycl::local_accessor<float, 1>        sA(sycl::range<1>(RP), cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(ne03, nkvh, (size_t) nsplit * DEC_WG), sycl::range<3>(1, 1, DEC_WG)),
            [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(16)]] {
                const int tid   = it.get_local_id(2);
                const int split = it.get_group(2);
                const int kvh   = it.get_group(1);
                const int seq   = it.get_group(0);
                auto      sg    = it.get_sub_group();
                const int w     = tid / 16;
                const int lane  = tid % 16;
                float *   sSf   = (float *) &sS[0];

                const int k_begin = split * chunk;
                const int k_end   = sycl::min(k_begin + chunk, ne11);
                const char * Kh = K + seq * nb13 + kvh * nb12;
                const char * Vh = V + seq * nb23 + kvh * nb22;
                const sycl::half * mrow = mask ? (const sycl::half *) (mask + (seq % ne33) * nb33) : nullptr;

                // software pipeline: a tile's raw rows are loaded into registers one phase before they are stored
                // to SLM (V during Q.K, the next K during softmax + P.V), so the loads overlap compute
                uint32_t pk[PKW], pv[PKW];
                auto load_rows = [&](const char * base, int64_t nb, int t0, uint32_t * reg) {
#pragma unroll
                    for (int i = 0; i < PKW; ++i) {
                        const int e = tid + i * DEC_WG, kk = e / KW, wd = e % KW, j = t0 + kk;
                        reg[i] = j < k_end ? ((const uint32_t *) (base + j * nb))[wd] : 0u;
                    }
                };
                auto store_rows = [&](const uint32_t * reg) {
#pragma unroll
                    for (int i = 0; i < PKW; ++i) {
                        const int e = tid + i * DEC_WG, kk = e / KW, wd = e % KW;
                        sK[kk * KWP + wd] = reg[i];
                    }
                };
                if constexpr (PIPE) {
                    load_rows(Kh, nb11, k_begin, pk);
                }

                // query rows r = c * G + g: token c, head kvh * G + g; pre-scaled like the other FA kernels
                for (int e = tid; e < R * DEC_D / 4; e += DEC_WG) {
                    const int r = e / (DEC_D / 4), d4 = e % (DEC_D / 4), c = r / G, g = r % G;
                    sycl::float4 qv(0.0f);
                    if (c < ne01) {
                        qv = *(const sycl::float4 *) (Q + seq * nb03 + c * nb01 + (int64_t) (kvh * G + g) * nb02 + d4 * sizeof(sycl::float4));
                    }
                    sQ[e] = qv * scale;
                }
                if (tid < R) {
                    sM[tid] = DEC_MINF;
                    sL[tid] = 0.0f;
                }
                if (tid < RP) {
                    sA[tid] = 1.0f;
                }

                float o[RP];
#pragma unroll
                for (int r = 0; r < RP; ++r) {
                    o[r] = 0.0f;
                }

                const int rg = tid / TK;
                const int jj = tid % TK;
                const int r0 = rg * RPT;
                for (int t0 = k_begin; t0 < k_end; t0 += TK) {
                    if constexpr (!PIPE) {
                        load_rows(Kh, nb11, t0, pk);
                    }
                    store_rows(pk);                  // K tile t
                    it.barrier(sycl::access::fence_space::local_space);
                    if constexpr (PIPE) {
                        load_rows(Vh, nb21, t0, pv);  // V tile t, in flight during Q.K
                    }

                    // scores of rows r0 .. r0 + RPT - 1 against this thread's key
                    {
                        const int j = t0 + jj;
                        float     s[RPT];
#pragma unroll
                        for (int i = 0; i < RPT; ++i) {
                            s[i] = 0.0f;
                        }
                        if (j < k_end) {
                            const int kr = jj * KWP;
                            const int qb = r0 * (DEC_D / 4);  // hoisted row base (v3 form): row rr at qb + rr * 64
#pragma unroll 2
                            for (int b = 0; b < NB; ++b) {
                                auto byte_at = [&](int o) -> uint32_t {
                                    return (sK[kr + o / 4] >> (8 * (o % 4))) & 0xFFu;
                                };
                                const int      o0 = b * (int) sizeof(block_q4_0);
                                const uint16_t hb = (uint16_t) (byte_at(o0) | (byte_at(o0 + 1) << 8));
                                const float    dk = static_cast<float>(sycl::bit_cast<sycl::half>(hb));
#pragma unroll
                                for (int i = 0; i < QK4_0 / 2; i += 4) {
                                    sycl::float4 k0, k1;
#pragma unroll
                                    for (int u = 0; u < 4; ++u) {
                                        const uint32_t by = byte_at(o0 + 2 + i + u);
                                        k0[u] = (float) ((int) (by & 0xF) - 8) * dk;
                                        k1[u] = (float) ((int) (by >> 4) - 8) * dk;
                                    }
                                    const int d4 = (b * QK4_0 + i) / 4;
#pragma unroll
                                    for (int rr = 0; rr < RPT; ++rr) {
                                        const sycl::float4 qa = sQ[qb + rr * (DEC_D / 4) + d4];
                                        const sycl::float4 qc = sQ[qb + rr * (DEC_D / 4) + d4 + QK4_0 / 8];
                                        s[rr] += sycl::dot(qa, k0) + sycl::dot(qc, k1);
                                    }
                                }
                            }
                        }
                        const int   c  = r0 / G;  // all RPT rows of this thread belong to token c
                        const float mv = (j < k_end && c < ne01 && mrow) ? static_cast<float>(mrow[(c * nb31) / (int64_t) sizeof(sycl::half) + j]) : 0.0f;
#pragma unroll
                        for (int rr = 0; rr < RPT; ++rr) {
                            sSf[jj * RP + r0 + rr] = (j < k_end && c < ne01) ? s[rr] + mv : -INFINITY;
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    if constexpr (!PIPE) {
                        load_rows(Vh, nb21, t0, pv);
                    }
                    store_rows(pv);                  // V tile t -> SLM (the K tile is done)
                    if constexpr (PIPE) {
                        if (t0 + TK < k_end) {
                            load_rows(Kh, nb11, t0 + TK, pk);  // K tile t + 1, in flight during softmax + P.V
                        }
                    }

                    // online softmax, one row per subgroup at a time
                    for (int r = w; r < R; r += DEC_WG / 16) {
                        float mt = -INFINITY;
                        for (int k = lane; k < TK; k += 16) {
                            mt = sycl::fmax(mt, sSf[k * RP + r]);
                        }
                        mt = sycl::reduce_over_group(sg, mt, sycl::maximum<float>());
                        const float m_old = sM[r];
                        const float m_new = sycl::fmax(m_old, mt);  // finite: m_old starts at DEC_MINF
                        float       lt    = 0.0f;
                        for (int k = lane; k < TK; k += 16) {
                            const float p = sycl::native::exp(sSf[k * RP + r] - m_new);  // exp(-inf) = 0
                            sSf[k * RP + r] = p;
                            lt += p;
                        }
                        lt = sycl::reduce_over_group(sg, lt, sycl::plus<float>());
                        if (lane == 0) {
                            const float a = sycl::native::exp(m_old - m_new);
                            sA[r] = a;
                            sL[r] = sL[r] * a + lt;
                            sM[r] = m_new;
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    // P.V for head dim d = tid (padded rows R..RP-1 are never written out)
                    {
                        const int d  = tid;
                        const int b  = d / QK4_0;
                        const int wi = d % QK4_0;
                        const int nk = sycl::min(TK, k_end - t0);
#pragma unroll
                        for (int r = 0; r < RP; ++r) {
                            o[r] *= sA[r];
                        }
                        const int ob = b * (int) sizeof(block_q4_0);          // scale bytes ob, ob + 1
                        const int oq = ob + 2 + wi % (QK4_0 / 2);             // this dim's nibble byte
                        const int sh = wi < QK4_0 / 2 ? 0 : 4;
#pragma unroll 4
                        for (int k = 0; k < nk; ++k) {
                            const int      vr   = k * KWP;
                            const uint32_t sw   = sK[vr + ob / 4] >> (8 * (ob % 4));  // ob even: both scale bytes in one dword
                            const float    dv   = static_cast<float>(sycl::bit_cast<sycl::half>((uint16_t) (sw & 0xFFFFu)));
                            const int      byte = (sK[vr + oq / 4] >> (8 * (oq % 4))) & 0xFF;
                            const float    vv   = (float) (((byte >> sh) & 0xF) - 8) * dv;
#pragma unroll
                            for (int r4 = 0; r4 < RP / 4; ++r4) {
                                const sycl::float4 p = sS[k * (RP / 4) + r4];
                                o[4 * r4 + 0] += p[0] * vv;
                                o[4 * r4 + 1] += p[1] * vv;
                                o[4 * r4 + 2] += p[2] * vv;
                                o[4 * r4 + 3] += p[3] * vv;
                            }
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }

#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const int cr = r / G;
                    if (cr >= ne01) {
                        continue;
                    }
                    const int64_t jdu = ((int64_t) seq * ne01 + cr) * ne02 + kvh * G + r % G;
                    if (nsplit == 1) {
                        const float l = sL[r];
                        dst[jdu * DEC_D + tid] = l > 0.0f ? o[r] / l : 0.0f;
                    } else {
                        parts[(jdu * nsplit + split) * DEC_D + tid] = o[r];
                        if (tid == 0) {
                            meta[jdu * nsplit + split] = sycl::float2(sM[r], sL[r]);
                        }
                    }
                }
            });
    });
}
bool ggml_sycl_flash_attn_ext_dec_supported(const ggml_tensor * dst) {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_SYCL_FA_Q4_DIRECT");
        return value && std::atoi(value) != 0;
    }();
    if (!enabled || !dst || !dst->src[0] || !dst->src[1] || !dst->src[2]) {
        return false;
    }

    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    float max_bias = 0.0f;
    float softcap  = 0.0f;
    std::memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    std::memcpy(&softcap, (const float *) dst->op_params + 2, sizeof(float));

    return dst->type == GGML_TYPE_F32 && K->type == GGML_TYPE_Q4_0 && V->type == GGML_TYPE_Q4_0 &&
           Q->type == GGML_TYPE_F32 && K->ne[0] == DEC_D && V->ne[0] == K->ne[0] && Q->ne[0] == K->ne[0] &&
           Q->ne[2] == 6 * K->ne[2] && V->ne[2] == K->ne[2] && Q->ne[1] >= 1 && Q->ne[1] <= 4 &&
           Q->ne[3] == K->ne[3] && V->ne[3] == K->ne[3] && !sinks && max_bias == 0.0f && softcap == 0.0f &&
           (!mask || (mask->type == GGML_TYPE_F16 && mask->ne[2] == 1)) &&
           Q->nb[0] == sizeof(float) && K->nb[0] == ggml_type_size(K->type) && V->nb[0] == ggml_type_size(V->type) &&
           K->nb[1] % 4 == 0 && K->nb[2] % 4 == 0 && K->nb[3] % 4 == 0 && ((uintptr_t) K->data) % 4 == 0 &&
           V->nb[1] % 4 == 0 && V->nb[2] % 4 == 0 && V->nb[3] % 4 == 0 && ((uintptr_t) V->data) % 4 == 0 &&
           Q->nb[1] % 16 == 0 && Q->nb[2] % 16 == 0 && Q->nb[3] % 16 == 0 && ((uintptr_t) Q->data) % 16 == 0;
}

void ggml_sycl_flash_attn_ext_dec(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float scale = 1.0f;
    std::memcpy(&scale, (const float *) dst->op_params, sizeof(float));

    const int ne01 = Q->ne[1];
    const int nq   = ne01 <= 1 ? 1 : ne01 <= 2 ? 2 : 4;
    const int tk   = nq == 4 ? 64 : 128;
    const int ne11 = K->ne[1];

    static const int target_env = [] {
        const char * value = std::getenv("GGML_SYCL_FA_DEC_SPLITS");
        return value ? std::atoi(value) : 0;
    }();
    const int target = target_env ? target_env : std::max(64, 256 / std::max(1, (int) K->ne[2]));
    const int ntiles = (ne11 + tk - 1) / tk;
    int nsplit       = std::max(1, std::min(target, ntiles));
    const int chunk  = ((ntiles + nsplit - 1) / nsplit) * tk;
    nsplit           = (ne11 + chunk - 1) / chunk;

    ggml_sycl_pool_alloc<float>        parts(ctx.pool());
    ggml_sycl_pool_alloc<sycl::float2> meta(ctx.pool());
    if (nsplit > 1) {
        parts.alloc((size_t) nsplit * ggml_nelements(dst));
        meta.alloc((size_t) nsplit * ggml_nrows(dst));
    }

    dpct::queue_ptr stream = ctx.stream();
    const char *    mdata  = mask ? (const char *) mask->data : nullptr;
    const int64_t   nb31   = mask ? mask->nb[1] : 0;
    const int64_t   nb33   = mask ? mask->nb[3] : 0;
    const int       ne33   = mask ? (int) mask->ne[3] : 1;

#define FATTN_DEC_ARGS \
    (const char *) Q->data, (const char *) K->data, (const char *) V->data, mdata, (float *) dst->data, \
        parts.get(), meta.get(), scale, ne01, (int) Q->ne[2], ne11, (int) K->ne[2], (int) Q->ne[3], \
        Q->nb[1], Q->nb[2], Q->nb[3], K->nb[1], K->nb[2], K->nb[3], V->nb[1], V->nb[2], V->nb[3], \
        nb31, nb33, ne33, nsplit, chunk, stream
    if (nq == 1) {
        fattn_dec_q4_0<6, 1, 128, true>(FATTN_DEC_ARGS);
    } else if (nq == 2) {
        fattn_dec_q4_0<6, 2, 128, true>(FATTN_DEC_ARGS);
    } else {
        fattn_dec_q4_0_v3<6, 4>(FATTN_DEC_ARGS);
    }
#undef FATTN_DEC_ARGS

    if (nsplit > 1) {
        const sycl::range<3> grid(Q->ne[3], Q->ne[2], (size_t) ne01 * DEC_D);
        const size_t         nbytes_shared = nsplit * sizeof(sycl::float2);
        float *              parts_p       = parts.get();
        sycl::float2 *       meta_p        = meta.get();
        float *              dst_p         = (float *) dst->data;
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<uint8_t, 1> lm(sycl::range<1>(nbytes_shared), cgh);
            cgh.parallel_for(sycl::nd_range<3>(grid, sycl::range<3>(1, 1, DEC_D)),
                             [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(16)]] {
                                 flash_attn_combine_results<DEC_D>(parts_p, meta_p, dst_p, nsplit,
                                     lm.get_multi_ptr<sycl::access::decorated::no>().get());
                             });
        });
    }
}
