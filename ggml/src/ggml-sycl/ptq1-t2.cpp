// Adapted from Torchit arc-b580 8160e3b, using TernSYCL int2 x int8 DPAS (BSD-3-Clause).
// See ternsycl/NOTICE. Activations use int8 values with fp16 scales.
#include "ptq1-t2.hpp"

#include "ternsycl/int2_int8_dpas.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

constexpr int     QK        = 128;
constexpr int64_t BLK_BYTES = 28;  // block_ptq1_0: qs[24], qh[2], fp16 d

}  // namespace

size_t ggml_sycl_t2_bytes(int64_t K, int64_t N) {
    return (size_t) (K / 16) * N * 4 + (size_t) (K / QK) * N * 2;
}

bool ggml_sycl_t2_repack(sycl::queue & q, void * data, int64_t K, int64_t N) {
    const int64_t nb        = K / QK;
    const int64_t blk_bytes = BLK_BYTES;
    const size_t  src_bytes = (size_t) N * nb * blk_bytes;
    uint8_t *     tmp       = (uint8_t *) sycl::malloc_device(src_bytes, q);
    if (!tmp) {
        fprintf(stderr, "%s: no device memory for a %zu-byte repack buffer\n", __func__, src_bytes);
        abort();
    }
    q.memcpy(tmp, data, src_bytes).wait_and_throw();
    uint32_t * B   = (uint32_t *) data;
    uint16_t * SB  = (uint16_t *) ((char *) data + (size_t) (K / 16) * N * 4);
    int *      bad = sycl::malloc_device<int>(1, q);
    q.memset(bad, 0, sizeof(int)).wait_and_throw();
    // Decline noncanonical bytes so restoring PTQ1 remains byte-exact.
    {
        q.parallel_for(sycl::range<1>((size_t) N * nb * 26), [=](sycl::id<1> it) {
            const size_t block = it[0] / 26, b = it[0] % 26;
            const unsigned original = tmp[block * BLK_BYTES + b];
            unsigned v = original, digits = 0;
            for (int j = 0; j < (b < 24 ? 5 : 4); ++j) {
                v *= 3;
                digits = digits * 3 + (v >> 8);
                v &= 255;
            }
            if (b >= 24) digits *= 3;
            if ((digits * 256 + 242) / 243 != original) {
                sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>(*bad).store(1);
            }
        }).wait_and_throw();
        int invalid = 0;
        q.memcpy(&invalid, bad, sizeof(int)).wait_and_throw();
        if (invalid) {
            sycl::free(bad, q);
            sycl::free(tmp, q);
            return false;
        }
    }
    q.parallel_for(sycl::range<2>((size_t) (K / 16), (size_t) N), [=](sycl::item<2> it) {
         const int64_t   kp = it[0], n = it[1];
         const int64_t   g  = kp / 8;
         const uint8_t * b  = tmp + (n * nb + g) * blk_bytes;
         const int       e0 = (int) (kp % 8) * 16;
         uint32_t        w  = 0;
         for (int j = 0; j < 16; ++j) {
             const int e = e0 + j;
             uint32_t  digit;  // weight = digit - 1
             {
                 uint32_t v;
                 int      lvl;
                 if (e < 80) {
                     v   = b[e % 16];
                     lvl = e / 16;
                 } else if (e < 120) {
                     v   = b[16 + (e - 80) % 8];
                     lvl = (e - 80) / 8;
                 } else {
                     v   = b[24 + (e - 120) % 2];
                     lvl = (e - 120) / 2;
                 }
                 digit = 0;
                 for (int t = 0; t <= lvl; ++t) {  // base-3 fixed-point digits, as ptq1_0_decode_block
                     const uint32_t x = v * 3;
                     digit            = x >> 8;
                     v                = x & 0xFF;
                 }
             }
             const uint32_t code = digit == 2 ? 1u : (digit == 1 ? 0u : 3u);  // 2-bit two's complement of digit - 1
             w |= code << (2 * j);
         }
         B[kp * N + n] = w;
         if (kp % 8 == 0) {
             SB[g * N + n] = (uint16_t) (b[26] | (b[27] << 8));
         }
     }).wait_and_throw();
    sycl::free(bad, q);
    sycl::free(tmp, q);
    return true;
}

namespace {

// GGML_SYCL_PTQ1_T2_S32: activation scales per 32 (q8_1 granularity) instead of per 128: 1 = GEMV (<= 8 tokens) only,
// 2 = GEMV + GEMM; 0 = TernSYCL's per 128
int t2_s32() {
    static const int v = getenv("GGML_SYCL_PTQ1_T2_S32") ? atoi(getenv("GGML_SYCL_PTQ1_T2_S32")) : 0;
    return v;
}

// fp32 activations -> int8 Aq [M, K] + fp16 SA [K/128, ldsa(M)] (TernSYCL QMODE 0: SA = 127 / absmax, saturate)
// G = 128 (one scale per 16 lanes x 8 values) or 32 (a scale per 4 lanes); SA [K/G, ldsa(M)]
void quant_a(sycl::queue & q, const float * x, int64_t x_stride, int64_t M, int64_t K, int8_t * Aq, uint16_t * SA,
             const int G = QK) {
    const int lda = int8dpas::ldsa((int) M);
    q.parallel_for(sycl::nd_range<2>({ (size_t) M, (size_t) (K / QK) * 16 }, { 1, 16 }),
                   [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                       const auto    sg   = it.get_sub_group();
                       const int     lane = sg.get_local_linear_id();
                       const int64_t m    = it.get_global_id(0);
                       const int64_t g    = it.get_global_id(1) / 16;
                       const float * a    = x + m * x_stride + g * QK + 8 * lane;
                       float         v[8], mx = 0.0f;
#pragma unroll
                       for (int i = 0; i < 8; ++i) {
                           v[i] = a[i];
                           mx   = sycl::fmax(mx, sycl::fabs(v[i]));
                       }
                       if (G == QK) {
                           mx = sycl::reduce_over_group(sg, mx, sycl::maximum<float>());
                       } else {  // 32-value groups = 4 lanes
                           mx = sycl::fmax(mx, sycl::permute_group_by_xor(sg, mx, 1));
                           mx = sycl::fmax(mx, sycl::permute_group_by_xor(sg, mx, 2));
                       }
                       const sycl::half sh = sycl::half(sycl::fmin(65504.0f, 127.0f / sycl::fmax(mx, int8dpas::EPS)));
                       if (G == QK ? lane == 0 : (lane & 3) == 0) {
                           const int64_t gi = G == QK ? g : g * 4 + lane / 4;
                           SA[gi * lda + m] = sycl::bit_cast<uint16_t>(sh);
                       }
                       const float s = (float) sh;
                       uint64_t    p = 0;
#pragma unroll
                       for (int i = 0; i < 8; ++i) {
                           // round to nearest (TernSYCL truncates: RTZ biases every product toward zero)
                           p |= (uint64_t) (uint8_t) (int8_t) sycl::rint(sycl::clamp(v[i] * s, -128.0f, 127.0f)) << (8 * i);
                       }
                       *(uint64_t *) (Aq + m * K + g * QK + 8 * lane) = p;
                   });
}

template <int SGM, int LS, int NS = 1>
void gemv(sycl::queue & q, const int8_t * Aq, const uint16_t * SA, const uint32_t * B, const uint16_t * SB, float * C,
          int M, int N, int K) {
    using Kern          = int8dpas::Gemv<false, 0, SGM, 2, LS, 2, NS>;
    const size_t    wgn = 16 * 2;
    const Epi       epi{ nullptr, nullptr, 0, 1 };
    const sycl::range<2> local(1, Kern::WG);
    const sycl::range<2> global((M + SGM - 1) / SGM, (N + wgn - 1) / wgn * Kern::WG);
    q.parallel_for(sycl::nd_range<2>(global, local),
                   Kern{ nullptr, (const signed char *) Aq, SA, B, SB, C, epi, M, N, K });
}

template <int NS, int MT_M, int MT_N, int WG_M, int WG_N>
void gemm_tile(sycl::queue & q, const int8_t * Aq, const uint16_t * SA, const uint32_t * B, const uint16_t * SB, float * C,
               int M, int N, int K) {
    const size_t    tm = MT_M * WG_M, tn = MT_N * WG_N;
    const Epi       epi{ nullptr, nullptr, 0, 1 };
    const sycl::range<2> local(1, 16 * WG_M * WG_N);
    const sycl::range<2> global((M + tm - 1) / tm, (N + tn - 1) / tn * local[1]);
    q.parallel_for(sycl::nd_range<2>(global, local),
                   int8dpas::GemmMT<false, 0, MT_M, MT_N, WG_M, WG_N, 0, true, NS>{
                       nullptr, (const signed char *) Aq, SA, B, SB, C, epi, M, N, K });
}

// ARC-LAB lab knob GGML_SYCL_PTQ1_T2_TILE = index into TernSYCL's large-M tile table (mt_m, mt_n, wg_m, wg_n):
// 0 {8,128,8,2} (default) 1 {8,128,4,2} 2 {8,128,4,4} 3 {8,128,16,1} 4 {8,128,2,4} 5 {16,64,4,2} 6 {16,64,8,2}
// 7 {8,64,8,2} 8 {32,32,4,2}
template <int NS = 1>
void gemm(sycl::queue & q, const int8_t * Aq, const uint16_t * SA, const uint32_t * B, const uint16_t * SB, float * C,
          int M, int N, int K) {
    static const int tile = getenv("GGML_SYCL_PTQ1_T2_TILE") ? atoi(getenv("GGML_SYCL_PTQ1_T2_TILE")) : 0;
    switch (tile) {
        case 1: gemm_tile<NS, 8, 128, 4, 2>(q, Aq, SA, B, SB, C, M, N, K); break;
        case 2: gemm_tile<NS, 8, 128, 4, 4>(q, Aq, SA, B, SB, C, M, N, K); break;
        case 3: gemm_tile<NS, 8, 128, 16, 1>(q, Aq, SA, B, SB, C, M, N, K); break;
        case 4: gemm_tile<NS, 8, 128, 2, 4>(q, Aq, SA, B, SB, C, M, N, K); break;
        case 5: gemm_tile<NS, 16, 64, 4, 2>(q, Aq, SA, B, SB, C, M, N, K); break;
        case 6: gemm_tile<NS, 16, 64, 8, 2>(q, Aq, SA, B, SB, C, M, N, K); break;
        case 7: gemm_tile<NS, 8, 64, 8, 2>(q, Aq, SA, B, SB, C, M, N, K); break;
        case 8: gemm_tile<NS, 32, 32, 4, 2>(q, Aq, SA, B, SB, C, M, N, K); break;
        default: gemm_tile<NS, 8, 128, 8, 2>(q, Aq, SA, B, SB, C, M, N, K); break;
    }
}

}  // namespace

namespace {
template <int NS>
void t2_dispatch(sycl::queue & q, const int8_t * Aq, const uint16_t * SA, const uint32_t * B, const uint16_t * SB,
                 float * dst, int m, int n, int k, bool use_gemm) {
    if (use_gemm) {
        gemm<NS>(q, Aq, SA, B, SB, dst, m, n, k);
    } else if (m == 1) {
        n <= 8192 ? gemv<1, 4, NS>(q, Aq, SA, B, SB, dst, m, n, k) : gemv<1, 2, NS>(q, Aq, SA, B, SB, dst, m, n, k);
    } else if (m == 2) {
        n <= 8192 ? gemv<2, 4, NS>(q, Aq, SA, B, SB, dst, m, n, k) : gemv<2, 2, NS>(q, Aq, SA, B, SB, dst, m, n, k);
    } else if (m <= 4) {
        n <= 8192 ? gemv<4, 4, NS>(q, Aq, SA, B, SB, dst, m, n, k) : gemv<4, 2, NS>(q, Aq, SA, B, SB, dst, m, n, k);
    } else {
        n <= 8192 ? gemv<8, 4, NS>(q, Aq, SA, B, SB, dst, m, n, k) : gemv<8, 2, NS>(q, Aq, SA, B, SB, dst, m, n, k);
    }
}
}  // namespace

void ggml_sycl_t2_mul_mat(sycl::queue & q, const void * w, const float * x, int64_t x_stride, float * dst, int64_t M,
                          int64_t N, int64_t K, void * scratch) {
    // batches > 8 (n-gram verify, prompts): GemmMT reads each weight once per 64 rows. Its SA read, A read and C write
    // were switched from 2D block I/O to plain / sub-group block access (the 2D forms misbehaved in this JIT build).
    static const bool gemm_off = getenv("GGML_SYCL_PTQ1_T2_GEMM_OFF") != nullptr;
    const bool use_gemm = M > 8 && !gemm_off;
    const bool s32      = use_gemm ? t2_s32() >= 2 : t2_s32() >= 1;

    const size_t   aq_bytes = (size_t) M * K;
    const size_t   sa_off   = (aq_bytes + 255) & ~(size_t) 255;
    char *         s        = (char *) scratch;
    int8_t *       Aq       = (int8_t *) s;
    uint16_t *     SA       = (uint16_t *) (s + sa_off);
    const uint32_t * B      = (const uint32_t *) w;
    const uint16_t * SB     = (const uint16_t *) ((const char *) w + (size_t) (K / 16) * N * 4);

    quant_a(q, x, x_stride, M, K, Aq, SA, s32 ? 32 : QK);
    if (s32) {
        t2_dispatch<4>(q, Aq, SA, B, SB, dst, (int) M, (int) N, (int) K, use_gemm);
    } else {
        t2_dispatch<1>(q, Aq, SA, B, SB, dst, (int) M, (int) N, (int) K, use_gemm);
    }
}


void ggml_sycl_t2_restore(sycl::queue & q, void * data, int64_t K, int64_t N) {
    const size_t blocks = (size_t) N * (K / QK);
    auto * tmp = sycl::malloc_device<uint8_t>(blocks * BLK_BYTES, q);
    if (!tmp) throw std::bad_alloc();
    const auto * B = (const uint32_t *) data;
    const auto * SB = (const uint16_t *) ((const char *) data + (K / 16) * N * 4);
    try {
        q.parallel_for(sycl::range<1>(blocks), [=](sycl::id<1> it) {
            const size_t i = it[0], g = i % (K / QK), n = i / (K / QK);
            uint8_t * out = tmp + i * BLK_BYTES;
            for (int b = 0; b < 26; ++b) {
                const int count = b < 24 ? 5 : 4;
                const int stride = b < 16 ? 16 : b < 24 ? 8 : 2;
                const int first = b < 16 ? b : b < 24 ? 80 + b - 16 : 120 + b - 24;
                unsigned value = 0;
                for (int j = 0; j < count; ++j) {
                    const int e = first + j * stride;
                    const unsigned code = (B[(g * 8 + e / 16) * N + n] >> (2 * (e % 16))) & 3;
                    value = value * 3 + (code == 3 ? 0 : code + 1);
                }
                if (count == 4) value *= 3;
                out[b] = (value * 256 + 242) / 243;
            }
            const uint16_t scale = SB[g * N + n];
            out[26] = scale & 255;
            out[27] = scale >> 8;
        }).wait_and_throw();
        q.memcpy(data, tmp, blocks * BLK_BYTES).wait_and_throw();
    } catch (...) {
        sycl::free(tmp, q);
        throw;
    }
    sycl::free(tmp, q);
}

size_t ggml_sycl_t2_scratch_bytes(int64_t M, int64_t K) {
    return (((size_t) M * K + 255) & ~(size_t) 255) + (size_t) (K / 32) * int8dpas::ldsa((int) M) * 2 + 256;
}
