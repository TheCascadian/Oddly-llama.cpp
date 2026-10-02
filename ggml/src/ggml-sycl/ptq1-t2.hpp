// ARC-LAB: PTQ1_0 weights repacked to the TernSYCL 2-bit layout and multiplied on XMX (s8 x s2 DPAS).
// See ptq1-t2.cpp.
#pragma once

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

// device bytes of the repacked weight: 2-bit codes [K/16, N] u32 + fp16 scales [K/128, N]
size_t ggml_sycl_t2_bytes(int64_t K, int64_t N);
// Convert canonical PTQ1 bytes in place; decline other encodings without modifying data.
bool ggml_sycl_t2_repack(sycl::queue & q, void * data, int64_t K, int64_t N);
// dst[m * N + n] = sum_k x[m * x_stride + k] * w[n, k], M tokens (fp32 in and out)
void   ggml_sycl_t2_mul_mat(sycl::queue & q, const void * w, const float * x, int64_t x_stride, float * dst, int64_t M,
                            int64_t N, int64_t K, void * scratch);
// Restore the standard PTQ1 encoding before generic tensor access.
void ggml_sycl_t2_restore(sycl::queue & q, void * data, int64_t K, int64_t N);
size_t ggml_sycl_t2_scratch_bytes(int64_t M, int64_t K);
