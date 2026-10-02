#include "fwht.hpp"

#include <cmath>
#include <cstdlib>

template <int N>
static void fwht_kernel(const float * __restrict__ src, float * __restrict__ dst, const int64_t n_rows,
                        const float scale, const sycl::nd_item<2> & item) {
    const sycl::sub_group sg = item.get_sub_group();

    const int64_t r = item.get_global_id(0);
    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    constexpr int el_w = N / WARP_SIZE;
    static_assert(el_w >= 1 && N % WARP_SIZE == 0, "row must be a whole number of sub-group widths");

    float     reg[el_w];
    const int lane = sg.get_local_linear_id();

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        reg[i] = src[i * WARP_SIZE + lane] * scale;
    }

    // Butterflies inside the sub-group. The partner of a lane with bit h clear is the
    // lower index of the pair, so it takes the sum and the upper takes lower - upper.
#pragma unroll
    for (int h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; ++j) {
            const float val  = reg[j];
            const float val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_SIZE);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // Butterflies across registers: h is a multiple of WARP_SIZE, so the partner of
    // element i*WARP_SIZE + lane lives in reg[i + h/WARP_SIZE] on the same lane.
#pragma unroll
    for (int h = WARP_SIZE; h < N; h *= 2) {
        const int step = h / WARP_SIZE;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; ++k) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        dst[i * WARP_SIZE + lane] = reg[i];
    }
}

template <int N>
static void launch_fwht(const float * src, float * dst, const int64_t n_rows, const float scale,
                        dpct::queue_ptr stream) {
    constexpr int rows_per_block = 4;

    const int64_t num_blocks = (n_rows + rows_per_block - 1) / rows_per_block;

    // dim 1 is the fastest-varying, so a sub-group is exactly one row's WARP_SIZE lanes.
    const sycl::range<2> global(num_blocks * rows_per_block, WARP_SIZE);
    const sycl::range<2> local(rows_per_block, WARP_SIZE);

    stream->parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             fwht_kernel<N>(src, dst, n_rows, scale, item);
                         });
}

// Wide blocks: one row per work-group instead of per sub-group, so each work-item
// keeps N/NT values rather than N/WARP_SIZE. Butterflies below the sub-group width
// still shuffle; those up to NT go through work-group local memory; the rest stay
// in registers.
template <int N, int NT>
static void fwht_kernel_wide(const float * __restrict__ src,
                             float * __restrict__ dst,
                             const int64_t            n_rows,
                             const float              scale,
                             const sycl::nd_item<2> & item,
                             float *                  smem) {
    const int64_t r = item.get_global_id(0);
    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    constexpr int el_w = N / NT;
    static_assert(el_w >= 1 && N % NT == 0, "row must be a whole number of work-group widths");

    const int tid = item.get_local_id(1);

    float reg[el_w];
#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        reg[i] = src[i * NT + tid] * scale;
    }

    const sycl::sub_group sg   = item.get_sub_group();
    const int             lane = sg.get_local_linear_id();

    // Butterflies inside the sub-group, same pattern as the narrow kernel.
#pragma unroll
    for (int h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; ++j) {
            const float val  = reg[j];
            const float val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_SIZE);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // Butterflies from the sub-group width up to NT: the partner lane is outside
    // this sub-group, so it goes through work-group local memory instead of a shuffle.
    for (int h = WARP_SIZE; h < NT; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; ++j) {
            smem[j * NT + tid] = reg[j];
        }
        item.barrier(sycl::access::fence_space::local_space);
#pragma unroll
        for (int j = 0; j < el_w; ++j) {
            const float val  = reg[j];
            const float val2 = smem[j * NT + (tid ^ h)];
            reg[j]           = (tid & h) == 0 ? val + val2 : val2 - val;
        }
        item.barrier(sycl::access::fence_space::local_space);
    }

    // Butterflies across registers: h is a multiple of NT, so the partner of element
    // i*NT + tid lives in reg[i + h/NT] on the same work-item.
    for (int h = NT; h < N; h *= 2) {
        const int step = h / NT;
        for (int j = 0; j < el_w; j += 2 * step) {
            for (int k = 0; k < step; ++k) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        dst[i * NT + tid] = reg[i];
    }
}

template <int N, int NT>
static void launch_fwht_wide(const float *   src,
                             float *         dst,
                             const int64_t   n_rows,
                             const float     scale,
                             dpct::queue_ptr stream) {
    const sycl::range<2> global(n_rows, NT);
    const sycl::range<2> local(1, NT);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> smem(sycl::range<1>(N), cgh);
        cgh.parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             fwht_kernel_wide<N, NT>(src, dst, n_rows, scale, item, get_pointer(smem));
                         });
    });
}


bool ggml_sycl_op_fwht(ggml_backend_sycl_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    if (src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_are_same_shape(src, dst)) {
        return false;
    }
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const int     n    = (int) src->ne[0];
    const int64_t rows = ggml_nrows(src);

    const float *   src_d  = (const float *) src->data;
    float *         dst_d  = (float *) dst->data;
    dpct::queue_ptr stream = ctx.stream();

    const float scale = 1.0f / std::sqrt((float) n);

    switch (n) {
        case 64:
            launch_fwht<64>(src_d, dst_d, rows, scale, stream);
            return true;
        case 128:
            launch_fwht<128>(src_d, dst_d, rows, scale, stream);
            return true;
        case 256:
            launch_fwht<256>(src_d, dst_d, rows, scale, stream);
            return true;
        case 512:
            launch_fwht<512>(src_d, dst_d, rows, scale, stream);
            return true;
        case 1024:
            if (std::getenv("GGML_SYCL_DISABLE_FWHT_WIDE") != nullptr) return false;
            GGML_SYCL_DEBUG("[SYCL] dispatch wide FWHT n=1024 rows=%lld\n", (long long) rows);
            launch_fwht_wide<1024, 256>(src_d, dst_d, rows, scale, stream);
            return true;
        case 2048:
            if (std::getenv("GGML_SYCL_DISABLE_FWHT_WIDE") != nullptr) return false;
            GGML_SYCL_DEBUG("[SYCL] dispatch wide FWHT n=2048 rows=%lld\n", (long long) rows);
            launch_fwht_wide<2048, 256>(src_d, dst_d, rows, scale, stream);
            return true;
        case 4096:
            if (std::getenv("GGML_SYCL_DISABLE_FWHT_WIDE") != nullptr) return false;
            GGML_SYCL_DEBUG("[SYCL] dispatch wide FWHT n=4096 rows=%lld\n", (long long) rows);
            launch_fwht_wide<4096, 256>(src_d, dst_d, rows, scale, stream);
            return true;
        case 8192:
            if (std::getenv("GGML_SYCL_DISABLE_FWHT_WIDE") != nullptr) return false;
            GGML_SYCL_DEBUG("[SYCL] dispatch wide FWHT n=8192 rows=%lld\n", (long long) rows);
            launch_fwht_wide<8192, 256>(src_d, dst_d, rows, scale, stream);
            return true;
        default:
            return false;
    }
}
