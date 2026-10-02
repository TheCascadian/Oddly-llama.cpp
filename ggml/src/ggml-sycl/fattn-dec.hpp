#pragma once

#include "common.hpp"

bool ggml_sycl_flash_attn_ext_dec_supported(const ggml_tensor * dst);
void ggml_sycl_flash_attn_ext_dec(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
