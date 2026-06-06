#pragma once

#include <cuda/std/cstdint>
#include <cutlass/float_subbyte.h>
#include <deep_gemm/common/compile.cuh>
#include <deep_gemm/common/exception.cuh>

namespace deep_gemm::math {

/// Pointer operations
template <typename dtype_t = void>
CUTLASS_HOST_DEVICE dtype_t* advance_ptr(void* ptr, const uint64_t num_bytes) {
    return reinterpret_cast<dtype_t*>(static_cast<uint8_t*>(ptr) + num_bytes);
}

/// Math functions
template <typename T>
CUTLASS_HOST_DEVICE T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

template <typename T>
CUTLASS_HOST_DEVICE constexpr T constexpr_ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

template <typename T, bool kDoCeilAlignment = true>
CUTLASS_HOST_DEVICE T align(T a, T b) {
    return (kDoCeilAlignment ? ceil_div(a, b) : (a / b)) * b;
}

template <typename T>
CUTLASS_HOST_DEVICE constexpr T constexpr_align(T a, T b) {
    return constexpr_ceil_div(a, b) * b;
}

template <typename T>
CUTLASS_HOST_DEVICE constexpr T constexpr_gcd(T a, T b) {
    return b == 0 ? a : constexpr_gcd(b, a % b);
}

template <typename T>
CUTLASS_HOST_DEVICE constexpr T constexpr_min(T a, T b) {
    return a < b ? a : b;
}

template <typename dtype_t>
CUTLASS_HOST_DEVICE constexpr uint32_t get_smem_elem_bits() {
    if constexpr (cute::is_same_v<dtype_t, cutlass::detail::float_e2m1_unpacksmem_t> or
                  cute::is_same_v<dtype_t, cutlass::detail::float_e2m3_unpacksmem_t> or
                  cute::is_same_v<dtype_t, cutlass::detail::float_e3m2_unpacksmem_t> or
                  cute::is_same_v<dtype_t, cutlass::detail::type_erased_dynamic_float4_unpacksmem_t> or
                  cute::is_same_v<dtype_t, cutlass::detail::type_erased_dynamic_float6_unpacksmem_t>) {
        return 8;
    } else {
        return cute::sizeof_bits_v<dtype_t>;
    }
}

template <typename T>
CUTLASS_DEVICE void swap(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}

#ifdef DG_IN_CUDA_COMPILATION
CUTLASS_DEVICE float2 fma2(const float2& a, const float2& b, const float2& c) {
#if defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)
    return __ffma2_rn(a, b, c);
#else
    return make_float2(
        __fmaf_rn(a.x, b.x, c.x),
        __fmaf_rn(a.y, b.y, c.y)
    );
#endif
}

CUTLASS_HOST_DEVICE float fast_rcp(const float& x) {
    float ret;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(ret) : "f"(x));
    return ret;
}

/// Casting
template <typename old_t>
CUTLASS_DEVICE int cast_into_bf16_and_pack(old_t& x, old_t& y) {
    auto bf16x2 = __float22bfloat162_rn({*reinterpret_cast<float*>(&x), *reinterpret_cast<float*>(&y)});
    return *reinterpret_cast<int*>(&bf16x2);
}

CUTLASS_DEVICE float fast_pow2(const int& x) {
    uint32_t bits_x = (x + 127) << 23;
    return *reinterpret_cast<float*>(&bits_x);
}

CUTLASS_DEVICE int fast_log2_ceil(float x) {
    const auto bits = *reinterpret_cast<uint32_t*>(&x);
    const auto exp = bits >> 23;
    const auto man = bits & ((1 << 23) - 1);
    return exp - 127 + (man != 0);
}

template <bool kUseUE8M0 = true>
CUTLASS_DEVICE void get_e4m3_sf_and_sf_inv(const float2& amax, float2& sf, float2& sf_inv) {
    DG_STATIC_ASSERT(kUseUE8M0, "Must use UE8M0");
    const float2 finfo_factor = {1.0 / 448.0, 1.0 / 448.0};
    const auto scaled = __fmul2_rn(amax, finfo_factor);
    const auto exp_x = fast_log2_ceil(scaled.x);
    const auto exp_y = fast_log2_ceil(scaled.y);
    sf.x = fast_pow2(exp_x), sf_inv.x = fast_pow2(-exp_x);
    sf.y = fast_pow2(exp_y), sf_inv.y = fast_pow2(-exp_y);
}

template <bool kUseUE8M0 = true>
CUTLASS_DEVICE void get_e2m1_sf_and_sf_inv(const float2& amax, float2& sf, float2& sf_inv) {
    DG_STATIC_ASSERT(kUseUE8M0, "Must use UE8M0");
    const float2 finfo_factor = {1.0 / 6.0, 1.0 / 6.0};
    const auto scaled = __fmul2_rn(amax, finfo_factor);
    const auto exp_x = fast_log2_ceil(scaled.x);
    const auto exp_y = fast_log2_ceil(scaled.y);
    sf.x = fast_pow2(exp_x), sf_inv.x = fast_pow2(-exp_x);
    sf.y = fast_pow2(exp_y), sf_inv.y = fast_pow2(-exp_y);
}

CUTLASS_DEVICE uint32_t cast_into_e2m1_code(const float& value) {
    const bool is_negative = value < 0;
    float abs_value = is_negative ? -value : value;
    abs_value = abs_value > 6.0f ? 6.0f : abs_value;
    uint32_t code = 0;
    code = abs_value > 0.25f ? 1 : code;
    code = abs_value > 0.75f ? 2 : code;
    code = abs_value > 1.25f ? 3 : code;
    code = abs_value > 1.75f ? 4 : code;
    code = abs_value > 2.50f ? 5 : code;
    code = abs_value > 3.50f ? 6 : code;
    code = abs_value > 5.00f ? 7 : code;
    return code | (is_negative and code != 0 ? 0x8u : 0u);
}

CUTLASS_DEVICE uint32_t cvt_pack_f32_to_e2m1x2(const float& a, const float& b) {
    return cast_into_e2m1_code(a) | (cast_into_e2m1_code(b) << 4);
}

/// Reduction
CUTLASS_DEVICE uint32_t warp_inclusive_sum(uint32_t value, const uint32_t& lane_idx) {
    #pragma unroll
    for (uint32_t offset = 1; offset < 32; offset <<= 1) {
        const uint32_t synced = __shfl_up_sync(0xffffffff, value, offset);
        if (lane_idx >= offset)
            value += synced;
    }
    return value;
}

// Operation functors
template <typename T> struct ReduceSum { CUTLASS_DEVICE T operator()(T a, T b) const { return a + b; } };
template <typename T> struct ReduceMax { CUTLASS_DEVICE T operator()(T a, T b) const { return a > b ? a : b; } };
template <typename T> struct ReduceMin { CUTLASS_DEVICE T operator()(T a, T b) const { return a < b ? a : b; } };
template <typename T> struct ReduceAnd { CUTLASS_DEVICE T operator()(T a, T b) const { return a & b; } };
template <typename T> struct ReduceOr  { CUTLASS_DEVICE T operator()(T a, T b) const { return a | b; } };

// Unified reduction function
template <uint32_t kNumLanesPerGroup, bool kIntergroupReduce, typename T, typename Op>
CUTLASS_DEVICE T warp_reduce(T value, Op op) {
    DG_STATIC_ASSERT(kNumLanesPerGroup == 32 or kNumLanesPerGroup == 16 or kNumLanesPerGroup == 8 or
                     kNumLanesPerGroup ==  4 or kNumLanesPerGroup == 2  or kNumLanesPerGroup == 1,
                     "Invalid number of lanes");
    constexpr uint32_t mask = 0xffffffff;
    if constexpr (kIntergroupReduce) {
        if constexpr (kNumLanesPerGroup <=  1) value = op(value, __shfl_xor_sync(mask, value,  1));
        if constexpr (kNumLanesPerGroup <=  2) value = op(value, __shfl_xor_sync(mask, value,  2));
        if constexpr (kNumLanesPerGroup <=  4) value = op(value, __shfl_xor_sync(mask, value,  4));
        if constexpr (kNumLanesPerGroup <=  8) value = op(value, __shfl_xor_sync(mask, value,  8));
        if constexpr (kNumLanesPerGroup <= 16) value = op(value, __shfl_xor_sync(mask, value, 16));
    } else {
        if constexpr (kNumLanesPerGroup >= 32) value = op(value, __shfl_xor_sync(mask, value, 16));
        if constexpr (kNumLanesPerGroup >= 16) value = op(value, __shfl_xor_sync(mask, value,  8));
        if constexpr (kNumLanesPerGroup >=  8) value = op(value, __shfl_xor_sync(mask, value,  4));
        if constexpr (kNumLanesPerGroup >=  4) value = op(value, __shfl_xor_sync(mask, value,  2));
        if constexpr (kNumLanesPerGroup >=  2) value = op(value, __shfl_xor_sync(mask, value,  1));
    }
    return value;
}

// Convenience aliases
template <uint32_t kNumLanesPerGroup = 32, bool kIntergroupReduce = false, typename T>
CUTLASS_DEVICE T warp_reduce_sum(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceSum<T>{});
}
#endif

} // namespace deep_gemm
