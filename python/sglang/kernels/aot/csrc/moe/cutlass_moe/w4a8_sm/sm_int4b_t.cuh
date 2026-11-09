#pragma once

#include "cute/numeric/numeric_types.hpp"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/integer_subbyte.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_size.h"

namespace cutlass {

// Sign-magnitude INT4 type.
// Same physical storage as int4b_t but semantically different:
//   bit layout: [sign][m2][m1][m0]
//   value = (-1)^sign * (m2*4 + m1*2 + m0)
//   range: [-7, +7]  (cannot represent -8)
//
// Converted to FP8 E4M3 by mapping to subnormal region:
//   FP8 bits: [sign]0000[m2][m1][m0]
//   FP8 value = (-1)^sign * m * 2^(-9)
// Weight scales must be pre-multiplied by 512 to compensate.

struct sm_int4b_t {
  using Storage = uint8_t;
  Storage storage;

  sm_int4b_t() = default;

  CUTLASS_HOST_DEVICE explicit sm_int4b_t(int value) : storage(static_cast<uint8_t>(value) & 0x0F) {}

  CUTLASS_HOST_DEVICE explicit sm_int4b_t(unsigned value) : storage(static_cast<uint8_t>(value) & 0x0F) {}

  CUTLASS_HOST_DEVICE
  operator int() const {
    int mag = storage & 0x07;
    return (storage & 0x08) ? -mag : mag;
  }

  CUTLASS_HOST_DEVICE
  bool operator==(sm_int4b_t const& rhs) const {
    return storage == rhs.storage;
  }

  CUTLASS_HOST_DEVICE
  bool operator!=(sm_int4b_t const& rhs) const {
    return storage != rhs.storage;
  }
};

template <>
struct sizeof_bits<sm_int4b_t> {
  static constexpr int value = 4;
};

// ─── NumericArrayConverter: sm_int4b_t → float_e4m3_t ───
//
// Uses bitwise ops to map sign-magnitude INT4 to FP8 E4M3 subnormal.
// The packed-K 8-element path writes both registers without PRMT.

template <FloatRoundStyle Round, int N>
struct NumericArrayConverter<float_e4m3_t, sm_int4b_t, N, Round> {
  using result_type = Array<float_e4m3_t, N>;
  using source_type = Array<sm_int4b_t, N>;

  static FloatRoundStyle const round_style = Round;

 private:
  using result_type_packed_8 = Array<float_e4m3_t, 8>;
  using result_type_packed_4 = Array<float_e4m3_t, 4>;
  using source_type_packed_8 = Array<sm_int4b_t, 8>;
  using source_type_packed_4 = Array<sm_int4b_t, 4>;

  using ScalarConverter = NumericConverter<float_e4m3_t, sm_int4b_t, Round>;

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_4 const& source) {
    return static_cast<uint32_t>(reinterpret_cast<const uint16_t&>(source));
  }

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_8 const& source) {
    return reinterpret_cast<const uint32_t&>(source);
  }

  // Convert 8 sign-magnitude INT4 values (packed in one uint32_t) to 8 FP8 E4M3 values.
  //
  // Input register layout (8 nibbles in 4 bytes, CUTLASS interleaved packing):
  //   byte3[7:4]=elem7 byte3[3:0]=elem6  byte2[7:4]=elem5 byte2[3:0]=elem4
  //   byte1[7:4]=elem3 byte1[3:0]=elem2  byte0[7:4]=elem1 byte0[3:0]=elem0
  //
  // Each nibble: [sign][m2][m1][m0]
  // Target FP8 E4M3: [sign]0000[m2][m1][m0]  (subnormal, exponent=0000)
  //
  // After conversion we need two uint32_t outputs with 4 FP8 bytes each.
  // Convert 8 sign-magnitude INT4 values to 8 FP8 E4M3 subnormals.
  //
  // Load-time K permutation removes the 8-element interleave instructions.
  template <typename PackedResultType, typename PackedSrcType>
  CUTLASS_DEVICE static PackedResultType packed_convert(PackedSrcType const& source) {
    static_assert(
        (platform::is_same<PackedSrcType, source_type_packed_4>::value &&
         platform::is_same<PackedResultType, result_type_packed_4>::value) ||
            (platform::is_same<PackedSrcType, source_type_packed_8>::value &&
             platform::is_same<PackedResultType, result_type_packed_8>::value),
        "Invalid PackedSrcType/PackedResultType");

    cutlass::AlignedArray<uint32_t, PackedResultType::kElements / 4, sizeof(PackedResultType)> r;

    uint32_t src_reg = to_reg(source);

    // High nibbles → FP8: sign stays at bit7, magnitude shifted to bits[2:0]
    uint32_t hi_sign = src_reg & 0x80808080;
    uint32_t hi_fp8 = (src_reg >> 4) & 0x07070707;
    hi_fp8 |= hi_sign;

    // Low nibbles → FP8: sign from bit3 moved to bit7, magnitude at bits[2:0]
    uint32_t lo_sign = (src_reg & 0x08080808) << 4;
    uint32_t lo_fp8 = src_reg & 0x07070707;
    lo_fp8 |= lo_sign;

    if constexpr (PackedSrcType::kElements == 4) {
      asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(r[0]) : "r"(lo_fp8), "r"(hi_fp8), "n"(0x5140));
    } else {
      // 8-element path: weights pre-reordered [0,4,1,5,2,6,3,7] per K-group-of-8
      // so lo_fp8 already contains [e0,e1,e2,e3] and hi_fp8 contains [e4,e5,e6,e7]
      r[0] = lo_fp8;
      r[1] = hi_fp8;
    }

    return reinterpret_cast<PackedResultType&>(r);
  }

  friend class detail::VectorizedConverter;

 public:
  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const& source) {
#if defined(__CUDA_ARCH__)
    result_type result;
    using ConverterType = NumericArrayConverter<typename result_type::Element, typename source_type::Element, N, Round>;
    detail::VectorizedConverter::
        convert<ConverterType, result_type_packed_8, source_type_packed_8, result_type_packed_4, source_type_packed_4>(
            result, source);
    return result;
#else
    result_type result;
    for (int i = 0; i < N; ++i) {
      int val = static_cast<int>(source[i]);
      int sign = (val < 0) ? 1 : 0;
      int mag = (val < 0) ? -val : val;
      uint8_t fp8_bits = static_cast<uint8_t>((sign << 7) | (mag & 0x7));
      result[i] = reinterpret_cast<float_e4m3_t const&>(fp8_bits);
    }
    return result;
#endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const& s) const {
    return convert(s);
  }
};

// Scalar converter (needed by VectorizedConverter fallback)
template <FloatRoundStyle Round>
struct NumericConverter<float_e4m3_t, sm_int4b_t, Round> {
  CUTLASS_HOST_DEVICE
  float_e4m3_t operator()(sm_int4b_t s) const {
    int val = static_cast<int>(s);
    int sign = (val < 0) ? 1 : 0;
    int mag = (val < 0) ? -val : val;
    uint8_t fp8_bits = static_cast<uint8_t>((sign << 7) | (mag & 0x7));
    return reinterpret_cast<float_e4m3_t const&>(fp8_bits);
  }
};

}  // namespace cutlass
