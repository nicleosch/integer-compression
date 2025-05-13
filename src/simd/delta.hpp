#pragma once
//---------------------------------------------------------------------------
#include <emmintrin.h>
//---------------------------------------------------------------------------
#include "common/Types.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
namespace simd {
//---------------------------------------------------------------------------
namespace delta {
//---------------------------------------------------------------------------
// Inspired by Daniel Lemire (https://github.com/fast-pack/FastPFOR)
template <typename T, u16 kLength> void compress(T *array) {
  static_assert(kLength % 4 == 0);

  const auto ipr = sizeof(__m128i) / sizeof(T);
  const u16 iterations = kLength / ipr;

  auto start_ptr = reinterpret_cast<const __m128i *>(array);
  auto cur_ptr = reinterpret_cast<__m128i *>(array) + iterations - 1;

  __m128i minuends = _mm_loadu_si128(cur_ptr);
  for (; cur_ptr > start_ptr; --cur_ptr) {
    __m128i subtrahends = _mm_loadu_si128(cur_ptr - 1);
    _mm_storeu_si128(cur_ptr, _mm_sub_epi32(minuends, subtrahends));
    minuends = subtrahends;
  }
}
//---------------------------------------------------------------------------
// Inspired by Daniel Lemire (https://github.com/fast-pack/FastPFOR)
template <typename T, u16 kLength> void decompress(T *array) {
  static_assert(kLength % 4 == 0);

  const auto ipr = sizeof(__m128i) / sizeof(T);
  const u16 iterations = kLength / ipr;

  auto cur_ptr = reinterpret_cast<__m128i *>(array);
  auto end_ptr = reinterpret_cast<__m128i *>(array) + iterations;

  __m128i base = _mm_loadu_si128(cur_ptr++);
  for (; cur_ptr < end_ptr; ++cur_ptr) {
    __m128i summands = _mm_loadu_si128(cur_ptr);
    base = _mm_add_epi32(base, summands);
    _mm_storeu_si128(cur_ptr, base);
  }
}
//---------------------------------------------------------------------------
} // namespace delta
//---------------------------------------------------------------------------
} // namespace simd
//---------------------------------------------------------------------------
} // namespace compression