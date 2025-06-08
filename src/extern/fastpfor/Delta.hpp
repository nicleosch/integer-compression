#pragma once
//---------------------------------------------------------------------------
#include <emmintrin.h>
//---------------------------------------------------------------------------
#include "common/Types.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
namespace pfor {
//---------------------------------------------------------------------------
namespace delta {
//---------------------------------------------------------------------------
/// In-place compression.
///
/// This function is derived from "deltaSIMD" from the deltautil.h header in
/// https://github.com/fast-pack/FastPFOR. To be precise, symbol names are
/// changed and some edge cases are not considered as unnecessary.
template <typename T, const u16 kBlockSize> void compress(T *array) {
  static_assert(kBlockSize % 4 == 0);

  const auto ipr = sizeof(__m128i) / sizeof(T);
  const u16 iterations = kBlockSize / ipr;

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
/// Out-of-place compression.
///
/// This function is derived from "deltaSIMD" from the deltautil.h header in
/// https://github.com/fast-pack/FastPFOR. To be precise, symbol names are
/// changed and some edge cases are not considered as unnecessary.
template <typename T, const u16 kBlockSize>
void compress(const T *src, T *dest) {
  static_assert(kBlockSize % 4 == 0);

  const auto ipr = sizeof(__m128i) / sizeof(T);
  const u16 iterations = kBlockSize / ipr;

  auto start_ptr = reinterpret_cast<const __m128i *>(src);
  auto cur_ptr = reinterpret_cast<const __m128i *>(src) + iterations - 1;
  auto write_ptr = reinterpret_cast<__m128i *>(dest) + iterations - 1;

  __m128i minuends = _mm_loadu_si128(cur_ptr);
  for (; cur_ptr > start_ptr; --cur_ptr, --write_ptr) {
    __m128i subtrahends = _mm_loadu_si128(cur_ptr - 1);
    _mm_storeu_si128(write_ptr, _mm_sub_epi32(minuends, subtrahends));
    minuends = subtrahends;
  }
  _mm_storeu_si128(write_ptr, *cur_ptr);
}
//---------------------------------------------------------------------------
/// This function is derived from "inverseDeltaSIMD" from the deltautil.h header
/// in https://github.com/fast-pack/FastPFOR. To be precise, symbol names are
/// changed and some edge cases are not considered as unnecessary.
template <typename T, const u16 kBlockSize> void decompress(T *array) {
  static_assert(kBlockSize % 4 == 0);

  const auto ipr = sizeof(__m128i) / sizeof(T);
  const u16 iterations = kBlockSize / ipr;

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
} // namespace pfor
//---------------------------------------------------------------------------
} // namespace compression