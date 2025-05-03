#include <emmintrin.h>
//---------------------------------------------------------------------------
#include "common/Utils.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
namespace bitpacking {
//---------------------------------------------------------------------------
namespace simd64 {
//---------------------------------------------------------------------------
/// @brief Reads 128 values from "in", writes "bit" 128-bit vectors to "out".
void simdpack(const u64 *in, __m128i *out, const u8 bit);
//---------------------------------------------------------------------------
/// @brief Reads 128 values from "in", writes "bit" 128-bit vectors to "out".
/// Note: Values are not masked before being packed.
void simdpackwithoutmask(const u64 *in, __m128i *out, const u8 bit);
//---------------------------------------------------------------------------
/// @brief Reads "bit" 128-bit vectors from "in", writes 128 values to "out".
void simdunpack(const __m128i *in, u64 *out, const u8 bit);
//---------------------------------------------------------------------------
/// @brief Reads "kLength" values from "in", and bitpacks them to "out".
template <u16 kLength>
__m128i *simdpack_shortlength(const u64 *in, __m128i *out, const u8 bit) {
  static_assert(kLength % 2 == 0);

  int k;
  int inwordpointer;
  __m128i P;
  u64 firstpass;
  if (bit == 0)
    return out; /* nothing to do */
  if (bit == 64) {
    memcpy(out, in, kLength * sizeof(u64));
    return (__m128i *)((u64 *)out + kLength);
  }
  inwordpointer = 0;
  P = _mm_setzero_si128();
  for (k = 0; k < kLength / 2; ++k) {
    __m128i value = _mm_loadu_si128(((const __m128i *)in + k));
    P = _mm_or_si128(P, _mm_slli_epi64(value, inwordpointer));
    firstpass = sizeof(u64) * 8 - inwordpointer;
    if (bit < firstpass) {
      inwordpointer += bit;
    } else {
      _mm_storeu_si128(out++, P);
      P = _mm_srli_epi64(value, firstpass);
      inwordpointer = bit - firstpass;
    }
  }
  if (inwordpointer != 0) {
    _mm_storeu_si128(out++, P);
  }
  return out;
}
//---------------------------------------------------------------------------
/// @brief Reads compressed data from "in" and writes "kLength" values to "out"
template <u16 kLength>
const __m128i *simdunpack_shortlength(const __m128i *in, u64 *out,
                                      const u8 bit) {
  static_assert(kLength % 2 == 0);

  int k;
  __m128i maskbits;
  int inwordpointer;
  __m128i P;
  if (kLength == 0)
    return in;
  if (bit == 0) {
    for (k = 0; k < kLength; ++k) {
      out[k] = 0;
    }
    return in;
  }
  if (bit == 64) {
    memcpy(out, in, kLength * sizeof(u64));
    return (const __m128i *)((u64 *)in + kLength);
  }
  maskbits = _mm_set1_epi64x((1U << bit) - 1);
  inwordpointer = 0;
  P = _mm_loadu_si128((__m128i *)in);
  ++in;

  for (k = 0; k + 1 < kLength / 2; ++k) {
    __m128i answer = _mm_srli_epi64(P, inwordpointer);
    const u64 firstpass = sizeof(u64) * 8 - inwordpointer;
    if (bit < firstpass) {
      inwordpointer += bit;
    } else {
      P = _mm_loadu_si128((__m128i *)in);
      ++in;
      answer = _mm_or_si128(_mm_slli_epi64(P, firstpass), answer);
      inwordpointer = bit - firstpass;
    }
    answer = _mm_and_si128(maskbits, answer);
    _mm_storeu_si128((__m128i *)out, answer);
    out += 2;
  }
  if (k < kLength / 2) {
    __m128i answer = _mm_srli_epi64(P, inwordpointer);
    const u64 firstpass = sizeof(u64) * 8 - inwordpointer;
    if (bit < firstpass) {
      inwordpointer += bit;
    } else if (bit == firstpass) {
      inwordpointer = 0;
    } else {
      P = _mm_loadu_si128((__m128i *)in);
      ++in;
      answer = _mm_or_si128(_mm_slli_epi64(P, firstpass), answer);
      inwordpointer = bit - firstpass;
    }
    answer = _mm_and_si128(maskbits, answer);
    _mm_storeu_si128((__m128i *)out, answer);
    out += 2;
  }
  return in;
}
//---------------------------------------------------------------------------
__m128i *simdpack_length(const u64 *in, u16 length, __m128i *out, const u8 bit);
//---------------------------------------------------------------------------
const __m128i *simdunpack_length(const __m128i *in, u16 length, u64 *out,
                                 const u8 bit);
//---------------------------------------------------------------------------
__m128i *simdpack_shortlength(const u64 *in, u16 length, __m128i *out,
                              const u8 bit);
//---------------------------------------------------------------------------
const __m128i *simdunpack_shortlength(const __m128i *in, u16 length, u64 *out,
                                      const u8 bit);
//---------------------------------------------------------------------------
} // namespace simd64
//---------------------------------------------------------------------------
} // namespace bitpacking
//---------------------------------------------------------------------------
} // namespace compression