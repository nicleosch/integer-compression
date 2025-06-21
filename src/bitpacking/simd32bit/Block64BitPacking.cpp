#include <cstring>
//---------------------------------------------------------------------------
#include "Block64BitPacking.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
namespace bitpacking {
//---------------------------------------------------------------------------
namespace simd32 {
//---------------------------------------------------------------------------
namespace block64 {
//---------------------------------------------------------------------------
typedef void (*simdpackblockfnc)(const u32 *pin, __m128i *compressed);
typedef void (*simdunpackblockfnc)(const __m128i *compressed, u32 *out);

static void simdpackblock0(const u32 *pin, __m128i *compressed) {
  (void)compressed;
  (void)pin; /* we consumed 64 32-bit integers */
}

/* we are going to pack 64 1-bit values, touching 1 128-bit words, using 16
 * bytes */
static void simdpackblock1(const u32 *pin, __m128i *compressed) {
  const __m128i *in = (const __m128i *)pin;
  /* we are going to touch  1 128-bit word */
  __m128i w0;
  w0 = _mm_loadu_si128(in + 0);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 1), 1));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 2), 2));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 3), 3));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 4), 4));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 5), 5));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 6), 6));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 7), 7));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 8), 8));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 9), 9));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 10), 10));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 11), 11));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 12), 12));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 13), 13));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 14), 14));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 15), 15));
  _mm_storeu_si128(compressed + 0, w0);
}

/* we are going to pack 64 2-bit values, touching 1 128-bit words, using 16
 * bytes */
static void simdpackblock2(const u32 *pin, __m128i *compressed) {
  const __m128i *in = (const __m128i *)pin;
  /* we are going to touch  1 128-bit word */
  __m128i w0;
  w0 = _mm_loadu_si128(in + 0);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 1), 2));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 2), 4));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 3), 6));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 4), 8));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 5), 10));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 6), 12));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 7), 14));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 8), 16));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 9), 18));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 10), 20));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 11), 22));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 12), 24));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 13), 26));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 14), 28));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 15), 30));
  _mm_storeu_si128(compressed + 0, w0);
}

/* we are going to pack 64 3-bit values, touching 2 128-bit words, using 32
 * bytes */
static void simdpackblock3(const u32 *pin, __m128i *compressed) {
  const __m128i *in = (const __m128i *)pin;
  /* we are going to touch  2 128-bit words */
  __m128i w0, w1;
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_loadu_si128(in + 0);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 1), 3));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 2), 6));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 3), 9));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 4), 12));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 5), 15));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 6), 18));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 7), 21));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 8), 24));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 9), 27));
  tmp = _mm_loadu_si128(in + 10);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 30));
  w1 = _mm_srli_epi32(tmp, 2);
  _mm_storeu_si128(compressed + 0, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 11), 1));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 12), 4));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 13), 7));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 14), 10));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 15), 13));
  _mm_storeu_si128(compressed + 1, w1);
}

/* we are going to pack 64 4-bit values, touching 2 128-bit words, using 32
 * bytes */
static void simdpackblock4(const u32 *pin, __m128i *compressed) {
  const __m128i *in = (const __m128i *)pin;
  /* we are going to touch  2 128-bit words */
  __m128i w0, w1;
  w0 = _mm_loadu_si128(in + 0);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 1), 4));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 2), 8));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 3), 12));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 4), 16));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 5), 20));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 6), 24));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 7), 28));
  _mm_storeu_si128(compressed + 0, w0);
  w1 = _mm_loadu_si128(in + 8);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 9), 4));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 10), 8));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 11), 12));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 12), 16));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 13), 20));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 14), 24));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 15), 28));
  _mm_storeu_si128(compressed + 1, w1);
}

/* we are going to pack 64 5-bit values, touching 3 128-bit words, using 48
 * bytes */
static void simdpackblock5(const u32 *pin, __m128i *compressed) {
  const __m128i *in = (const __m128i *)pin;
  /* we are going to touch  3 128-bit words */
  __m128i w0, w1;
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_loadu_si128(in + 0);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 1), 5));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 2), 10));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 3), 15));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 4), 20));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 5), 25));
  tmp = _mm_loadu_si128(in + 6);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 30));
  w1 = _mm_srli_epi32(tmp, 2);
  _mm_storeu_si128(compressed + 0, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 7), 3));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 8), 8));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 9), 13));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 10), 18));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 11), 23));
  tmp = _mm_loadu_si128(in + 12);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 28));
  w0 = _mm_srli_epi32(tmp, 4);
  _mm_storeu_si128(compressed + 1, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 13), 1));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 14), 6));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 15), 11));
  _mm_storeu_si128(compressed + 2, w0);
}

/* we are going to pack 64 6-bit values, touching 3 128-bit words, using 48
 * bytes */
static void simdpackblock6(const u32 *pin, __m128i *compressed) {
  const __m128i *in = (const __m128i *)pin;
  /* we are going to touch  3 128-bit words */
  __m128i w0, w1;
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_loadu_si128(in + 0);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 1), 6));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 2), 12));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 3), 18));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 4), 24));
  tmp = _mm_loadu_si128(in + 5);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 30));
  w1 = _mm_srli_epi32(tmp, 2);
  _mm_storeu_si128(compressed + 0, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 6), 4));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 7), 10));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 8), 16));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 9), 22));
  tmp = _mm_loadu_si128(in + 10);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 28));
  w0 = _mm_srli_epi32(tmp, 4);
  _mm_storeu_si128(compressed + 1, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 11), 2));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 12), 8));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 13), 14));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 14), 20));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 15), 26));
  _mm_storeu_si128(compressed + 2, w0);
}

/* we are going to pack 64 7-bit values, touching 4 128-bit words, using 64
 * bytes */
static void simdpackblock7(const u32 *pin, __m128i *compressed) {
  const __m128i *in = (const __m128i *)pin;
  /* we are going to touch  4 128-bit words */
  __m128i w0, w1;
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_loadu_si128(in + 0);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 1), 7));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 2), 14));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 3), 21));
  tmp = _mm_loadu_si128(in + 4);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 28));
  w1 = _mm_srli_epi32(tmp, 4);
  _mm_storeu_si128(compressed + 0, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 5), 3));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 6), 10));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 7), 17));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 8), 24));
  tmp = _mm_loadu_si128(in + 9);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 31));
  w0 = _mm_srli_epi32(tmp, 1);
  _mm_storeu_si128(compressed + 1, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 10), 6));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 11), 13));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 12), 20));
  tmp = _mm_loadu_si128(in + 13);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 27));
  w1 = _mm_srli_epi32(tmp, 5);
  _mm_storeu_si128(compressed + 2, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 14), 2));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 15), 9));
  _mm_storeu_si128(compressed + 3, w1);
}

/* we are going to pack 64 8-bit values, touching 4 128-bit words, using 64
 * bytes */
static void simdpackblock8(const u32 *pin, __m128i *compressed) {
  const __m128i *in = (const __m128i *)pin;
  /* we are going to touch  4 128-bit words */
  __m128i w0, w1;
  w0 = _mm_loadu_si128(in + 0);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 1), 8));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 2), 16));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 3), 24));
  _mm_storeu_si128(compressed + 0, w0);
  w1 = _mm_loadu_si128(in + 4);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 5), 8));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 6), 16));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 7), 24));
  _mm_storeu_si128(compressed + 1, w1);
  w0 = _mm_loadu_si128(in + 8);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 9), 8));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 10), 16));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 11), 24));
  _mm_storeu_si128(compressed + 2, w0);
  w1 = _mm_loadu_si128(in + 12);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 13), 8));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 14), 16));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 15), 24));
  _mm_storeu_si128(compressed + 3, w1);
}

/* we are going to pack 64 9-bit values, touching 5 128-bit words, using 80
 * bytes */
static void simdpackblock9(const u32 *pin, __m128i *compressed) {
  const __m128i *in = (const __m128i *)pin;
  /* we are going to touch  5 128-bit words */
  __m128i w0, w1;
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_loadu_si128(in + 0);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 1), 9));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 2), 18));
  tmp = _mm_loadu_si128(in + 3);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 27));
  w1 = _mm_srli_epi32(tmp, 5);
  _mm_storeu_si128(compressed + 0, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 4), 4));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 5), 13));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 6), 22));
  tmp = _mm_loadu_si128(in + 7);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 31));
  w0 = _mm_srli_epi32(tmp, 1);
  _mm_storeu_si128(compressed + 1, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 8), 8));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 9), 17));
  tmp = _mm_loadu_si128(in + 10);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 26));
  w1 = _mm_srli_epi32(tmp, 6);
  _mm_storeu_si128(compressed + 2, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 11), 3));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 12), 12));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 13), 21));
  tmp = _mm_loadu_si128(in + 14);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 30));
  w0 = _mm_srli_epi32(tmp, 2);
  _mm_storeu_si128(compressed + 3, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 15), 7));
  _mm_storeu_si128(compressed + 4, w0);
}

/* we are going to pack 64 10-bit values, touching 5 128-bit words, using 80
 * bytes */
static void simdpackblock10(const u32 *pin, __m128i *compressed) {
  const __m128i *in = (const __m128i *)pin;
  /* we are going to touch  5 128-bit words */
  __m128i w0, w1;
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_loadu_si128(in + 0);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 1), 10));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 2), 20));
  tmp = _mm_loadu_si128(in + 3);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 30));
  w1 = _mm_srli_epi32(tmp, 2);
  _mm_storeu_si128(compressed + 0, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 4), 8));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 5), 18));
  tmp = _mm_loadu_si128(in + 6);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 28));
  w0 = _mm_srli_epi32(tmp, 4);
  _mm_storeu_si128(compressed + 1, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 7), 6));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 8), 16));
  tmp = _mm_loadu_si128(in + 9);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 26));
  w1 = _mm_srli_epi32(tmp, 6);
  _mm_storeu_si128(compressed + 2, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 10), 4));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 11), 14));
  tmp = _mm_loadu_si128(in + 12);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 24));
  w0 = _mm_srli_epi32(tmp, 8);
  _mm_storeu_si128(compressed + 3, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 13), 2));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 14), 12));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 15), 22));
  _mm_storeu_si128(compressed + 4, w0);
}

/* we are going to pack 64 11-bit values, touching 6 128-bit words, using 96
 * bytes */
static void simdpackblock11(const u32 *pin, __m128i *compressed) {
  const __m128i *in = (const __m128i *)pin;
  /* we are going to touch  6 128-bit words */
  __m128i w0, w1;
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_loadu_si128(in + 0);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 1), 11));
  tmp = _mm_loadu_si128(in + 2);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 22));
  w1 = _mm_srli_epi32(tmp, 10);
  _mm_storeu_si128(compressed + 0, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 3), 1));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 4), 12));
  tmp = _mm_loadu_si128(in + 5);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 23));
  w0 = _mm_srli_epi32(tmp, 9);
  _mm_storeu_si128(compressed + 1, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 6), 2));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 7), 13));
  tmp = _mm_loadu_si128(in + 8);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 24));
  w1 = _mm_srli_epi32(tmp, 8);
  _mm_storeu_si128(compressed + 2, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 9), 3));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 10), 14));
  tmp = _mm_loadu_si128(in + 11);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 25));
  w0 = _mm_srli_epi32(tmp, 7);
  _mm_storeu_si128(compressed + 3, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 12), 4));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 13), 15));
  tmp = _mm_loadu_si128(in + 14);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 26));
  w1 = _mm_srli_epi32(tmp, 6);
  _mm_storeu_si128(compressed + 4, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 15), 5));
  _mm_storeu_si128(compressed + 5, w1);
}

/* we are going to pack 64 12-bit values, touching 6 128-bit words, using 96
 * bytes */
static void simdpackblock12(const u32 *pin, __m128i *compressed) {
  const __m128i *in = (const __m128i *)pin;
  /* we are going to touch  6 128-bit words */
  __m128i w0, w1;
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_loadu_si128(in + 0);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 1), 12));
  tmp = _mm_loadu_si128(in + 2);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 24));
  w1 = _mm_srli_epi32(tmp, 8);
  _mm_storeu_si128(compressed + 0, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 3), 4));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 4), 16));
  tmp = _mm_loadu_si128(in + 5);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 28));
  w0 = _mm_srli_epi32(tmp, 4);
  _mm_storeu_si128(compressed + 1, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 6), 8));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 7), 20));
  _mm_storeu_si128(compressed + 2, w0);
  w1 = _mm_loadu_si128(in + 8);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 9), 12));
  tmp = _mm_loadu_si128(in + 10);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 24));
  w0 = _mm_srli_epi32(tmp, 8);
  _mm_storeu_si128(compressed + 3, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 11), 4));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 12), 16));
  tmp = _mm_loadu_si128(in + 13);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 28));
  w1 = _mm_srli_epi32(tmp, 4);
  _mm_storeu_si128(compressed + 4, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 14), 8));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 15), 20));
  _mm_storeu_si128(compressed + 5, w1);
}

/* we are going to pack 64 13-bit values, touching 7 128-bit words, using 112
 * bytes */
static void simdpackblock13(const u32 *pin, __m128i *compressed) {
  const __m128i *in = (const __m128i *)pin;
  /* we are going to touch  7 128-bit words */
  __m128i w0, w1;
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_loadu_si128(in + 0);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 1), 13));
  tmp = _mm_loadu_si128(in + 2);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 26));
  w1 = _mm_srli_epi32(tmp, 6);
  _mm_storeu_si128(compressed + 0, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 3), 7));
  tmp = _mm_loadu_si128(in + 4);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 20));
  w0 = _mm_srli_epi32(tmp, 12);
  _mm_storeu_si128(compressed + 1, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 5), 1));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 6), 14));
  tmp = _mm_loadu_si128(in + 7);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 27));
  w1 = _mm_srli_epi32(tmp, 5);
  _mm_storeu_si128(compressed + 2, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 8), 8));
  tmp = _mm_loadu_si128(in + 9);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 21));
  w0 = _mm_srli_epi32(tmp, 11);
  _mm_storeu_si128(compressed + 3, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 10), 2));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 11), 15));
  tmp = _mm_loadu_si128(in + 12);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 28));
  w1 = _mm_srli_epi32(tmp, 4);
  _mm_storeu_si128(compressed + 4, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 13), 9));
  tmp = _mm_loadu_si128(in + 14);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 22));
  w0 = _mm_srli_epi32(tmp, 10);
  _mm_storeu_si128(compressed + 5, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 15), 3));
  _mm_storeu_si128(compressed + 6, w0);
}

/* we are going to pack 64 14-bit values, touching 7 128-bit words, using 112
 * bytes */
static void simdpackblock14(const u32 *pin, __m128i *compressed) {
  const __m128i *in = (const __m128i *)pin;
  /* we are going to touch  7 128-bit words */
  __m128i w0, w1;
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_loadu_si128(in + 0);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 1), 14));
  tmp = _mm_loadu_si128(in + 2);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 28));
  w1 = _mm_srli_epi32(tmp, 4);
  _mm_storeu_si128(compressed + 0, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 3), 10));
  tmp = _mm_loadu_si128(in + 4);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 24));
  w0 = _mm_srli_epi32(tmp, 8);
  _mm_storeu_si128(compressed + 1, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 5), 6));
  tmp = _mm_loadu_si128(in + 6);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 20));
  w1 = _mm_srli_epi32(tmp, 12);
  _mm_storeu_si128(compressed + 2, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 7), 2));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 8), 16));
  tmp = _mm_loadu_si128(in + 9);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 30));
  w0 = _mm_srli_epi32(tmp, 2);
  _mm_storeu_si128(compressed + 3, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 10), 12));
  tmp = _mm_loadu_si128(in + 11);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 26));
  w1 = _mm_srli_epi32(tmp, 6);
  _mm_storeu_si128(compressed + 4, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 12), 8));
  tmp = _mm_loadu_si128(in + 13);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 22));
  w0 = _mm_srli_epi32(tmp, 10);
  _mm_storeu_si128(compressed + 5, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 14), 4));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 15), 18));
  _mm_storeu_si128(compressed + 6, w0);
}

/* we are going to pack 64 15-bit values, touching 8 128-bit words, using 128
 * bytes */
static void simdpackblock15(const u32 *pin, __m128i *compressed) {
  const __m128i *in = (const __m128i *)pin;
  /* we are going to touch  8 128-bit words */
  __m128i w0, w1;
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_loadu_si128(in + 0);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 1), 15));
  tmp = _mm_loadu_si128(in + 2);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 30));
  w1 = _mm_srli_epi32(tmp, 2);
  _mm_storeu_si128(compressed + 0, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 3), 13));
  tmp = _mm_loadu_si128(in + 4);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 28));
  w0 = _mm_srli_epi32(tmp, 4);
  _mm_storeu_si128(compressed + 1, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 5), 11));
  tmp = _mm_loadu_si128(in + 6);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 26));
  w1 = _mm_srli_epi32(tmp, 6);
  _mm_storeu_si128(compressed + 2, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 7), 9));
  tmp = _mm_loadu_si128(in + 8);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 24));
  w0 = _mm_srli_epi32(tmp, 8);
  _mm_storeu_si128(compressed + 3, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 9), 7));
  tmp = _mm_loadu_si128(in + 10);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 22));
  w1 = _mm_srli_epi32(tmp, 10);
  _mm_storeu_si128(compressed + 4, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 11), 5));
  tmp = _mm_loadu_si128(in + 12);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 20));
  w0 = _mm_srli_epi32(tmp, 12);
  _mm_storeu_si128(compressed + 5, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 13), 3));
  tmp = _mm_loadu_si128(in + 14);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 18));
  w1 = _mm_srli_epi32(tmp, 14);
  _mm_storeu_si128(compressed + 6, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 15), 1));
  _mm_storeu_si128(compressed + 7, w1);
}

/* we are going to pack 64 16-bit values, touching 8 128-bit words, using 128
 * bytes */
static void simdpackblock16(const u32 *pin, __m128i *compressed) {
  const __m128i *in = (const __m128i *)pin;
  /* we are going to touch  8 128-bit words */
  __m128i w0, w1;
  w0 = _mm_loadu_si128(in + 0);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 1), 16));
  _mm_storeu_si128(compressed + 0, w0);
  w1 = _mm_loadu_si128(in + 2);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 3), 16));
  _mm_storeu_si128(compressed + 1, w1);
  w0 = _mm_loadu_si128(in + 4);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 5), 16));
  _mm_storeu_si128(compressed + 2, w0);
  w1 = _mm_loadu_si128(in + 6);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 7), 16));
  _mm_storeu_si128(compressed + 3, w1);
  w0 = _mm_loadu_si128(in + 8);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 9), 16));
  _mm_storeu_si128(compressed + 4, w0);
  w1 = _mm_loadu_si128(in + 10);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 11), 16));
  _mm_storeu_si128(compressed + 5, w1);
  w0 = _mm_loadu_si128(in + 12);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 13), 16));
  _mm_storeu_si128(compressed + 6, w0);
  w1 = _mm_loadu_si128(in + 14);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 15), 16));
  _mm_storeu_si128(compressed + 7, w1);
}

/* we are going to pack 64 17-bit values, touching 9 128-bit words, using 144
 * bytes */
static void simdpackblock17(const u32 *pin, __m128i *compressed) {
  const __m128i *in = (const __m128i *)pin;
  /* we are going to touch  9 128-bit words */
  __m128i w0, w1;
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_loadu_si128(in + 0);
  tmp = _mm_loadu_si128(in + 1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 17));
  w1 = _mm_srli_epi32(tmp, 15);
  _mm_storeu_si128(compressed + 0, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 2), 2));
  tmp = _mm_loadu_si128(in + 3);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 19));
  w0 = _mm_srli_epi32(tmp, 13);
  _mm_storeu_si128(compressed + 1, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 4), 4));
  tmp = _mm_loadu_si128(in + 5);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 21));
  w1 = _mm_srli_epi32(tmp, 11);
  _mm_storeu_si128(compressed + 2, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 6), 6));
  tmp = _mm_loadu_si128(in + 7);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 23));
  w0 = _mm_srli_epi32(tmp, 9);
  _mm_storeu_si128(compressed + 3, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 8), 8));
  tmp = _mm_loadu_si128(in + 9);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 25));
  w1 = _mm_srli_epi32(tmp, 7);
  _mm_storeu_si128(compressed + 4, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 10), 10));
  tmp = _mm_loadu_si128(in + 11);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 27));
  w0 = _mm_srli_epi32(tmp, 5);
  _mm_storeu_si128(compressed + 5, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 12), 12));
  tmp = _mm_loadu_si128(in + 13);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 29));
  w1 = _mm_srli_epi32(tmp, 3);
  _mm_storeu_si128(compressed + 6, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 14), 14));
  tmp = _mm_loadu_si128(in + 15);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 31));
  w0 = _mm_srli_epi32(tmp, 1);
  _mm_storeu_si128(compressed + 8, w0);
}

/* we are going to pack 64 18-bit values, touching 9 128-bit words, using 144
 * bytes */
static void simdpackblock18(const u32 *pin, __m128i *compressed) {
  const __m128i *in = (const __m128i *)pin;
  /* we are going to touch  9 128-bit words */
  __m128i w0, w1;
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_loadu_si128(in + 0);
  tmp = _mm_loadu_si128(in + 1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 18));
  w1 = _mm_srli_epi32(tmp, 14);
  _mm_storeu_si128(compressed + 0, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 2), 4));
  tmp = _mm_loadu_si128(in + 3);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 22));
  w0 = _mm_srli_epi32(tmp, 10);
  _mm_storeu_si128(compressed + 1, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 4), 8));
  tmp = _mm_loadu_si128(in + 5);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 26));
  w1 = _mm_srli_epi32(tmp, 6);
  _mm_storeu_si128(compressed + 2, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 6), 12));
  tmp = _mm_loadu_si128(in + 7);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 30));
  w0 = _mm_srli_epi32(tmp, 2);
  _mm_storeu_si128(compressed + 3, w1);
  tmp = _mm_loadu_si128(in + 8);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 16));
  w1 = _mm_srli_epi32(tmp, 16);
  _mm_storeu_si128(compressed + 4, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 9), 2));
  tmp = _mm_loadu_si128(in + 10);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 20));
  w0 = _mm_srli_epi32(tmp, 12);
  _mm_storeu_si128(compressed + 5, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 11), 6));
  tmp = _mm_loadu_si128(in + 12);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 24));
  w1 = _mm_srli_epi32(tmp, 8);
  _mm_storeu_si128(compressed + 6, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 13), 10));
  tmp = _mm_loadu_si128(in + 14);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 28));
  w0 = _mm_srli_epi32(tmp, 4);
  _mm_storeu_si128(compressed + 7, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 15), 14));
  _mm_storeu_si128(compressed + 8, w0);
}

/* we are going to pack 64 19-bit values, touching 10 128-bit words, using 160
 * bytes */
static void simdpackblock19(const u32 *pin, __m128i *compressed) {
  const __m128i *in = (const __m128i *)pin;
  /* we are going to touch  10 128-bit words */
  __m128i w0, w1;
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_loadu_si128(in + 0);
  tmp = _mm_loadu_si128(in + 1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 19));
  w1 = _mm_srli_epi32(tmp, 13);
  _mm_storeu_si128(compressed + 0, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 2), 6));
  tmp = _mm_loadu_si128(in + 3);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 25));
  w0 = _mm_srli_epi32(tmp, 7);
  _mm_storeu_si128(compressed + 1, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 4), 12));
  tmp = _mm_loadu_si128(in + 5);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 31));
  w1 = _mm_srli_epi32(tmp, 1);
  _mm_storeu_si128(compressed + 2, w0);
  tmp = _mm_loadu_si128(in + 6);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 18));
  w0 = _mm_srli_epi32(tmp, 14);
  _mm_storeu_si128(compressed + 3, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 7), 5));
  tmp = _mm_loadu_si128(in + 8);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 24));
  w1 = _mm_srli_epi32(tmp, 8);
  _mm_storeu_si128(compressed + 4, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 9), 11));
  tmp = _mm_loadu_si128(in + 10);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 30));
  w0 = _mm_srli_epi32(tmp, 2);
  _mm_storeu_si128(compressed + 5, w1);
  tmp = _mm_loadu_si128(in + 11);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 17));
  w1 = _mm_srli_epi32(tmp, 15);
  _mm_storeu_si128(compressed + 6, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 12), 4));
  tmp = _mm_loadu_si128(in + 13);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 23));
  w0 = _mm_srli_epi32(tmp, 9);
  _mm_storeu_si128(compressed + 7, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 14), 10));
  tmp = _mm_loadu_si128(in + 15);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 29));
  w1 = _mm_srli_epi32(tmp, 3);
  _mm_storeu_si128(compressed + 9, w1);
}

/* we are going to pack 64 20-bit values, touching 10 128-bit words, using 160
 * bytes */
static void simdpackblock20(const u32 *pin, __m128i *compressed) {
  const __m128i *in = (const __m128i *)pin;
  /* we are going to touch  10 128-bit words */
  __m128i w0, w1;
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_loadu_si128(in + 0);
  tmp = _mm_loadu_si128(in + 1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 20));
  w1 = _mm_srli_epi32(tmp, 12);
  _mm_storeu_si128(compressed + 0, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 2), 8));
  tmp = _mm_loadu_si128(in + 3);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 28));
  w0 = _mm_srli_epi32(tmp, 4);
  _mm_storeu_si128(compressed + 1, w1);
  tmp = _mm_loadu_si128(in + 4);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 16));
  w1 = _mm_srli_epi32(tmp, 16);
  _mm_storeu_si128(compressed + 2, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 5), 4));
  tmp = _mm_loadu_si128(in + 6);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 24));
  w0 = _mm_srli_epi32(tmp, 8);
  _mm_storeu_si128(compressed + 3, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 7), 12));
  _mm_storeu_si128(compressed + 4, w0);
  w1 = _mm_loadu_si128(in + 8);
  tmp = _mm_loadu_si128(in + 9);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 20));
  w0 = _mm_srli_epi32(tmp, 12);
  _mm_storeu_si128(compressed + 5, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 10), 8));
  tmp = _mm_loadu_si128(in + 11);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 28));
  w1 = _mm_srli_epi32(tmp, 4);
  _mm_storeu_si128(compressed + 6, w0);
  tmp = _mm_loadu_si128(in + 12);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 16));
  w0 = _mm_srli_epi32(tmp, 16);
  _mm_storeu_si128(compressed + 7, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 13), 4));
  tmp = _mm_loadu_si128(in + 14);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 24));
  w1 = _mm_srli_epi32(tmp, 8);
  _mm_storeu_si128(compressed + 8, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 15), 12));
  _mm_storeu_si128(compressed + 9, w1);
}

/* we are going to pack 64 21-bit values, touching 11 128-bit words, using 176
 * bytes */
static void simdpackblock21(const u32 *pin, __m128i *compressed) {
  const __m128i *in = (const __m128i *)pin;
  /* we are going to touch  11 128-bit words */
  __m128i w0, w1;
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_loadu_si128(in + 0);
  tmp = _mm_loadu_si128(in + 1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 21));
  w1 = _mm_srli_epi32(tmp, 11);
  _mm_storeu_si128(compressed + 0, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 2), 10));
  tmp = _mm_loadu_si128(in + 3);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 31));
  w0 = _mm_srli_epi32(tmp, 1);
  _mm_storeu_si128(compressed + 1, w1);
  tmp = _mm_loadu_si128(in + 4);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 20));
  w1 = _mm_srli_epi32(tmp, 12);
  _mm_storeu_si128(compressed + 2, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 5), 9));
  tmp = _mm_loadu_si128(in + 6);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 30));
  w0 = _mm_srli_epi32(tmp, 2);
  _mm_storeu_si128(compressed + 3, w1);
  tmp = _mm_loadu_si128(in + 7);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 19));
  w1 = _mm_srli_epi32(tmp, 13);
  _mm_storeu_si128(compressed + 4, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 8), 8));
  tmp = _mm_loadu_si128(in + 9);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 29));
  w0 = _mm_srli_epi32(tmp, 3);
  _mm_storeu_si128(compressed + 5, w1);
  tmp = _mm_loadu_si128(in + 10);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 18));
  w1 = _mm_srli_epi32(tmp, 14);
  _mm_storeu_si128(compressed + 6, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 11), 7));
  tmp = _mm_loadu_si128(in + 12);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 28));
  w0 = _mm_srli_epi32(tmp, 4);
  _mm_storeu_si128(compressed + 7, w1);
  tmp = _mm_loadu_si128(in + 13);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 17));
  w1 = _mm_srli_epi32(tmp, 15);
  _mm_storeu_si128(compressed + 8, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 14), 6));
  tmp = _mm_loadu_si128(in + 15);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 27));
  w0 = _mm_srli_epi32(tmp, 5);
  _mm_storeu_si128(compressed + 10, w0);
}

/* we are going to pack 64 22-bit values, touching 11 128-bit words, using 176
 * bytes */
static void simdpackblock22(const u32 *pin, __m128i *compressed) {
  const __m128i *in = (const __m128i *)pin;
  /* we are going to touch  11 128-bit words */
  __m128i w0, w1;
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_loadu_si128(in + 0);
  tmp = _mm_loadu_si128(in + 1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 22));
  w1 = _mm_srli_epi32(tmp, 10);
  _mm_storeu_si128(compressed + 0, w0);
  tmp = _mm_loadu_si128(in + 2);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 12));
  w0 = _mm_srli_epi32(tmp, 20);
  _mm_storeu_si128(compressed + 1, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 3), 2));
  tmp = _mm_loadu_si128(in + 4);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 24));
  w1 = _mm_srli_epi32(tmp, 8);
  _mm_storeu_si128(compressed + 2, w0);
  tmp = _mm_loadu_si128(in + 5);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 14));
  w0 = _mm_srli_epi32(tmp, 18);
  _mm_storeu_si128(compressed + 3, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 6), 4));
  tmp = _mm_loadu_si128(in + 7);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 26));
  w1 = _mm_srli_epi32(tmp, 6);
  _mm_storeu_si128(compressed + 4, w0);
  tmp = _mm_loadu_si128(in + 8);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 16));
  w0 = _mm_srli_epi32(tmp, 16);
  _mm_storeu_si128(compressed + 5, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 9), 6));
  tmp = _mm_loadu_si128(in + 10);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 28));
  w1 = _mm_srli_epi32(tmp, 4);
  _mm_storeu_si128(compressed + 6, w0);
  tmp = _mm_loadu_si128(in + 11);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 18));
  w0 = _mm_srli_epi32(tmp, 14);
  _mm_storeu_si128(compressed + 7, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 12), 8));
  tmp = _mm_loadu_si128(in + 13);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 30));
  w1 = _mm_srli_epi32(tmp, 2);
  _mm_storeu_si128(compressed + 8, w0);
  tmp = _mm_loadu_si128(in + 14);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 20));
  w0 = _mm_srli_epi32(tmp, 12);
  _mm_storeu_si128(compressed + 9, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 15), 10));
  _mm_storeu_si128(compressed + 10, w0);
}

/* we are going to pack 64 23-bit values, touching 12 128-bit words, using 192
 * bytes */
static void simdpackblock23(const u32 *pin, __m128i *compressed) {
  const __m128i *in = (const __m128i *)pin;
  /* we are going to touch  12 128-bit words */
  __m128i w0, w1;
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_loadu_si128(in + 0);
  tmp = _mm_loadu_si128(in + 1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 23));
  w1 = _mm_srli_epi32(tmp, 9);
  _mm_storeu_si128(compressed + 0, w0);
  tmp = _mm_loadu_si128(in + 2);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 14));
  w0 = _mm_srli_epi32(tmp, 18);
  _mm_storeu_si128(compressed + 1, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 3), 5));
  tmp = _mm_loadu_si128(in + 4);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 28));
  w1 = _mm_srli_epi32(tmp, 4);
  _mm_storeu_si128(compressed + 2, w0);
  tmp = _mm_loadu_si128(in + 5);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 19));
  w0 = _mm_srli_epi32(tmp, 13);
  _mm_storeu_si128(compressed + 3, w1);
  tmp = _mm_loadu_si128(in + 6);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 10));
  w1 = _mm_srli_epi32(tmp, 22);
  _mm_storeu_si128(compressed + 4, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 7), 1));
  tmp = _mm_loadu_si128(in + 8);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 24));
  w0 = _mm_srli_epi32(tmp, 8);
  _mm_storeu_si128(compressed + 5, w1);
  tmp = _mm_loadu_si128(in + 9);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 15));
  w1 = _mm_srli_epi32(tmp, 17);
  _mm_storeu_si128(compressed + 6, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 10), 6));
  tmp = _mm_loadu_si128(in + 11);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 29));
  w0 = _mm_srli_epi32(tmp, 3);
  _mm_storeu_si128(compressed + 7, w1);
  tmp = _mm_loadu_si128(in + 12);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 20));
  w1 = _mm_srli_epi32(tmp, 12);
  _mm_storeu_si128(compressed + 8, w0);
  tmp = _mm_loadu_si128(in + 13);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 11));
  w0 = _mm_srli_epi32(tmp, 21);
  _mm_storeu_si128(compressed + 9, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 14), 2));
  tmp = _mm_loadu_si128(in + 15);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 25));
  w1 = _mm_srli_epi32(tmp, 7);
  _mm_storeu_si128(compressed + 11, w1);
}

/* we are going to pack 64 24-bit values, touching 12 128-bit words, using 192
 * bytes */
static void simdpackblock24(const u32 *pin, __m128i *compressed) {
  const __m128i *in = (const __m128i *)pin;
  /* we are going to touch  12 128-bit words */
  __m128i w0, w1;
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_loadu_si128(in + 0);
  tmp = _mm_loadu_si128(in + 1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 24));
  w1 = _mm_srli_epi32(tmp, 8);
  _mm_storeu_si128(compressed + 0, w0);
  tmp = _mm_loadu_si128(in + 2);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 16));
  w0 = _mm_srli_epi32(tmp, 16);
  _mm_storeu_si128(compressed + 1, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 3), 8));
  _mm_storeu_si128(compressed + 2, w0);
  w1 = _mm_loadu_si128(in + 4);
  tmp = _mm_loadu_si128(in + 5);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 24));
  w0 = _mm_srli_epi32(tmp, 8);
  _mm_storeu_si128(compressed + 3, w1);
  tmp = _mm_loadu_si128(in + 6);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 16));
  w1 = _mm_srli_epi32(tmp, 16);
  _mm_storeu_si128(compressed + 4, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 7), 8));
  _mm_storeu_si128(compressed + 5, w1);
  w0 = _mm_loadu_si128(in + 8);
  tmp = _mm_loadu_si128(in + 9);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 24));
  w1 = _mm_srli_epi32(tmp, 8);
  _mm_storeu_si128(compressed + 6, w0);
  tmp = _mm_loadu_si128(in + 10);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 16));
  w0 = _mm_srli_epi32(tmp, 16);
  _mm_storeu_si128(compressed + 7, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 11), 8));
  _mm_storeu_si128(compressed + 8, w0);
  w1 = _mm_loadu_si128(in + 12);
  tmp = _mm_loadu_si128(in + 13);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 24));
  w0 = _mm_srli_epi32(tmp, 8);
  _mm_storeu_si128(compressed + 9, w1);
  tmp = _mm_loadu_si128(in + 14);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 16));
  w1 = _mm_srli_epi32(tmp, 16);
  _mm_storeu_si128(compressed + 10, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 15), 8));
  _mm_storeu_si128(compressed + 11, w1);
}

/* we are going to pack 64 25-bit values, touching 13 128-bit words, using 208
 * bytes */
static void simdpackblock25(const u32 *pin, __m128i *compressed) {
  const __m128i *in = (const __m128i *)pin;
  /* we are going to touch  13 128-bit words */
  __m128i w0, w1;
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_loadu_si128(in + 0);
  tmp = _mm_loadu_si128(in + 1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 25));
  w1 = _mm_srli_epi32(tmp, 7);
  _mm_storeu_si128(compressed + 0, w0);
  tmp = _mm_loadu_si128(in + 2);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 18));
  w0 = _mm_srli_epi32(tmp, 14);
  _mm_storeu_si128(compressed + 1, w1);
  tmp = _mm_loadu_si128(in + 3);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 11));
  w1 = _mm_srli_epi32(tmp, 21);
  _mm_storeu_si128(compressed + 2, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 4), 4));
  tmp = _mm_loadu_si128(in + 5);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 29));
  w0 = _mm_srli_epi32(tmp, 3);
  _mm_storeu_si128(compressed + 3, w1);
  tmp = _mm_loadu_si128(in + 6);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 22));
  w1 = _mm_srli_epi32(tmp, 10);
  _mm_storeu_si128(compressed + 4, w0);
  tmp = _mm_loadu_si128(in + 7);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 15));
  w0 = _mm_srli_epi32(tmp, 17);
  _mm_storeu_si128(compressed + 5, w1);
  tmp = _mm_loadu_si128(in + 8);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 8));
  w1 = _mm_srli_epi32(tmp, 24);
  _mm_storeu_si128(compressed + 6, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 9), 1));
  tmp = _mm_loadu_si128(in + 10);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 26));
  w0 = _mm_srli_epi32(tmp, 6);
  _mm_storeu_si128(compressed + 7, w1);
  tmp = _mm_loadu_si128(in + 11);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 19));
  w1 = _mm_srli_epi32(tmp, 13);
  _mm_storeu_si128(compressed + 8, w0);
  tmp = _mm_loadu_si128(in + 12);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 12));
  w0 = _mm_srli_epi32(tmp, 20);
  _mm_storeu_si128(compressed + 9, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 13), 5));
  tmp = _mm_loadu_si128(in + 14);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 30));
  w1 = _mm_srli_epi32(tmp, 2);
  _mm_storeu_si128(compressed + 10, w0);
  tmp = _mm_loadu_si128(in + 15);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 23));
  w0 = _mm_srli_epi32(tmp, 9);
  _mm_storeu_si128(compressed + 12, w0);
}

/* we are going to pack 64 26-bit values, touching 13 128-bit words, using 208
 * bytes */
static void simdpackblock26(const u32 *pin, __m128i *compressed) {
  const __m128i *in = (const __m128i *)pin;
  /* we are going to touch  13 128-bit words */
  __m128i w0, w1;
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_loadu_si128(in + 0);
  tmp = _mm_loadu_si128(in + 1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 26));
  w1 = _mm_srli_epi32(tmp, 6);
  _mm_storeu_si128(compressed + 0, w0);
  tmp = _mm_loadu_si128(in + 2);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 20));
  w0 = _mm_srli_epi32(tmp, 12);
  _mm_storeu_si128(compressed + 1, w1);
  tmp = _mm_loadu_si128(in + 3);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 14));
  w1 = _mm_srli_epi32(tmp, 18);
  _mm_storeu_si128(compressed + 2, w0);
  tmp = _mm_loadu_si128(in + 4);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 8));
  w0 = _mm_srli_epi32(tmp, 24);
  _mm_storeu_si128(compressed + 3, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 5), 2));
  tmp = _mm_loadu_si128(in + 6);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 28));
  w1 = _mm_srli_epi32(tmp, 4);
  _mm_storeu_si128(compressed + 4, w0);
  tmp = _mm_loadu_si128(in + 7);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 22));
  w0 = _mm_srli_epi32(tmp, 10);
  _mm_storeu_si128(compressed + 5, w1);
  tmp = _mm_loadu_si128(in + 8);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 16));
  w1 = _mm_srli_epi32(tmp, 16);
  _mm_storeu_si128(compressed + 6, w0);
  tmp = _mm_loadu_si128(in + 9);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 10));
  w0 = _mm_srli_epi32(tmp, 22);
  _mm_storeu_si128(compressed + 7, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 10), 4));
  tmp = _mm_loadu_si128(in + 11);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 30));
  w1 = _mm_srli_epi32(tmp, 2);
  _mm_storeu_si128(compressed + 8, w0);
  tmp = _mm_loadu_si128(in + 12);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 24));
  w0 = _mm_srli_epi32(tmp, 8);
  _mm_storeu_si128(compressed + 9, w1);
  tmp = _mm_loadu_si128(in + 13);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 18));
  w1 = _mm_srli_epi32(tmp, 14);
  _mm_storeu_si128(compressed + 10, w0);
  tmp = _mm_loadu_si128(in + 14);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 12));
  w0 = _mm_srli_epi32(tmp, 20);
  _mm_storeu_si128(compressed + 11, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 15), 6));
  _mm_storeu_si128(compressed + 12, w0);
}

/* we are going to pack 64 27-bit values, touching 14 128-bit words, using 224
 * bytes */
static void simdpackblock27(const u32 *pin, __m128i *compressed) {
  const __m128i *in = (const __m128i *)pin;
  /* we are going to touch  14 128-bit words */
  __m128i w0, w1;
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_loadu_si128(in + 0);
  tmp = _mm_loadu_si128(in + 1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 27));
  w1 = _mm_srli_epi32(tmp, 5);
  _mm_storeu_si128(compressed + 0, w0);
  tmp = _mm_loadu_si128(in + 2);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 22));
  w0 = _mm_srli_epi32(tmp, 10);
  _mm_storeu_si128(compressed + 1, w1);
  tmp = _mm_loadu_si128(in + 3);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 17));
  w1 = _mm_srli_epi32(tmp, 15);
  _mm_storeu_si128(compressed + 2, w0);
  tmp = _mm_loadu_si128(in + 4);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 12));
  w0 = _mm_srli_epi32(tmp, 20);
  _mm_storeu_si128(compressed + 3, w1);
  tmp = _mm_loadu_si128(in + 5);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 7));
  w1 = _mm_srli_epi32(tmp, 25);
  _mm_storeu_si128(compressed + 4, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 6), 2));
  tmp = _mm_loadu_si128(in + 7);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 29));
  w0 = _mm_srli_epi32(tmp, 3);
  _mm_storeu_si128(compressed + 5, w1);
  tmp = _mm_loadu_si128(in + 8);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 24));
  w1 = _mm_srli_epi32(tmp, 8);
  _mm_storeu_si128(compressed + 6, w0);
  tmp = _mm_loadu_si128(in + 9);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 19));
  w0 = _mm_srli_epi32(tmp, 13);
  _mm_storeu_si128(compressed + 7, w1);
  tmp = _mm_loadu_si128(in + 10);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 14));
  w1 = _mm_srli_epi32(tmp, 18);
  _mm_storeu_si128(compressed + 8, w0);
  tmp = _mm_loadu_si128(in + 11);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 9));
  w0 = _mm_srli_epi32(tmp, 23);
  _mm_storeu_si128(compressed + 9, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 12), 4));
  tmp = _mm_loadu_si128(in + 13);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 31));
  w1 = _mm_srli_epi32(tmp, 1);
  _mm_storeu_si128(compressed + 10, w0);
  tmp = _mm_loadu_si128(in + 14);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 26));
  w0 = _mm_srli_epi32(tmp, 6);
  _mm_storeu_si128(compressed + 11, w1);
  tmp = _mm_loadu_si128(in + 15);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 21));
  w1 = _mm_srli_epi32(tmp, 11);
  _mm_storeu_si128(compressed + 13, w1);
}

/* we are going to pack 64 28-bit values, touching 14 128-bit words, using 224
 * bytes */
static void simdpackblock28(const u32 *pin, __m128i *compressed) {
  const __m128i *in = (const __m128i *)pin;
  /* we are going to touch  14 128-bit words */
  __m128i w0, w1;
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_loadu_si128(in + 0);
  tmp = _mm_loadu_si128(in + 1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 28));
  w1 = _mm_srli_epi32(tmp, 4);
  _mm_storeu_si128(compressed + 0, w0);
  tmp = _mm_loadu_si128(in + 2);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 24));
  w0 = _mm_srli_epi32(tmp, 8);
  _mm_storeu_si128(compressed + 1, w1);
  tmp = _mm_loadu_si128(in + 3);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 20));
  w1 = _mm_srli_epi32(tmp, 12);
  _mm_storeu_si128(compressed + 2, w0);
  tmp = _mm_loadu_si128(in + 4);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 16));
  w0 = _mm_srli_epi32(tmp, 16);
  _mm_storeu_si128(compressed + 3, w1);
  tmp = _mm_loadu_si128(in + 5);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 12));
  w1 = _mm_srli_epi32(tmp, 20);
  _mm_storeu_si128(compressed + 4, w0);
  tmp = _mm_loadu_si128(in + 6);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 8));
  w0 = _mm_srli_epi32(tmp, 24);
  _mm_storeu_si128(compressed + 5, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 7), 4));
  _mm_storeu_si128(compressed + 6, w0);
  w1 = _mm_loadu_si128(in + 8);
  tmp = _mm_loadu_si128(in + 9);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 28));
  w0 = _mm_srli_epi32(tmp, 4);
  _mm_storeu_si128(compressed + 7, w1);
  tmp = _mm_loadu_si128(in + 10);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 24));
  w1 = _mm_srli_epi32(tmp, 8);
  _mm_storeu_si128(compressed + 8, w0);
  tmp = _mm_loadu_si128(in + 11);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 20));
  w0 = _mm_srli_epi32(tmp, 12);
  _mm_storeu_si128(compressed + 9, w1);
  tmp = _mm_loadu_si128(in + 12);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 16));
  w1 = _mm_srli_epi32(tmp, 16);
  _mm_storeu_si128(compressed + 10, w0);
  tmp = _mm_loadu_si128(in + 13);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 12));
  w0 = _mm_srli_epi32(tmp, 20);
  _mm_storeu_si128(compressed + 11, w1);
  tmp = _mm_loadu_si128(in + 14);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 8));
  w1 = _mm_srli_epi32(tmp, 24);
  _mm_storeu_si128(compressed + 12, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 15), 4));
  _mm_storeu_si128(compressed + 13, w1);
}

/* we are going to pack 64 29-bit values, touching 15 128-bit words, using 240
 * bytes */
static void simdpackblock29(const u32 *pin, __m128i *compressed) {
  const __m128i *in = (const __m128i *)pin;
  /* we are going to touch  15 128-bit words */
  __m128i w0, w1;
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_loadu_si128(in + 0);
  tmp = _mm_loadu_si128(in + 1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 29));
  w1 = _mm_srli_epi32(tmp, 3);
  _mm_storeu_si128(compressed + 0, w0);
  tmp = _mm_loadu_si128(in + 2);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 26));
  w0 = _mm_srli_epi32(tmp, 6);
  _mm_storeu_si128(compressed + 1, w1);
  tmp = _mm_loadu_si128(in + 3);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 23));
  w1 = _mm_srli_epi32(tmp, 9);
  _mm_storeu_si128(compressed + 2, w0);
  tmp = _mm_loadu_si128(in + 4);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 20));
  w0 = _mm_srli_epi32(tmp, 12);
  _mm_storeu_si128(compressed + 3, w1);
  tmp = _mm_loadu_si128(in + 5);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 17));
  w1 = _mm_srli_epi32(tmp, 15);
  _mm_storeu_si128(compressed + 4, w0);
  tmp = _mm_loadu_si128(in + 6);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 14));
  w0 = _mm_srli_epi32(tmp, 18);
  _mm_storeu_si128(compressed + 5, w1);
  tmp = _mm_loadu_si128(in + 7);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 11));
  w1 = _mm_srli_epi32(tmp, 21);
  _mm_storeu_si128(compressed + 6, w0);
  tmp = _mm_loadu_si128(in + 8);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 8));
  w0 = _mm_srli_epi32(tmp, 24);
  _mm_storeu_si128(compressed + 7, w1);
  tmp = _mm_loadu_si128(in + 9);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 5));
  w1 = _mm_srli_epi32(tmp, 27);
  _mm_storeu_si128(compressed + 8, w0);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(_mm_loadu_si128(in + 10), 2));
  tmp = _mm_loadu_si128(in + 11);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 31));
  w0 = _mm_srli_epi32(tmp, 1);
  _mm_storeu_si128(compressed + 9, w1);
  tmp = _mm_loadu_si128(in + 12);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 28));
  w1 = _mm_srli_epi32(tmp, 4);
  _mm_storeu_si128(compressed + 10, w0);
  tmp = _mm_loadu_si128(in + 13);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 25));
  w0 = _mm_srli_epi32(tmp, 7);
  _mm_storeu_si128(compressed + 11, w1);
  tmp = _mm_loadu_si128(in + 14);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 22));
  w1 = _mm_srli_epi32(tmp, 10);
  _mm_storeu_si128(compressed + 12, w0);
  tmp = _mm_loadu_si128(in + 15);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 19));
  w0 = _mm_srli_epi32(tmp, 13);
  _mm_storeu_si128(compressed + 14, w0);
}

/* we are going to pack 64 30-bit values, touching 15 128-bit words, using 240
 * bytes */
static void simdpackblock30(const u32 *pin, __m128i *compressed) {
  const __m128i *in = (const __m128i *)pin;
  /* we are going to touch  15 128-bit words */
  __m128i w0, w1;
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_loadu_si128(in + 0);
  tmp = _mm_loadu_si128(in + 1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 30));
  w1 = _mm_srli_epi32(tmp, 2);
  _mm_storeu_si128(compressed + 0, w0);
  tmp = _mm_loadu_si128(in + 2);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 28));
  w0 = _mm_srli_epi32(tmp, 4);
  _mm_storeu_si128(compressed + 1, w1);
  tmp = _mm_loadu_si128(in + 3);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 26));
  w1 = _mm_srli_epi32(tmp, 6);
  _mm_storeu_si128(compressed + 2, w0);
  tmp = _mm_loadu_si128(in + 4);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 24));
  w0 = _mm_srli_epi32(tmp, 8);
  _mm_storeu_si128(compressed + 3, w1);
  tmp = _mm_loadu_si128(in + 5);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 22));
  w1 = _mm_srli_epi32(tmp, 10);
  _mm_storeu_si128(compressed + 4, w0);
  tmp = _mm_loadu_si128(in + 6);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 20));
  w0 = _mm_srli_epi32(tmp, 12);
  _mm_storeu_si128(compressed + 5, w1);
  tmp = _mm_loadu_si128(in + 7);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 18));
  w1 = _mm_srli_epi32(tmp, 14);
  _mm_storeu_si128(compressed + 6, w0);
  tmp = _mm_loadu_si128(in + 8);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 16));
  w0 = _mm_srli_epi32(tmp, 16);
  _mm_storeu_si128(compressed + 7, w1);
  tmp = _mm_loadu_si128(in + 9);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 14));
  w1 = _mm_srli_epi32(tmp, 18);
  _mm_storeu_si128(compressed + 8, w0);
  tmp = _mm_loadu_si128(in + 10);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 12));
  w0 = _mm_srli_epi32(tmp, 20);
  _mm_storeu_si128(compressed + 9, w1);
  tmp = _mm_loadu_si128(in + 11);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 10));
  w1 = _mm_srli_epi32(tmp, 22);
  _mm_storeu_si128(compressed + 10, w0);
  tmp = _mm_loadu_si128(in + 12);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 8));
  w0 = _mm_srli_epi32(tmp, 24);
  _mm_storeu_si128(compressed + 11, w1);
  tmp = _mm_loadu_si128(in + 13);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 6));
  w1 = _mm_srli_epi32(tmp, 26);
  _mm_storeu_si128(compressed + 12, w0);
  tmp = _mm_loadu_si128(in + 14);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 4));
  w0 = _mm_srli_epi32(tmp, 28);
  _mm_storeu_si128(compressed + 13, w1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(_mm_loadu_si128(in + 15), 2));
  _mm_storeu_si128(compressed + 14, w0);
}

/* we are going to pack 64 31-bit values, touching 16 128-bit words, using 256
 * bytes */
static void simdpackblock31(const u32 *pin, __m128i *compressed) {
  const __m128i *in = (const __m128i *)pin;
  /* we are going to touch  16 128-bit words */
  __m128i w0, w1;
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_loadu_si128(in + 0);
  tmp = _mm_loadu_si128(in + 1);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 31));
  w1 = _mm_srli_epi32(tmp, 1);
  _mm_storeu_si128(compressed + 0, w0);
  tmp = _mm_loadu_si128(in + 2);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 30));
  w0 = _mm_srli_epi32(tmp, 2);
  _mm_storeu_si128(compressed + 1, w1);
  tmp = _mm_loadu_si128(in + 3);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 29));
  w1 = _mm_srli_epi32(tmp, 3);
  _mm_storeu_si128(compressed + 2, w0);
  tmp = _mm_loadu_si128(in + 4);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 28));
  w0 = _mm_srli_epi32(tmp, 4);
  _mm_storeu_si128(compressed + 3, w1);
  tmp = _mm_loadu_si128(in + 5);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 27));
  w1 = _mm_srli_epi32(tmp, 5);
  _mm_storeu_si128(compressed + 4, w0);
  tmp = _mm_loadu_si128(in + 6);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 26));
  w0 = _mm_srli_epi32(tmp, 6);
  _mm_storeu_si128(compressed + 5, w1);
  tmp = _mm_loadu_si128(in + 7);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 25));
  w1 = _mm_srli_epi32(tmp, 7);
  _mm_storeu_si128(compressed + 6, w0);
  tmp = _mm_loadu_si128(in + 8);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 24));
  w0 = _mm_srli_epi32(tmp, 8);
  _mm_storeu_si128(compressed + 7, w1);
  tmp = _mm_loadu_si128(in + 9);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 23));
  w1 = _mm_srli_epi32(tmp, 9);
  _mm_storeu_si128(compressed + 8, w0);
  tmp = _mm_loadu_si128(in + 10);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 22));
  w0 = _mm_srli_epi32(tmp, 10);
  _mm_storeu_si128(compressed + 9, w1);
  tmp = _mm_loadu_si128(in + 11);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 21));
  w1 = _mm_srli_epi32(tmp, 11);
  _mm_storeu_si128(compressed + 10, w0);
  tmp = _mm_loadu_si128(in + 12);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 20));
  w0 = _mm_srli_epi32(tmp, 12);
  _mm_storeu_si128(compressed + 11, w1);
  tmp = _mm_loadu_si128(in + 13);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 19));
  w1 = _mm_srli_epi32(tmp, 13);
  _mm_storeu_si128(compressed + 12, w0);
  tmp = _mm_loadu_si128(in + 14);
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 18));
  w0 = _mm_srli_epi32(tmp, 14);
  _mm_storeu_si128(compressed + 13, w1);
  tmp = _mm_loadu_si128(in + 15);
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 17));
  w1 = _mm_srli_epi32(tmp, 15);
  _mm_storeu_si128(compressed + 15, w1);
}

/* we are going to pack 64 32-bit values, touching 16 128-bit words, using 256
 * bytes */
static void simdpackblock32(const u32 *pin, __m128i *compressed) {
  const __m128i *in = (const __m128i *)pin;
  /* we are going to touch  16 128-bit words */
  __m128i w0, w1;
  w0 = _mm_loadu_si128(in + 0);
  _mm_storeu_si128(compressed + 0, w0);
  w1 = _mm_loadu_si128(in + 1);
  _mm_storeu_si128(compressed + 1, w1);
  w0 = _mm_loadu_si128(in + 2);
  _mm_storeu_si128(compressed + 2, w0);
  w1 = _mm_loadu_si128(in + 3);
  _mm_storeu_si128(compressed + 3, w1);
  w0 = _mm_loadu_si128(in + 4);
  _mm_storeu_si128(compressed + 4, w0);
  w1 = _mm_loadu_si128(in + 5);
  _mm_storeu_si128(compressed + 5, w1);
  w0 = _mm_loadu_si128(in + 6);
  _mm_storeu_si128(compressed + 6, w0);
  w1 = _mm_loadu_si128(in + 7);
  _mm_storeu_si128(compressed + 7, w1);
  w0 = _mm_loadu_si128(in + 8);
  _mm_storeu_si128(compressed + 8, w0);
  w1 = _mm_loadu_si128(in + 9);
  _mm_storeu_si128(compressed + 9, w1);
  w0 = _mm_loadu_si128(in + 10);
  _mm_storeu_si128(compressed + 10, w0);
  w1 = _mm_loadu_si128(in + 11);
  _mm_storeu_si128(compressed + 11, w1);
  w0 = _mm_loadu_si128(in + 12);
  _mm_storeu_si128(compressed + 12, w0);
  w1 = _mm_loadu_si128(in + 13);
  _mm_storeu_si128(compressed + 13, w1);
  w0 = _mm_loadu_si128(in + 14);
  _mm_storeu_si128(compressed + 14, w0);
  w1 = _mm_loadu_si128(in + 15);
  _mm_storeu_si128(compressed + 15, w1);
}

static void simdpackblockmask0(const u32 *pin, __m128i *compressed) {
  (void)compressed;
  (void)pin; /* we consumed 64 32-bit integers */
}

/* we are going to pack 64 1-bit values, touching 1 128-bit words, using 16
 * bytes */
static void simdpackblockmask1(const u32 *pin, __m128i *compressed) {
  /* we are going to touch  1 128-bit word */
  __m128i w0;
  const __m128i *in = (const __m128i *)pin;
  const __m128i mask = _mm_set1_epi32(1);
  w0 = _mm_and_si128(mask, _mm_loadu_si128(in + 0));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 1)), 1));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 2)), 2));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 3)), 3));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 4)), 4));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 5)), 5));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 6)), 6));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 7)), 7));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 8)), 8));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 9)), 9));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 10)), 10));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 11)), 11));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 12)), 12));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 13)), 13));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 14)), 14));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 15)), 15));
  _mm_storeu_si128(compressed + 0, w0);
}

/* we are going to pack 64 2-bit values, touching 1 128-bit words, using 16
 * bytes */
static void simdpackblockmask2(const u32 *pin, __m128i *compressed) {
  /* we are going to touch  1 128-bit word */
  __m128i w0;
  const __m128i *in = (const __m128i *)pin;
  const __m128i mask = _mm_set1_epi32(3);
  w0 = _mm_and_si128(mask, _mm_loadu_si128(in + 0));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 1)), 2));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 2)), 4));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 3)), 6));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 4)), 8));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 5)), 10));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 6)), 12));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 7)), 14));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 8)), 16));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 9)), 18));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 10)), 20));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 11)), 22));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 12)), 24));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 13)), 26));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 14)), 28));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 15)), 30));
  _mm_storeu_si128(compressed + 0, w0);
}

/* we are going to pack 64 3-bit values, touching 2 128-bit words, using 32
 * bytes */
static void simdpackblockmask3(const u32 *pin, __m128i *compressed) {
  /* we are going to touch  2 128-bit words */
  __m128i w0, w1;
  const __m128i *in = (const __m128i *)pin;
  const __m128i mask = _mm_set1_epi32(7);
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_and_si128(mask, _mm_loadu_si128(in + 0));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 1)), 3));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 2)), 6));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 3)), 9));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 4)), 12));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 5)), 15));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 6)), 18));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 7)), 21));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 8)), 24));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 9)), 27));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 10));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 30));
  w1 = _mm_srli_epi32(tmp, 2);
  _mm_storeu_si128(compressed + 0, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 11)), 1));
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 12)), 4));
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 13)), 7));
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 14)), 10));
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 15)), 13));
  _mm_storeu_si128(compressed + 1, w1);
}

/* we are going to pack 64 4-bit values, touching 2 128-bit words, using 32
 * bytes */
static void simdpackblockmask4(const u32 *pin, __m128i *compressed) {
  /* we are going to touch  2 128-bit words */
  __m128i w0, w1;
  const __m128i *in = (const __m128i *)pin;
  const __m128i mask = _mm_set1_epi32(15);
  w0 = _mm_and_si128(mask, _mm_loadu_si128(in + 0));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 1)), 4));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 2)), 8));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 3)), 12));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 4)), 16));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 5)), 20));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 6)), 24));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 7)), 28));
  _mm_storeu_si128(compressed + 0, w0);
  w1 = _mm_and_si128(mask, _mm_loadu_si128(in + 8));
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 9)), 4));
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 10)), 8));
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 11)), 12));
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 12)), 16));
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 13)), 20));
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 14)), 24));
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 15)), 28));
  _mm_storeu_si128(compressed + 1, w1);
}

/* we are going to pack 64 5-bit values, touching 3 128-bit words, using 48
 * bytes */
static void simdpackblockmask5(const u32 *pin, __m128i *compressed) {
  /* we are going to touch  3 128-bit words */
  __m128i w0, w1;
  const __m128i *in = (const __m128i *)pin;
  const __m128i mask = _mm_set1_epi32(31);
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_and_si128(mask, _mm_loadu_si128(in + 0));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 1)), 5));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 2)), 10));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 3)), 15));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 4)), 20));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 5)), 25));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 6));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 30));
  w1 = _mm_srli_epi32(tmp, 2);
  _mm_storeu_si128(compressed + 0, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 7)), 3));
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 8)), 8));
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 9)), 13));
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 10)), 18));
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 11)), 23));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 12));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 28));
  w0 = _mm_srli_epi32(tmp, 4);
  _mm_storeu_si128(compressed + 1, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 13)), 1));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 14)), 6));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 15)), 11));
  _mm_storeu_si128(compressed + 2, w0);
}

/* we are going to pack 64 6-bit values, touching 3 128-bit words, using 48
 * bytes */
static void simdpackblockmask6(const u32 *pin, __m128i *compressed) {
  /* we are going to touch  3 128-bit words */
  __m128i w0, w1;
  const __m128i *in = (const __m128i *)pin;
  const __m128i mask = _mm_set1_epi32(63);
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_and_si128(mask, _mm_loadu_si128(in + 0));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 1)), 6));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 2)), 12));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 3)), 18));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 4)), 24));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 5));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 30));
  w1 = _mm_srli_epi32(tmp, 2);
  _mm_storeu_si128(compressed + 0, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 6)), 4));
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 7)), 10));
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 8)), 16));
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 9)), 22));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 10));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 28));
  w0 = _mm_srli_epi32(tmp, 4);
  _mm_storeu_si128(compressed + 1, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 11)), 2));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 12)), 8));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 13)), 14));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 14)), 20));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 15)), 26));
  _mm_storeu_si128(compressed + 2, w0);
}

/* we are going to pack 64 7-bit values, touching 4 128-bit words, using 64
 * bytes */
static void simdpackblockmask7(const u32 *pin, __m128i *compressed) {
  /* we are going to touch  4 128-bit words */
  __m128i w0, w1;
  const __m128i *in = (const __m128i *)pin;
  const __m128i mask = _mm_set1_epi32(127);
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_and_si128(mask, _mm_loadu_si128(in + 0));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 1)), 7));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 2)), 14));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 3)), 21));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 4));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 28));
  w1 = _mm_srli_epi32(tmp, 4);
  _mm_storeu_si128(compressed + 0, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 5)), 3));
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 6)), 10));
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 7)), 17));
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 8)), 24));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 9));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 31));
  w0 = _mm_srli_epi32(tmp, 1);
  _mm_storeu_si128(compressed + 1, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 10)), 6));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 11)), 13));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 12)), 20));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 13));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 27));
  w1 = _mm_srli_epi32(tmp, 5);
  _mm_storeu_si128(compressed + 2, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 14)), 2));
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 15)), 9));
  _mm_storeu_si128(compressed + 3, w1);
}

/* we are going to pack 64 8-bit values, touching 4 128-bit words, using 64
 * bytes */
static void simdpackblockmask8(const u32 *pin, __m128i *compressed) {
  /* we are going to touch  4 128-bit words */
  __m128i w0, w1;
  const __m128i *in = (const __m128i *)pin;
  const __m128i mask = _mm_set1_epi32(255);
  w0 = _mm_and_si128(mask, _mm_loadu_si128(in + 0));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 1)), 8));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 2)), 16));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 3)), 24));
  _mm_storeu_si128(compressed + 0, w0);
  w1 = _mm_and_si128(mask, _mm_loadu_si128(in + 4));
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 5)), 8));
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 6)), 16));
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 7)), 24));
  _mm_storeu_si128(compressed + 1, w1);
  w0 = _mm_and_si128(mask, _mm_loadu_si128(in + 8));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 9)), 8));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 10)), 16));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 11)), 24));
  _mm_storeu_si128(compressed + 2, w0);
  w1 = _mm_and_si128(mask, _mm_loadu_si128(in + 12));
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 13)), 8));
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 14)), 16));
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 15)), 24));
  _mm_storeu_si128(compressed + 3, w1);
}

/* we are going to pack 64 9-bit values, touching 5 128-bit words, using 80
 * bytes */
static void simdpackblockmask9(const u32 *pin, __m128i *compressed) {
  /* we are going to touch  5 128-bit words */
  __m128i w0, w1;
  const __m128i *in = (const __m128i *)pin;
  const __m128i mask = _mm_set1_epi32(511);
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_and_si128(mask, _mm_loadu_si128(in + 0));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 1)), 9));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 2)), 18));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 3));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 27));
  w1 = _mm_srli_epi32(tmp, 5);
  _mm_storeu_si128(compressed + 0, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 4)), 4));
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 5)), 13));
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 6)), 22));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 7));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 31));
  w0 = _mm_srli_epi32(tmp, 1);
  _mm_storeu_si128(compressed + 1, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 8)), 8));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 9)), 17));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 10));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 26));
  w1 = _mm_srli_epi32(tmp, 6);
  _mm_storeu_si128(compressed + 2, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 11)), 3));
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 12)), 12));
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 13)), 21));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 14));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 30));
  w0 = _mm_srli_epi32(tmp, 2);
  _mm_storeu_si128(compressed + 3, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 15)), 7));
  _mm_storeu_si128(compressed + 4, w0);
}

/* we are going to pack 64 10-bit values, touching 5 128-bit words, using 80
 * bytes */
static void simdpackblockmask10(const u32 *pin, __m128i *compressed) {
  /* we are going to touch  5 128-bit words */
  __m128i w0, w1;
  const __m128i *in = (const __m128i *)pin;
  const __m128i mask = _mm_set1_epi32(1023);
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_and_si128(mask, _mm_loadu_si128(in + 0));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 1)), 10));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 2)), 20));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 3));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 30));
  w1 = _mm_srli_epi32(tmp, 2);
  _mm_storeu_si128(compressed + 0, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 4)), 8));
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 5)), 18));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 6));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 28));
  w0 = _mm_srli_epi32(tmp, 4);
  _mm_storeu_si128(compressed + 1, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 7)), 6));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 8)), 16));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 9));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 26));
  w1 = _mm_srli_epi32(tmp, 6);
  _mm_storeu_si128(compressed + 2, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 10)), 4));
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 11)), 14));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 12));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 24));
  w0 = _mm_srli_epi32(tmp, 8);
  _mm_storeu_si128(compressed + 3, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 13)), 2));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 14)), 12));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 15)), 22));
  _mm_storeu_si128(compressed + 4, w0);
}

/* we are going to pack 64 11-bit values, touching 6 128-bit words, using 96
 * bytes */
static void simdpackblockmask11(const u32 *pin, __m128i *compressed) {
  /* we are going to touch  6 128-bit words */
  __m128i w0, w1;
  const __m128i *in = (const __m128i *)pin;
  const __m128i mask = _mm_set1_epi32(2047);
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_and_si128(mask, _mm_loadu_si128(in + 0));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 1)), 11));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 2));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 22));
  w1 = _mm_srli_epi32(tmp, 10);
  _mm_storeu_si128(compressed + 0, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 3)), 1));
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 4)), 12));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 5));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 23));
  w0 = _mm_srli_epi32(tmp, 9);
  _mm_storeu_si128(compressed + 1, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 6)), 2));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 7)), 13));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 8));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 24));
  w1 = _mm_srli_epi32(tmp, 8);
  _mm_storeu_si128(compressed + 2, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 9)), 3));
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 10)), 14));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 11));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 25));
  w0 = _mm_srli_epi32(tmp, 7);
  _mm_storeu_si128(compressed + 3, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 12)), 4));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 13)), 15));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 14));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 26));
  w1 = _mm_srli_epi32(tmp, 6);
  _mm_storeu_si128(compressed + 4, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 15)), 5));
  _mm_storeu_si128(compressed + 5, w1);
}

/* we are going to pack 64 12-bit values, touching 6 128-bit words, using 96
 * bytes */
static void simdpackblockmask12(const u32 *pin, __m128i *compressed) {
  /* we are going to touch  6 128-bit words */
  __m128i w0, w1;
  const __m128i *in = (const __m128i *)pin;
  const __m128i mask = _mm_set1_epi32(4095);
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_and_si128(mask, _mm_loadu_si128(in + 0));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 1)), 12));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 2));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 24));
  w1 = _mm_srli_epi32(tmp, 8);
  _mm_storeu_si128(compressed + 0, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 3)), 4));
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 4)), 16));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 5));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 28));
  w0 = _mm_srli_epi32(tmp, 4);
  _mm_storeu_si128(compressed + 1, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 6)), 8));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 7)), 20));
  _mm_storeu_si128(compressed + 2, w0);
  w1 = _mm_and_si128(mask, _mm_loadu_si128(in + 8));
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 9)), 12));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 10));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 24));
  w0 = _mm_srli_epi32(tmp, 8);
  _mm_storeu_si128(compressed + 3, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 11)), 4));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 12)), 16));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 13));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 28));
  w1 = _mm_srli_epi32(tmp, 4);
  _mm_storeu_si128(compressed + 4, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 14)), 8));
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 15)), 20));
  _mm_storeu_si128(compressed + 5, w1);
}

/* we are going to pack 64 13-bit values, touching 7 128-bit words, using 112
 * bytes */
static void simdpackblockmask13(const u32 *pin, __m128i *compressed) {
  /* we are going to touch  7 128-bit words */
  __m128i w0, w1;
  const __m128i *in = (const __m128i *)pin;
  const __m128i mask = _mm_set1_epi32(8191);
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_and_si128(mask, _mm_loadu_si128(in + 0));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 1)), 13));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 2));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 26));
  w1 = _mm_srli_epi32(tmp, 6);
  _mm_storeu_si128(compressed + 0, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 3)), 7));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 4));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 20));
  w0 = _mm_srli_epi32(tmp, 12);
  _mm_storeu_si128(compressed + 1, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 5)), 1));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 6)), 14));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 7));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 27));
  w1 = _mm_srli_epi32(tmp, 5);
  _mm_storeu_si128(compressed + 2, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 8)), 8));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 9));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 21));
  w0 = _mm_srli_epi32(tmp, 11);
  _mm_storeu_si128(compressed + 3, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 10)), 2));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 11)), 15));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 12));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 28));
  w1 = _mm_srli_epi32(tmp, 4);
  _mm_storeu_si128(compressed + 4, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 13)), 9));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 14));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 22));
  w0 = _mm_srli_epi32(tmp, 10);
  _mm_storeu_si128(compressed + 5, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 15)), 3));
  _mm_storeu_si128(compressed + 6, w0);
}

/* we are going to pack 64 14-bit values, touching 7 128-bit words, using 112
 * bytes */
static void simdpackblockmask14(const u32 *pin, __m128i *compressed) {
  /* we are going to touch  7 128-bit words */
  __m128i w0, w1;
  const __m128i *in = (const __m128i *)pin;
  const __m128i mask = _mm_set1_epi32(16383);
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_and_si128(mask, _mm_loadu_si128(in + 0));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 1)), 14));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 2));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 28));
  w1 = _mm_srli_epi32(tmp, 4);
  _mm_storeu_si128(compressed + 0, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 3)), 10));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 4));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 24));
  w0 = _mm_srli_epi32(tmp, 8);
  _mm_storeu_si128(compressed + 1, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 5)), 6));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 6));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 20));
  w1 = _mm_srli_epi32(tmp, 12);
  _mm_storeu_si128(compressed + 2, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 7)), 2));
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 8)), 16));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 9));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 30));
  w0 = _mm_srli_epi32(tmp, 2);
  _mm_storeu_si128(compressed + 3, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 10)), 12));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 11));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 26));
  w1 = _mm_srli_epi32(tmp, 6);
  _mm_storeu_si128(compressed + 4, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 12)), 8));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 13));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 22));
  w0 = _mm_srli_epi32(tmp, 10);
  _mm_storeu_si128(compressed + 5, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 14)), 4));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 15)), 18));
  _mm_storeu_si128(compressed + 6, w0);
}

/* we are going to pack 64 15-bit values, touching 8 128-bit words, using 128
 * bytes */
static void simdpackblockmask15(const u32 *pin, __m128i *compressed) {
  /* we are going to touch  8 128-bit words */
  __m128i w0, w1;
  const __m128i *in = (const __m128i *)pin;
  const __m128i mask = _mm_set1_epi32(32767);
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_and_si128(mask, _mm_loadu_si128(in + 0));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 1)), 15));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 2));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 30));
  w1 = _mm_srli_epi32(tmp, 2);
  _mm_storeu_si128(compressed + 0, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 3)), 13));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 4));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 28));
  w0 = _mm_srli_epi32(tmp, 4);
  _mm_storeu_si128(compressed + 1, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 5)), 11));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 6));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 26));
  w1 = _mm_srli_epi32(tmp, 6);
  _mm_storeu_si128(compressed + 2, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 7)), 9));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 8));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 24));
  w0 = _mm_srli_epi32(tmp, 8);
  _mm_storeu_si128(compressed + 3, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 9)), 7));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 10));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 22));
  w1 = _mm_srli_epi32(tmp, 10);
  _mm_storeu_si128(compressed + 4, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 11)), 5));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 12));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 20));
  w0 = _mm_srli_epi32(tmp, 12);
  _mm_storeu_si128(compressed + 5, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 13)), 3));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 14));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 18));
  w1 = _mm_srli_epi32(tmp, 14);
  _mm_storeu_si128(compressed + 6, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 15)), 1));
  _mm_storeu_si128(compressed + 7, w1);
}

/* we are going to pack 64 16-bit values, touching 8 128-bit words, using 128
 * bytes */
static void simdpackblockmask16(const u32 *pin, __m128i *compressed) {
  /* we are going to touch  8 128-bit words */
  __m128i w0, w1;
  const __m128i *in = (const __m128i *)pin;
  const __m128i mask = _mm_set1_epi32(65535);
  w0 = _mm_and_si128(mask, _mm_loadu_si128(in + 0));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 1)), 16));
  _mm_storeu_si128(compressed + 0, w0);
  w1 = _mm_and_si128(mask, _mm_loadu_si128(in + 2));
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 3)), 16));
  _mm_storeu_si128(compressed + 1, w1);
  w0 = _mm_and_si128(mask, _mm_loadu_si128(in + 4));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 5)), 16));
  _mm_storeu_si128(compressed + 2, w0);
  w1 = _mm_and_si128(mask, _mm_loadu_si128(in + 6));
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 7)), 16));
  _mm_storeu_si128(compressed + 3, w1);
  w0 = _mm_and_si128(mask, _mm_loadu_si128(in + 8));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 9)), 16));
  _mm_storeu_si128(compressed + 4, w0);
  w1 = _mm_and_si128(mask, _mm_loadu_si128(in + 10));
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 11)), 16));
  _mm_storeu_si128(compressed + 5, w1);
  w0 = _mm_and_si128(mask, _mm_loadu_si128(in + 12));
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 13)), 16));
  _mm_storeu_si128(compressed + 6, w0);
  w1 = _mm_and_si128(mask, _mm_loadu_si128(in + 14));
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 15)), 16));
  _mm_storeu_si128(compressed + 7, w1);
}

/* we are going to pack 64 17-bit values, touching 9 128-bit words, using 144
 * bytes */
static void simdpackblockmask17(const u32 *pin, __m128i *compressed) {
  /* we are going to touch  9 128-bit words */
  __m128i w0, w1;
  const __m128i *in = (const __m128i *)pin;
  const __m128i mask = _mm_set1_epi32(131071);
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_and_si128(mask, _mm_loadu_si128(in + 0));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 1));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 17));
  w1 = _mm_srli_epi32(tmp, 15);
  _mm_storeu_si128(compressed + 0, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 2)), 2));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 3));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 19));
  w0 = _mm_srli_epi32(tmp, 13);
  _mm_storeu_si128(compressed + 1, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 4)), 4));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 5));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 21));
  w1 = _mm_srli_epi32(tmp, 11);
  _mm_storeu_si128(compressed + 2, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 6)), 6));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 7));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 23));
  w0 = _mm_srli_epi32(tmp, 9);
  _mm_storeu_si128(compressed + 3, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 8)), 8));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 9));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 25));
  w1 = _mm_srli_epi32(tmp, 7);
  _mm_storeu_si128(compressed + 4, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 10)), 10));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 11));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 27));
  w0 = _mm_srli_epi32(tmp, 5);
  _mm_storeu_si128(compressed + 5, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 12)), 12));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 13));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 29));
  w1 = _mm_srli_epi32(tmp, 3);
  _mm_storeu_si128(compressed + 6, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 14)), 14));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 15));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 31));
  w0 = _mm_srli_epi32(tmp, 1);
  _mm_storeu_si128(compressed + 8, w0);
}

/* we are going to pack 64 18-bit values, touching 9 128-bit words, using 144
 * bytes */
static void simdpackblockmask18(const u32 *pin, __m128i *compressed) {
  /* we are going to touch  9 128-bit words */
  __m128i w0, w1;
  const __m128i *in = (const __m128i *)pin;
  const __m128i mask = _mm_set1_epi32(262143);
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_and_si128(mask, _mm_loadu_si128(in + 0));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 1));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 18));
  w1 = _mm_srli_epi32(tmp, 14);
  _mm_storeu_si128(compressed + 0, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 2)), 4));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 3));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 22));
  w0 = _mm_srli_epi32(tmp, 10);
  _mm_storeu_si128(compressed + 1, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 4)), 8));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 5));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 26));
  w1 = _mm_srli_epi32(tmp, 6);
  _mm_storeu_si128(compressed + 2, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 6)), 12));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 7));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 30));
  w0 = _mm_srli_epi32(tmp, 2);
  _mm_storeu_si128(compressed + 3, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 8));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 16));
  w1 = _mm_srli_epi32(tmp, 16);
  _mm_storeu_si128(compressed + 4, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 9)), 2));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 10));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 20));
  w0 = _mm_srli_epi32(tmp, 12);
  _mm_storeu_si128(compressed + 5, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 11)), 6));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 12));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 24));
  w1 = _mm_srli_epi32(tmp, 8);
  _mm_storeu_si128(compressed + 6, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 13)), 10));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 14));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 28));
  w0 = _mm_srli_epi32(tmp, 4);
  _mm_storeu_si128(compressed + 7, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 15)), 14));
  _mm_storeu_si128(compressed + 8, w0);
}

/* we are going to pack 64 19-bit values, touching 10 128-bit words, using 160
 * bytes */
static void simdpackblockmask19(const u32 *pin, __m128i *compressed) {
  /* we are going to touch  10 128-bit words */
  __m128i w0, w1;
  const __m128i *in = (const __m128i *)pin;
  const __m128i mask = _mm_set1_epi32(524287);
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_and_si128(mask, _mm_loadu_si128(in + 0));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 1));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 19));
  w1 = _mm_srli_epi32(tmp, 13);
  _mm_storeu_si128(compressed + 0, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 2)), 6));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 3));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 25));
  w0 = _mm_srli_epi32(tmp, 7);
  _mm_storeu_si128(compressed + 1, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 4)), 12));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 5));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 31));
  w1 = _mm_srli_epi32(tmp, 1);
  _mm_storeu_si128(compressed + 2, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 6));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 18));
  w0 = _mm_srli_epi32(tmp, 14);
  _mm_storeu_si128(compressed + 3, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 7)), 5));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 8));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 24));
  w1 = _mm_srli_epi32(tmp, 8);
  _mm_storeu_si128(compressed + 4, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 9)), 11));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 10));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 30));
  w0 = _mm_srli_epi32(tmp, 2);
  _mm_storeu_si128(compressed + 5, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 11));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 17));
  w1 = _mm_srli_epi32(tmp, 15);
  _mm_storeu_si128(compressed + 6, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 12)), 4));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 13));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 23));
  w0 = _mm_srli_epi32(tmp, 9);
  _mm_storeu_si128(compressed + 7, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 14)), 10));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 15));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 29));
  w1 = _mm_srli_epi32(tmp, 3);
  _mm_storeu_si128(compressed + 9, w1);
}

/* we are going to pack 64 20-bit values, touching 10 128-bit words, using 160
 * bytes */
static void simdpackblockmask20(const u32 *pin, __m128i *compressed) {
  /* we are going to touch  10 128-bit words */
  __m128i w0, w1;
  const __m128i *in = (const __m128i *)pin;
  const __m128i mask = _mm_set1_epi32(1048575);
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_and_si128(mask, _mm_loadu_si128(in + 0));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 1));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 20));
  w1 = _mm_srli_epi32(tmp, 12);
  _mm_storeu_si128(compressed + 0, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 2)), 8));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 3));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 28));
  w0 = _mm_srli_epi32(tmp, 4);
  _mm_storeu_si128(compressed + 1, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 4));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 16));
  w1 = _mm_srli_epi32(tmp, 16);
  _mm_storeu_si128(compressed + 2, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 5)), 4));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 6));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 24));
  w0 = _mm_srli_epi32(tmp, 8);
  _mm_storeu_si128(compressed + 3, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 7)), 12));
  _mm_storeu_si128(compressed + 4, w0);
  w1 = _mm_and_si128(mask, _mm_loadu_si128(in + 8));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 9));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 20));
  w0 = _mm_srli_epi32(tmp, 12);
  _mm_storeu_si128(compressed + 5, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 10)), 8));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 11));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 28));
  w1 = _mm_srli_epi32(tmp, 4);
  _mm_storeu_si128(compressed + 6, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 12));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 16));
  w0 = _mm_srli_epi32(tmp, 16);
  _mm_storeu_si128(compressed + 7, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 13)), 4));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 14));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 24));
  w1 = _mm_srli_epi32(tmp, 8);
  _mm_storeu_si128(compressed + 8, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 15)), 12));
  _mm_storeu_si128(compressed + 9, w1);
}

/* we are going to pack 64 21-bit values, touching 11 128-bit words, using 176
 * bytes */
static void simdpackblockmask21(const u32 *pin, __m128i *compressed) {
  /* we are going to touch  11 128-bit words */
  __m128i w0, w1;
  const __m128i *in = (const __m128i *)pin;
  const __m128i mask = _mm_set1_epi32(2097151);
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_and_si128(mask, _mm_loadu_si128(in + 0));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 1));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 21));
  w1 = _mm_srli_epi32(tmp, 11);
  _mm_storeu_si128(compressed + 0, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 2)), 10));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 3));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 31));
  w0 = _mm_srli_epi32(tmp, 1);
  _mm_storeu_si128(compressed + 1, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 4));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 20));
  w1 = _mm_srli_epi32(tmp, 12);
  _mm_storeu_si128(compressed + 2, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 5)), 9));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 6));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 30));
  w0 = _mm_srli_epi32(tmp, 2);
  _mm_storeu_si128(compressed + 3, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 7));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 19));
  w1 = _mm_srli_epi32(tmp, 13);
  _mm_storeu_si128(compressed + 4, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 8)), 8));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 9));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 29));
  w0 = _mm_srli_epi32(tmp, 3);
  _mm_storeu_si128(compressed + 5, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 10));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 18));
  w1 = _mm_srli_epi32(tmp, 14);
  _mm_storeu_si128(compressed + 6, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 11)), 7));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 12));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 28));
  w0 = _mm_srli_epi32(tmp, 4);
  _mm_storeu_si128(compressed + 7, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 13));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 17));
  w1 = _mm_srli_epi32(tmp, 15);
  _mm_storeu_si128(compressed + 8, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 14)), 6));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 15));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 27));
  w0 = _mm_srli_epi32(tmp, 5);
  _mm_storeu_si128(compressed + 10, w0);
}

/* we are going to pack 64 22-bit values, touching 11 128-bit words, using 176
 * bytes */
static void simdpackblockmask22(const u32 *pin, __m128i *compressed) {
  /* we are going to touch  11 128-bit words */
  __m128i w0, w1;
  const __m128i *in = (const __m128i *)pin;
  const __m128i mask = _mm_set1_epi32(4194303);
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_and_si128(mask, _mm_loadu_si128(in + 0));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 1));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 22));
  w1 = _mm_srli_epi32(tmp, 10);
  _mm_storeu_si128(compressed + 0, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 2));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 12));
  w0 = _mm_srli_epi32(tmp, 20);
  _mm_storeu_si128(compressed + 1, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 3)), 2));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 4));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 24));
  w1 = _mm_srli_epi32(tmp, 8);
  _mm_storeu_si128(compressed + 2, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 5));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 14));
  w0 = _mm_srli_epi32(tmp, 18);
  _mm_storeu_si128(compressed + 3, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 6)), 4));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 7));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 26));
  w1 = _mm_srli_epi32(tmp, 6);
  _mm_storeu_si128(compressed + 4, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 8));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 16));
  w0 = _mm_srli_epi32(tmp, 16);
  _mm_storeu_si128(compressed + 5, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 9)), 6));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 10));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 28));
  w1 = _mm_srli_epi32(tmp, 4);
  _mm_storeu_si128(compressed + 6, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 11));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 18));
  w0 = _mm_srli_epi32(tmp, 14);
  _mm_storeu_si128(compressed + 7, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 12)), 8));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 13));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 30));
  w1 = _mm_srli_epi32(tmp, 2);
  _mm_storeu_si128(compressed + 8, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 14));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 20));
  w0 = _mm_srli_epi32(tmp, 12);
  _mm_storeu_si128(compressed + 9, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 15)), 10));
  _mm_storeu_si128(compressed + 10, w0);
}

/* we are going to pack 64 23-bit values, touching 12 128-bit words, using 192
 * bytes */
static void simdpackblockmask23(const u32 *pin, __m128i *compressed) {
  /* we are going to touch  12 128-bit words */
  __m128i w0, w1;
  const __m128i *in = (const __m128i *)pin;
  const __m128i mask = _mm_set1_epi32(8388607);
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_and_si128(mask, _mm_loadu_si128(in + 0));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 1));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 23));
  w1 = _mm_srli_epi32(tmp, 9);
  _mm_storeu_si128(compressed + 0, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 2));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 14));
  w0 = _mm_srli_epi32(tmp, 18);
  _mm_storeu_si128(compressed + 1, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 3)), 5));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 4));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 28));
  w1 = _mm_srli_epi32(tmp, 4);
  _mm_storeu_si128(compressed + 2, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 5));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 19));
  w0 = _mm_srli_epi32(tmp, 13);
  _mm_storeu_si128(compressed + 3, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 6));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 10));
  w1 = _mm_srli_epi32(tmp, 22);
  _mm_storeu_si128(compressed + 4, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 7)), 1));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 8));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 24));
  w0 = _mm_srli_epi32(tmp, 8);
  _mm_storeu_si128(compressed + 5, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 9));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 15));
  w1 = _mm_srli_epi32(tmp, 17);
  _mm_storeu_si128(compressed + 6, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 10)), 6));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 11));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 29));
  w0 = _mm_srli_epi32(tmp, 3);
  _mm_storeu_si128(compressed + 7, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 12));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 20));
  w1 = _mm_srli_epi32(tmp, 12);
  _mm_storeu_si128(compressed + 8, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 13));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 11));
  w0 = _mm_srli_epi32(tmp, 21);
  _mm_storeu_si128(compressed + 9, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 14)), 2));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 15));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 25));
  w1 = _mm_srli_epi32(tmp, 7);
  _mm_storeu_si128(compressed + 11, w1);
}

/* we are going to pack 64 24-bit values, touching 12 128-bit words, using 192
 * bytes */
static void simdpackblockmask24(const u32 *pin, __m128i *compressed) {
  /* we are going to touch  12 128-bit words */
  __m128i w0, w1;
  const __m128i *in = (const __m128i *)pin;
  const __m128i mask = _mm_set1_epi32(16777215);
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_and_si128(mask, _mm_loadu_si128(in + 0));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 1));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 24));
  w1 = _mm_srli_epi32(tmp, 8);
  _mm_storeu_si128(compressed + 0, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 2));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 16));
  w0 = _mm_srli_epi32(tmp, 16);
  _mm_storeu_si128(compressed + 1, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 3)), 8));
  _mm_storeu_si128(compressed + 2, w0);
  w1 = _mm_and_si128(mask, _mm_loadu_si128(in + 4));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 5));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 24));
  w0 = _mm_srli_epi32(tmp, 8);
  _mm_storeu_si128(compressed + 3, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 6));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 16));
  w1 = _mm_srli_epi32(tmp, 16);
  _mm_storeu_si128(compressed + 4, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 7)), 8));
  _mm_storeu_si128(compressed + 5, w1);
  w0 = _mm_and_si128(mask, _mm_loadu_si128(in + 8));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 9));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 24));
  w1 = _mm_srli_epi32(tmp, 8);
  _mm_storeu_si128(compressed + 6, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 10));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 16));
  w0 = _mm_srli_epi32(tmp, 16);
  _mm_storeu_si128(compressed + 7, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 11)), 8));
  _mm_storeu_si128(compressed + 8, w0);
  w1 = _mm_and_si128(mask, _mm_loadu_si128(in + 12));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 13));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 24));
  w0 = _mm_srli_epi32(tmp, 8);
  _mm_storeu_si128(compressed + 9, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 14));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 16));
  w1 = _mm_srli_epi32(tmp, 16);
  _mm_storeu_si128(compressed + 10, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 15)), 8));
  _mm_storeu_si128(compressed + 11, w1);
}

/* we are going to pack 64 25-bit values, touching 13 128-bit words, using 208
 * bytes */
static void simdpackblockmask25(const u32 *pin, __m128i *compressed) {
  /* we are going to touch  13 128-bit words */
  __m128i w0, w1;
  const __m128i *in = (const __m128i *)pin;
  const __m128i mask = _mm_set1_epi32(33554431);
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_and_si128(mask, _mm_loadu_si128(in + 0));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 1));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 25));
  w1 = _mm_srli_epi32(tmp, 7);
  _mm_storeu_si128(compressed + 0, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 2));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 18));
  w0 = _mm_srli_epi32(tmp, 14);
  _mm_storeu_si128(compressed + 1, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 3));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 11));
  w1 = _mm_srli_epi32(tmp, 21);
  _mm_storeu_si128(compressed + 2, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 4)), 4));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 5));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 29));
  w0 = _mm_srli_epi32(tmp, 3);
  _mm_storeu_si128(compressed + 3, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 6));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 22));
  w1 = _mm_srli_epi32(tmp, 10);
  _mm_storeu_si128(compressed + 4, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 7));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 15));
  w0 = _mm_srli_epi32(tmp, 17);
  _mm_storeu_si128(compressed + 5, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 8));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 8));
  w1 = _mm_srli_epi32(tmp, 24);
  _mm_storeu_si128(compressed + 6, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 9)), 1));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 10));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 26));
  w0 = _mm_srli_epi32(tmp, 6);
  _mm_storeu_si128(compressed + 7, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 11));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 19));
  w1 = _mm_srli_epi32(tmp, 13);
  _mm_storeu_si128(compressed + 8, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 12));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 12));
  w0 = _mm_srli_epi32(tmp, 20);
  _mm_storeu_si128(compressed + 9, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 13)), 5));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 14));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 30));
  w1 = _mm_srli_epi32(tmp, 2);
  _mm_storeu_si128(compressed + 10, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 15));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 23));
  w0 = _mm_srli_epi32(tmp, 9);
  _mm_storeu_si128(compressed + 12, w0);
}

/* we are going to pack 64 26-bit values, touching 13 128-bit words, using 208
 * bytes */
static void simdpackblockmask26(const u32 *pin, __m128i *compressed) {
  /* we are going to touch  13 128-bit words */
  __m128i w0, w1;
  const __m128i *in = (const __m128i *)pin;
  const __m128i mask = _mm_set1_epi32(67108863);
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_and_si128(mask, _mm_loadu_si128(in + 0));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 1));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 26));
  w1 = _mm_srli_epi32(tmp, 6);
  _mm_storeu_si128(compressed + 0, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 2));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 20));
  w0 = _mm_srli_epi32(tmp, 12);
  _mm_storeu_si128(compressed + 1, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 3));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 14));
  w1 = _mm_srli_epi32(tmp, 18);
  _mm_storeu_si128(compressed + 2, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 4));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 8));
  w0 = _mm_srli_epi32(tmp, 24);
  _mm_storeu_si128(compressed + 3, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 5)), 2));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 6));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 28));
  w1 = _mm_srli_epi32(tmp, 4);
  _mm_storeu_si128(compressed + 4, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 7));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 22));
  w0 = _mm_srli_epi32(tmp, 10);
  _mm_storeu_si128(compressed + 5, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 8));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 16));
  w1 = _mm_srli_epi32(tmp, 16);
  _mm_storeu_si128(compressed + 6, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 9));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 10));
  w0 = _mm_srli_epi32(tmp, 22);
  _mm_storeu_si128(compressed + 7, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 10)), 4));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 11));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 30));
  w1 = _mm_srli_epi32(tmp, 2);
  _mm_storeu_si128(compressed + 8, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 12));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 24));
  w0 = _mm_srli_epi32(tmp, 8);
  _mm_storeu_si128(compressed + 9, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 13));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 18));
  w1 = _mm_srli_epi32(tmp, 14);
  _mm_storeu_si128(compressed + 10, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 14));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 12));
  w0 = _mm_srli_epi32(tmp, 20);
  _mm_storeu_si128(compressed + 11, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 15)), 6));
  _mm_storeu_si128(compressed + 12, w0);
}

/* we are going to pack 64 27-bit values, touching 14 128-bit words, using 224
 * bytes */
static void simdpackblockmask27(const u32 *pin, __m128i *compressed) {
  /* we are going to touch  14 128-bit words */
  __m128i w0, w1;
  const __m128i *in = (const __m128i *)pin;
  const __m128i mask = _mm_set1_epi32(134217727);
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_and_si128(mask, _mm_loadu_si128(in + 0));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 1));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 27));
  w1 = _mm_srli_epi32(tmp, 5);
  _mm_storeu_si128(compressed + 0, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 2));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 22));
  w0 = _mm_srli_epi32(tmp, 10);
  _mm_storeu_si128(compressed + 1, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 3));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 17));
  w1 = _mm_srli_epi32(tmp, 15);
  _mm_storeu_si128(compressed + 2, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 4));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 12));
  w0 = _mm_srli_epi32(tmp, 20);
  _mm_storeu_si128(compressed + 3, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 5));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 7));
  w1 = _mm_srli_epi32(tmp, 25);
  _mm_storeu_si128(compressed + 4, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 6)), 2));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 7));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 29));
  w0 = _mm_srli_epi32(tmp, 3);
  _mm_storeu_si128(compressed + 5, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 8));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 24));
  w1 = _mm_srli_epi32(tmp, 8);
  _mm_storeu_si128(compressed + 6, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 9));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 19));
  w0 = _mm_srli_epi32(tmp, 13);
  _mm_storeu_si128(compressed + 7, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 10));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 14));
  w1 = _mm_srli_epi32(tmp, 18);
  _mm_storeu_si128(compressed + 8, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 11));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 9));
  w0 = _mm_srli_epi32(tmp, 23);
  _mm_storeu_si128(compressed + 9, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 12)), 4));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 13));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 31));
  w1 = _mm_srli_epi32(tmp, 1);
  _mm_storeu_si128(compressed + 10, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 14));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 26));
  w0 = _mm_srli_epi32(tmp, 6);
  _mm_storeu_si128(compressed + 11, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 15));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 21));
  w1 = _mm_srli_epi32(tmp, 11);
  _mm_storeu_si128(compressed + 13, w1);
}

/* we are going to pack 64 28-bit values, touching 14 128-bit words, using 224
 * bytes */
static void simdpackblockmask28(const u32 *pin, __m128i *compressed) {
  /* we are going to touch  14 128-bit words */
  __m128i w0, w1;
  const __m128i *in = (const __m128i *)pin;
  const __m128i mask = _mm_set1_epi32(268435455);
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_and_si128(mask, _mm_loadu_si128(in + 0));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 1));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 28));
  w1 = _mm_srli_epi32(tmp, 4);
  _mm_storeu_si128(compressed + 0, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 2));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 24));
  w0 = _mm_srli_epi32(tmp, 8);
  _mm_storeu_si128(compressed + 1, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 3));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 20));
  w1 = _mm_srli_epi32(tmp, 12);
  _mm_storeu_si128(compressed + 2, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 4));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 16));
  w0 = _mm_srli_epi32(tmp, 16);
  _mm_storeu_si128(compressed + 3, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 5));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 12));
  w1 = _mm_srli_epi32(tmp, 20);
  _mm_storeu_si128(compressed + 4, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 6));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 8));
  w0 = _mm_srli_epi32(tmp, 24);
  _mm_storeu_si128(compressed + 5, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 7)), 4));
  _mm_storeu_si128(compressed + 6, w0);
  w1 = _mm_and_si128(mask, _mm_loadu_si128(in + 8));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 9));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 28));
  w0 = _mm_srli_epi32(tmp, 4);
  _mm_storeu_si128(compressed + 7, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 10));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 24));
  w1 = _mm_srli_epi32(tmp, 8);
  _mm_storeu_si128(compressed + 8, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 11));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 20));
  w0 = _mm_srli_epi32(tmp, 12);
  _mm_storeu_si128(compressed + 9, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 12));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 16));
  w1 = _mm_srli_epi32(tmp, 16);
  _mm_storeu_si128(compressed + 10, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 13));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 12));
  w0 = _mm_srli_epi32(tmp, 20);
  _mm_storeu_si128(compressed + 11, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 14));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 8));
  w1 = _mm_srli_epi32(tmp, 24);
  _mm_storeu_si128(compressed + 12, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 15)), 4));
  _mm_storeu_si128(compressed + 13, w1);
}

/* we are going to pack 64 29-bit values, touching 15 128-bit words, using 240
 * bytes */
static void simdpackblockmask29(const u32 *pin, __m128i *compressed) {
  /* we are going to touch  15 128-bit words */
  __m128i w0, w1;
  const __m128i *in = (const __m128i *)pin;
  const __m128i mask = _mm_set1_epi32(536870911);
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_and_si128(mask, _mm_loadu_si128(in + 0));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 1));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 29));
  w1 = _mm_srli_epi32(tmp, 3);
  _mm_storeu_si128(compressed + 0, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 2));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 26));
  w0 = _mm_srli_epi32(tmp, 6);
  _mm_storeu_si128(compressed + 1, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 3));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 23));
  w1 = _mm_srli_epi32(tmp, 9);
  _mm_storeu_si128(compressed + 2, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 4));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 20));
  w0 = _mm_srli_epi32(tmp, 12);
  _mm_storeu_si128(compressed + 3, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 5));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 17));
  w1 = _mm_srli_epi32(tmp, 15);
  _mm_storeu_si128(compressed + 4, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 6));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 14));
  w0 = _mm_srli_epi32(tmp, 18);
  _mm_storeu_si128(compressed + 5, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 7));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 11));
  w1 = _mm_srli_epi32(tmp, 21);
  _mm_storeu_si128(compressed + 6, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 8));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 8));
  w0 = _mm_srli_epi32(tmp, 24);
  _mm_storeu_si128(compressed + 7, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 9));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 5));
  w1 = _mm_srli_epi32(tmp, 27);
  _mm_storeu_si128(compressed + 8, w0);
  w1 = _mm_or_si128(
      w1, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 10)), 2));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 11));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 31));
  w0 = _mm_srli_epi32(tmp, 1);
  _mm_storeu_si128(compressed + 9, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 12));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 28));
  w1 = _mm_srli_epi32(tmp, 4);
  _mm_storeu_si128(compressed + 10, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 13));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 25));
  w0 = _mm_srli_epi32(tmp, 7);
  _mm_storeu_si128(compressed + 11, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 14));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 22));
  w1 = _mm_srli_epi32(tmp, 10);
  _mm_storeu_si128(compressed + 12, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 15));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 19));
  w0 = _mm_srli_epi32(tmp, 13);
  _mm_storeu_si128(compressed + 14, w0);
}

/* we are going to pack 64 30-bit values, touching 15 128-bit words, using 240
 * bytes */
static void simdpackblockmask30(const u32 *pin, __m128i *compressed) {
  /* we are going to touch  15 128-bit words */
  __m128i w0, w1;
  const __m128i *in = (const __m128i *)pin;
  const __m128i mask = _mm_set1_epi32(1073741823);
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_and_si128(mask, _mm_loadu_si128(in + 0));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 1));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 30));
  w1 = _mm_srli_epi32(tmp, 2);
  _mm_storeu_si128(compressed + 0, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 2));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 28));
  w0 = _mm_srli_epi32(tmp, 4);
  _mm_storeu_si128(compressed + 1, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 3));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 26));
  w1 = _mm_srli_epi32(tmp, 6);
  _mm_storeu_si128(compressed + 2, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 4));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 24));
  w0 = _mm_srli_epi32(tmp, 8);
  _mm_storeu_si128(compressed + 3, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 5));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 22));
  w1 = _mm_srli_epi32(tmp, 10);
  _mm_storeu_si128(compressed + 4, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 6));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 20));
  w0 = _mm_srli_epi32(tmp, 12);
  _mm_storeu_si128(compressed + 5, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 7));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 18));
  w1 = _mm_srli_epi32(tmp, 14);
  _mm_storeu_si128(compressed + 6, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 8));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 16));
  w0 = _mm_srli_epi32(tmp, 16);
  _mm_storeu_si128(compressed + 7, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 9));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 14));
  w1 = _mm_srli_epi32(tmp, 18);
  _mm_storeu_si128(compressed + 8, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 10));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 12));
  w0 = _mm_srli_epi32(tmp, 20);
  _mm_storeu_si128(compressed + 9, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 11));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 10));
  w1 = _mm_srli_epi32(tmp, 22);
  _mm_storeu_si128(compressed + 10, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 12));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 8));
  w0 = _mm_srli_epi32(tmp, 24);
  _mm_storeu_si128(compressed + 11, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 13));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 6));
  w1 = _mm_srli_epi32(tmp, 26);
  _mm_storeu_si128(compressed + 12, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 14));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 4));
  w0 = _mm_srli_epi32(tmp, 28);
  _mm_storeu_si128(compressed + 13, w1);
  w0 = _mm_or_si128(
      w0, _mm_slli_epi32(_mm_and_si128(mask, _mm_loadu_si128(in + 15)), 2));
  _mm_storeu_si128(compressed + 14, w0);
}

/* we are going to pack 64 31-bit values, touching 16 128-bit words, using 256
 * bytes */
static void simdpackblockmask31(const u32 *pin, __m128i *compressed) {
  /* we are going to touch  16 128-bit words */
  __m128i w0, w1;
  const __m128i *in = (const __m128i *)pin;
  const __m128i mask = _mm_set1_epi32(2147483647);
  __m128i tmp; /* used to store inputs at word boundary */
  w0 = _mm_and_si128(mask, _mm_loadu_si128(in + 0));
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 1));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 31));
  w1 = _mm_srli_epi32(tmp, 1);
  _mm_storeu_si128(compressed + 0, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 2));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 30));
  w0 = _mm_srli_epi32(tmp, 2);
  _mm_storeu_si128(compressed + 1, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 3));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 29));
  w1 = _mm_srli_epi32(tmp, 3);
  _mm_storeu_si128(compressed + 2, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 4));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 28));
  w0 = _mm_srli_epi32(tmp, 4);
  _mm_storeu_si128(compressed + 3, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 5));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 27));
  w1 = _mm_srli_epi32(tmp, 5);
  _mm_storeu_si128(compressed + 4, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 6));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 26));
  w0 = _mm_srli_epi32(tmp, 6);
  _mm_storeu_si128(compressed + 5, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 7));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 25));
  w1 = _mm_srli_epi32(tmp, 7);
  _mm_storeu_si128(compressed + 6, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 8));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 24));
  w0 = _mm_srli_epi32(tmp, 8);
  _mm_storeu_si128(compressed + 7, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 9));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 23));
  w1 = _mm_srli_epi32(tmp, 9);
  _mm_storeu_si128(compressed + 8, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 10));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 22));
  w0 = _mm_srli_epi32(tmp, 10);
  _mm_storeu_si128(compressed + 9, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 11));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 21));
  w1 = _mm_srli_epi32(tmp, 11);
  _mm_storeu_si128(compressed + 10, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 12));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 20));
  w0 = _mm_srli_epi32(tmp, 12);
  _mm_storeu_si128(compressed + 11, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 13));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 19));
  w1 = _mm_srli_epi32(tmp, 13);
  _mm_storeu_si128(compressed + 12, w0);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 14));
  w1 = _mm_or_si128(w1, _mm_slli_epi32(tmp, 18));
  w0 = _mm_srli_epi32(tmp, 14);
  _mm_storeu_si128(compressed + 13, w1);
  tmp = _mm_and_si128(mask, _mm_loadu_si128(in + 15));
  w0 = _mm_or_si128(w0, _mm_slli_epi32(tmp, 17));
  w1 = _mm_srli_epi32(tmp, 15);
  _mm_storeu_si128(compressed + 15, w1);
}

/* we are going to pack 64 32-bit values, touching 16 128-bit words, using 256
 * bytes */
static void simdpackblockmask32(const u32 *pin, __m128i *compressed) {
  /* we are going to touch  16 128-bit words */
  __m128i w0, w1;
  const __m128i *in = (const __m128i *)pin;
  w0 = _mm_loadu_si128(in + 0);
  _mm_storeu_si128(compressed + 0, w0);
  w1 = _mm_loadu_si128(in + 1);
  _mm_storeu_si128(compressed + 1, w1);
  w0 = _mm_loadu_si128(in + 2);
  _mm_storeu_si128(compressed + 2, w0);
  w1 = _mm_loadu_si128(in + 3);
  _mm_storeu_si128(compressed + 3, w1);
  w0 = _mm_loadu_si128(in + 4);
  _mm_storeu_si128(compressed + 4, w0);
  w1 = _mm_loadu_si128(in + 5);
  _mm_storeu_si128(compressed + 5, w1);
  w0 = _mm_loadu_si128(in + 6);
  _mm_storeu_si128(compressed + 6, w0);
  w1 = _mm_loadu_si128(in + 7);
  _mm_storeu_si128(compressed + 7, w1);
  w0 = _mm_loadu_si128(in + 8);
  _mm_storeu_si128(compressed + 8, w0);
  w1 = _mm_loadu_si128(in + 9);
  _mm_storeu_si128(compressed + 9, w1);
  w0 = _mm_loadu_si128(in + 10);
  _mm_storeu_si128(compressed + 10, w0);
  w1 = _mm_loadu_si128(in + 11);
  _mm_storeu_si128(compressed + 11, w1);
  w0 = _mm_loadu_si128(in + 12);
  _mm_storeu_si128(compressed + 12, w0);
  w1 = _mm_loadu_si128(in + 13);
  _mm_storeu_si128(compressed + 13, w1);
  w0 = _mm_loadu_si128(in + 14);
  _mm_storeu_si128(compressed + 14, w0);
  w1 = _mm_loadu_si128(in + 15);
  _mm_storeu_si128(compressed + 15, w1);
}

static void simdunpackblock0(const __m128i *compressed, u32 *pout) {
  (void)compressed;
  memset(pout, 0, 64);
}

/* we packed 64 1-bit values, touching 1 128-bit words, using 16 bytes */
static void simdunpackblock1(const __m128i *compressed, u32 *pout) {
  /* we are going to access  1 128-bit word */
  __m128i w0;
  __m128i *out = (__m128i *)pout;
  const __m128i mask = _mm_set1_epi32(1);
  w0 = _mm_loadu_si128(compressed);
  _mm_storeu_si128(out + 0, _mm_and_si128(mask, w0));
  _mm_storeu_si128(out + 1, _mm_and_si128(mask, _mm_srli_epi32(w0, 1)));
  _mm_storeu_si128(out + 2, _mm_and_si128(mask, _mm_srli_epi32(w0, 2)));
  _mm_storeu_si128(out + 3, _mm_and_si128(mask, _mm_srli_epi32(w0, 3)));
  _mm_storeu_si128(out + 4, _mm_and_si128(mask, _mm_srli_epi32(w0, 4)));
  _mm_storeu_si128(out + 5, _mm_and_si128(mask, _mm_srli_epi32(w0, 5)));
  _mm_storeu_si128(out + 6, _mm_and_si128(mask, _mm_srli_epi32(w0, 6)));
  _mm_storeu_si128(out + 7, _mm_and_si128(mask, _mm_srli_epi32(w0, 7)));
  _mm_storeu_si128(out + 8, _mm_and_si128(mask, _mm_srli_epi32(w0, 8)));
  _mm_storeu_si128(out + 9, _mm_and_si128(mask, _mm_srli_epi32(w0, 9)));
  _mm_storeu_si128(out + 10, _mm_and_si128(mask, _mm_srli_epi32(w0, 10)));
  _mm_storeu_si128(out + 11, _mm_and_si128(mask, _mm_srli_epi32(w0, 11)));
  _mm_storeu_si128(out + 12, _mm_and_si128(mask, _mm_srli_epi32(w0, 12)));
  _mm_storeu_si128(out + 13, _mm_and_si128(mask, _mm_srli_epi32(w0, 13)));
  _mm_storeu_si128(out + 14, _mm_and_si128(mask, _mm_srli_epi32(w0, 14)));
  _mm_storeu_si128(out + 15, _mm_and_si128(mask, _mm_srli_epi32(w0, 15)));
}

/* we packed 64 2-bit values, touching 1 128-bit words, using 16 bytes */
static void simdunpackblock2(const __m128i *compressed, u32 *pout) {
  /* we are going to access  1 128-bit word */
  __m128i w0;
  __m128i *out = (__m128i *)pout;
  const __m128i mask = _mm_set1_epi32(3);
  w0 = _mm_loadu_si128(compressed);
  _mm_storeu_si128(out + 0, _mm_and_si128(mask, w0));
  _mm_storeu_si128(out + 1, _mm_and_si128(mask, _mm_srli_epi32(w0, 2)));
  _mm_storeu_si128(out + 2, _mm_and_si128(mask, _mm_srli_epi32(w0, 4)));
  _mm_storeu_si128(out + 3, _mm_and_si128(mask, _mm_srli_epi32(w0, 6)));
  _mm_storeu_si128(out + 4, _mm_and_si128(mask, _mm_srli_epi32(w0, 8)));
  _mm_storeu_si128(out + 5, _mm_and_si128(mask, _mm_srli_epi32(w0, 10)));
  _mm_storeu_si128(out + 6, _mm_and_si128(mask, _mm_srli_epi32(w0, 12)));
  _mm_storeu_si128(out + 7, _mm_and_si128(mask, _mm_srli_epi32(w0, 14)));
  _mm_storeu_si128(out + 8, _mm_and_si128(mask, _mm_srli_epi32(w0, 16)));
  _mm_storeu_si128(out + 9, _mm_and_si128(mask, _mm_srli_epi32(w0, 18)));
  _mm_storeu_si128(out + 10, _mm_and_si128(mask, _mm_srli_epi32(w0, 20)));
  _mm_storeu_si128(out + 11, _mm_and_si128(mask, _mm_srli_epi32(w0, 22)));
  _mm_storeu_si128(out + 12, _mm_and_si128(mask, _mm_srli_epi32(w0, 24)));
  _mm_storeu_si128(out + 13, _mm_and_si128(mask, _mm_srli_epi32(w0, 26)));
  _mm_storeu_si128(out + 14, _mm_and_si128(mask, _mm_srli_epi32(w0, 28)));
  _mm_storeu_si128(out + 15, _mm_srli_epi32(w0, 30));
}

/* we packed 64 3-bit values, touching 2 128-bit words, using 32 bytes */
static void simdunpackblock3(const __m128i *compressed, u32 *pout) {
  /* we are going to access  2 128-bit words */
  __m128i w0, w1;
  __m128i *out = (__m128i *)pout;
  const __m128i mask = _mm_set1_epi32(7);
  w0 = _mm_loadu_si128(compressed);
  _mm_storeu_si128(out + 0, _mm_and_si128(mask, w0));
  _mm_storeu_si128(out + 1, _mm_and_si128(mask, _mm_srli_epi32(w0, 3)));
  _mm_storeu_si128(out + 2, _mm_and_si128(mask, _mm_srli_epi32(w0, 6)));
  _mm_storeu_si128(out + 3, _mm_and_si128(mask, _mm_srli_epi32(w0, 9)));
  _mm_storeu_si128(out + 4, _mm_and_si128(mask, _mm_srli_epi32(w0, 12)));
  _mm_storeu_si128(out + 5, _mm_and_si128(mask, _mm_srli_epi32(w0, 15)));
  _mm_storeu_si128(out + 6, _mm_and_si128(mask, _mm_srli_epi32(w0, 18)));
  _mm_storeu_si128(out + 7, _mm_and_si128(mask, _mm_srli_epi32(w0, 21)));
  _mm_storeu_si128(out + 8, _mm_and_si128(mask, _mm_srli_epi32(w0, 24)));
  _mm_storeu_si128(out + 9, _mm_and_si128(mask, _mm_srli_epi32(w0, 27)));
  w1 = _mm_loadu_si128(compressed + 1);
  _mm_storeu_si128(out + 10,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 30),
                                                    _mm_slli_epi32(w1, 2))));
  _mm_storeu_si128(out + 11, _mm_and_si128(mask, _mm_srli_epi32(w1, 1)));
  _mm_storeu_si128(out + 12, _mm_and_si128(mask, _mm_srli_epi32(w1, 4)));
  _mm_storeu_si128(out + 13, _mm_and_si128(mask, _mm_srli_epi32(w1, 7)));
  _mm_storeu_si128(out + 14, _mm_and_si128(mask, _mm_srli_epi32(w1, 10)));
  _mm_storeu_si128(out + 15, _mm_and_si128(mask, _mm_srli_epi32(w1, 13)));
}

/* we packed 64 4-bit values, touching 2 128-bit words, using 32 bytes */
static void simdunpackblock4(const __m128i *compressed, u32 *pout) {
  /* we are going to access  2 128-bit words */
  __m128i w0, w1;
  __m128i *out = (__m128i *)pout;
  const __m128i mask = _mm_set1_epi32(15);
  w0 = _mm_loadu_si128(compressed);
  _mm_storeu_si128(out + 0, _mm_and_si128(mask, w0));
  _mm_storeu_si128(out + 1, _mm_and_si128(mask, _mm_srli_epi32(w0, 4)));
  _mm_storeu_si128(out + 2, _mm_and_si128(mask, _mm_srli_epi32(w0, 8)));
  _mm_storeu_si128(out + 3, _mm_and_si128(mask, _mm_srli_epi32(w0, 12)));
  _mm_storeu_si128(out + 4, _mm_and_si128(mask, _mm_srli_epi32(w0, 16)));
  _mm_storeu_si128(out + 5, _mm_and_si128(mask, _mm_srli_epi32(w0, 20)));
  _mm_storeu_si128(out + 6, _mm_and_si128(mask, _mm_srli_epi32(w0, 24)));
  _mm_storeu_si128(out + 7, _mm_srli_epi32(w0, 28));
  w1 = _mm_loadu_si128(compressed + 1);
  _mm_storeu_si128(out + 8, _mm_and_si128(mask, w1));
  _mm_storeu_si128(out + 9, _mm_and_si128(mask, _mm_srli_epi32(w1, 4)));
  _mm_storeu_si128(out + 10, _mm_and_si128(mask, _mm_srli_epi32(w1, 8)));
  _mm_storeu_si128(out + 11, _mm_and_si128(mask, _mm_srli_epi32(w1, 12)));
  _mm_storeu_si128(out + 12, _mm_and_si128(mask, _mm_srli_epi32(w1, 16)));
  _mm_storeu_si128(out + 13, _mm_and_si128(mask, _mm_srli_epi32(w1, 20)));
  _mm_storeu_si128(out + 14, _mm_and_si128(mask, _mm_srli_epi32(w1, 24)));
  _mm_storeu_si128(out + 15, _mm_srli_epi32(w1, 28));
}

/* we packed 64 5-bit values, touching 3 128-bit words, using 48 bytes */
static void simdunpackblock5(const __m128i *compressed, u32 *pout) {
  /* we are going to access  3 128-bit words */
  __m128i w0, w1;
  __m128i *out = (__m128i *)pout;
  const __m128i mask = _mm_set1_epi32(31);
  w0 = _mm_loadu_si128(compressed);
  _mm_storeu_si128(out + 0, _mm_and_si128(mask, w0));
  _mm_storeu_si128(out + 1, _mm_and_si128(mask, _mm_srli_epi32(w0, 5)));
  _mm_storeu_si128(out + 2, _mm_and_si128(mask, _mm_srli_epi32(w0, 10)));
  _mm_storeu_si128(out + 3, _mm_and_si128(mask, _mm_srli_epi32(w0, 15)));
  _mm_storeu_si128(out + 4, _mm_and_si128(mask, _mm_srli_epi32(w0, 20)));
  _mm_storeu_si128(out + 5, _mm_and_si128(mask, _mm_srli_epi32(w0, 25)));
  w1 = _mm_loadu_si128(compressed + 1);
  _mm_storeu_si128(out + 6,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 30),
                                                    _mm_slli_epi32(w1, 2))));
  _mm_storeu_si128(out + 7, _mm_and_si128(mask, _mm_srli_epi32(w1, 3)));
  _mm_storeu_si128(out + 8, _mm_and_si128(mask, _mm_srli_epi32(w1, 8)));
  _mm_storeu_si128(out + 9, _mm_and_si128(mask, _mm_srli_epi32(w1, 13)));
  _mm_storeu_si128(out + 10, _mm_and_si128(mask, _mm_srli_epi32(w1, 18)));
  _mm_storeu_si128(out + 11, _mm_and_si128(mask, _mm_srli_epi32(w1, 23)));
  w0 = _mm_loadu_si128(compressed + 2);
  _mm_storeu_si128(out + 12,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 28),
                                                    _mm_slli_epi32(w0, 4))));
  _mm_storeu_si128(out + 13, _mm_and_si128(mask, _mm_srli_epi32(w0, 1)));
  _mm_storeu_si128(out + 14, _mm_and_si128(mask, _mm_srli_epi32(w0, 6)));
  _mm_storeu_si128(out + 15, _mm_and_si128(mask, _mm_srli_epi32(w0, 11)));
}

/* we packed 64 6-bit values, touching 3 128-bit words, using 48 bytes */
static void simdunpackblock6(const __m128i *compressed, u32 *pout) {
  /* we are going to access  3 128-bit words */
  __m128i w0, w1;
  __m128i *out = (__m128i *)pout;
  const __m128i mask = _mm_set1_epi32(63);
  w0 = _mm_loadu_si128(compressed);
  _mm_storeu_si128(out + 0, _mm_and_si128(mask, w0));
  _mm_storeu_si128(out + 1, _mm_and_si128(mask, _mm_srli_epi32(w0, 6)));
  _mm_storeu_si128(out + 2, _mm_and_si128(mask, _mm_srli_epi32(w0, 12)));
  _mm_storeu_si128(out + 3, _mm_and_si128(mask, _mm_srli_epi32(w0, 18)));
  _mm_storeu_si128(out + 4, _mm_and_si128(mask, _mm_srli_epi32(w0, 24)));
  w1 = _mm_loadu_si128(compressed + 1);
  _mm_storeu_si128(out + 5,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 30),
                                                    _mm_slli_epi32(w1, 2))));
  _mm_storeu_si128(out + 6, _mm_and_si128(mask, _mm_srli_epi32(w1, 4)));
  _mm_storeu_si128(out + 7, _mm_and_si128(mask, _mm_srli_epi32(w1, 10)));
  _mm_storeu_si128(out + 8, _mm_and_si128(mask, _mm_srli_epi32(w1, 16)));
  _mm_storeu_si128(out + 9, _mm_and_si128(mask, _mm_srli_epi32(w1, 22)));
  w0 = _mm_loadu_si128(compressed + 2);
  _mm_storeu_si128(out + 10,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 28),
                                                    _mm_slli_epi32(w0, 4))));
  _mm_storeu_si128(out + 11, _mm_and_si128(mask, _mm_srli_epi32(w0, 2)));
  _mm_storeu_si128(out + 12, _mm_and_si128(mask, _mm_srli_epi32(w0, 8)));
  _mm_storeu_si128(out + 13, _mm_and_si128(mask, _mm_srli_epi32(w0, 14)));
  _mm_storeu_si128(out + 14, _mm_and_si128(mask, _mm_srli_epi32(w0, 20)));
  _mm_storeu_si128(out + 15, _mm_srli_epi32(w0, 26));
}

/* we packed 64 7-bit values, touching 4 128-bit words, using 64 bytes */
static void simdunpackblock7(const __m128i *compressed, u32 *pout) {
  /* we are going to access  4 128-bit words */
  __m128i w0, w1;
  __m128i *out = (__m128i *)pout;
  const __m128i mask = _mm_set1_epi32(127);
  w0 = _mm_loadu_si128(compressed);
  _mm_storeu_si128(out + 0, _mm_and_si128(mask, w0));
  _mm_storeu_si128(out + 1, _mm_and_si128(mask, _mm_srli_epi32(w0, 7)));
  _mm_storeu_si128(out + 2, _mm_and_si128(mask, _mm_srli_epi32(w0, 14)));
  _mm_storeu_si128(out + 3, _mm_and_si128(mask, _mm_srli_epi32(w0, 21)));
  w1 = _mm_loadu_si128(compressed + 1);
  _mm_storeu_si128(out + 4,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 28),
                                                    _mm_slli_epi32(w1, 4))));
  _mm_storeu_si128(out + 5, _mm_and_si128(mask, _mm_srli_epi32(w1, 3)));
  _mm_storeu_si128(out + 6, _mm_and_si128(mask, _mm_srli_epi32(w1, 10)));
  _mm_storeu_si128(out + 7, _mm_and_si128(mask, _mm_srli_epi32(w1, 17)));
  _mm_storeu_si128(out + 8, _mm_and_si128(mask, _mm_srli_epi32(w1, 24)));
  w0 = _mm_loadu_si128(compressed + 2);
  _mm_storeu_si128(out + 9,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 31),
                                                    _mm_slli_epi32(w0, 1))));
  _mm_storeu_si128(out + 10, _mm_and_si128(mask, _mm_srli_epi32(w0, 6)));
  _mm_storeu_si128(out + 11, _mm_and_si128(mask, _mm_srli_epi32(w0, 13)));
  _mm_storeu_si128(out + 12, _mm_and_si128(mask, _mm_srli_epi32(w0, 20)));
  w1 = _mm_loadu_si128(compressed + 3);
  _mm_storeu_si128(out + 13,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 27),
                                                    _mm_slli_epi32(w1, 5))));
  _mm_storeu_si128(out + 14, _mm_and_si128(mask, _mm_srli_epi32(w1, 2)));
  _mm_storeu_si128(out + 15, _mm_and_si128(mask, _mm_srli_epi32(w1, 9)));
}

/* we packed 64 8-bit values, touching 4 128-bit words, using 64 bytes */
static void simdunpackblock8(const __m128i *compressed, u32 *pout) {
  /* we are going to access  4 128-bit words */
  __m128i w0, w1;
  __m128i *out = (__m128i *)pout;
  const __m128i mask = _mm_set1_epi32(255);
  w0 = _mm_loadu_si128(compressed);
  _mm_storeu_si128(out + 0, _mm_and_si128(mask, w0));
  _mm_storeu_si128(out + 1, _mm_and_si128(mask, _mm_srli_epi32(w0, 8)));
  _mm_storeu_si128(out + 2, _mm_and_si128(mask, _mm_srli_epi32(w0, 16)));
  _mm_storeu_si128(out + 3, _mm_srli_epi32(w0, 24));
  w1 = _mm_loadu_si128(compressed + 1);
  _mm_storeu_si128(out + 4, _mm_and_si128(mask, w1));
  _mm_storeu_si128(out + 5, _mm_and_si128(mask, _mm_srli_epi32(w1, 8)));
  _mm_storeu_si128(out + 6, _mm_and_si128(mask, _mm_srli_epi32(w1, 16)));
  _mm_storeu_si128(out + 7, _mm_srli_epi32(w1, 24));
  w0 = _mm_loadu_si128(compressed + 2);
  _mm_storeu_si128(out + 8, _mm_and_si128(mask, w0));
  _mm_storeu_si128(out + 9, _mm_and_si128(mask, _mm_srli_epi32(w0, 8)));
  _mm_storeu_si128(out + 10, _mm_and_si128(mask, _mm_srli_epi32(w0, 16)));
  _mm_storeu_si128(out + 11, _mm_srli_epi32(w0, 24));
  w1 = _mm_loadu_si128(compressed + 3);
  _mm_storeu_si128(out + 12, _mm_and_si128(mask, w1));
  _mm_storeu_si128(out + 13, _mm_and_si128(mask, _mm_srli_epi32(w1, 8)));
  _mm_storeu_si128(out + 14, _mm_and_si128(mask, _mm_srli_epi32(w1, 16)));
  _mm_storeu_si128(out + 15, _mm_srli_epi32(w1, 24));
}

/* we packed 64 9-bit values, touching 5 128-bit words, using 80 bytes */
static void simdunpackblock9(const __m128i *compressed, u32 *pout) {
  /* we are going to access  5 128-bit words */
  __m128i w0, w1;
  __m128i *out = (__m128i *)pout;
  const __m128i mask = _mm_set1_epi32(511);
  w0 = _mm_loadu_si128(compressed);
  _mm_storeu_si128(out + 0, _mm_and_si128(mask, w0));
  _mm_storeu_si128(out + 1, _mm_and_si128(mask, _mm_srli_epi32(w0, 9)));
  _mm_storeu_si128(out + 2, _mm_and_si128(mask, _mm_srli_epi32(w0, 18)));
  w1 = _mm_loadu_si128(compressed + 1);
  _mm_storeu_si128(out + 3,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 27),
                                                    _mm_slli_epi32(w1, 5))));
  _mm_storeu_si128(out + 4, _mm_and_si128(mask, _mm_srli_epi32(w1, 4)));
  _mm_storeu_si128(out + 5, _mm_and_si128(mask, _mm_srli_epi32(w1, 13)));
  _mm_storeu_si128(out + 6, _mm_and_si128(mask, _mm_srli_epi32(w1, 22)));
  w0 = _mm_loadu_si128(compressed + 2);
  _mm_storeu_si128(out + 7,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 31),
                                                    _mm_slli_epi32(w0, 1))));
  _mm_storeu_si128(out + 8, _mm_and_si128(mask, _mm_srli_epi32(w0, 8)));
  _mm_storeu_si128(out + 9, _mm_and_si128(mask, _mm_srli_epi32(w0, 17)));
  w1 = _mm_loadu_si128(compressed + 3);
  _mm_storeu_si128(out + 10,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 26),
                                                    _mm_slli_epi32(w1, 6))));
  _mm_storeu_si128(out + 11, _mm_and_si128(mask, _mm_srli_epi32(w1, 3)));
  _mm_storeu_si128(out + 12, _mm_and_si128(mask, _mm_srli_epi32(w1, 12)));
  _mm_storeu_si128(out + 13, _mm_and_si128(mask, _mm_srli_epi32(w1, 21)));
  w0 = _mm_loadu_si128(compressed + 4);
  _mm_storeu_si128(out + 14,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 30),
                                                    _mm_slli_epi32(w0, 2))));
  _mm_storeu_si128(out + 15, _mm_and_si128(mask, _mm_srli_epi32(w0, 7)));
}

/* we packed 64 10-bit values, touching 5 128-bit words, using 80 bytes */
static void simdunpackblock10(const __m128i *compressed, u32 *pout) {
  /* we are going to access  5 128-bit words */
  __m128i w0, w1;
  __m128i *out = (__m128i *)pout;
  const __m128i mask = _mm_set1_epi32(1023);
  w0 = _mm_loadu_si128(compressed);
  _mm_storeu_si128(out + 0, _mm_and_si128(mask, w0));
  _mm_storeu_si128(out + 1, _mm_and_si128(mask, _mm_srli_epi32(w0, 10)));
  _mm_storeu_si128(out + 2, _mm_and_si128(mask, _mm_srli_epi32(w0, 20)));
  w1 = _mm_loadu_si128(compressed + 1);
  _mm_storeu_si128(out + 3,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 30),
                                                    _mm_slli_epi32(w1, 2))));
  _mm_storeu_si128(out + 4, _mm_and_si128(mask, _mm_srli_epi32(w1, 8)));
  _mm_storeu_si128(out + 5, _mm_and_si128(mask, _mm_srli_epi32(w1, 18)));
  w0 = _mm_loadu_si128(compressed + 2);
  _mm_storeu_si128(out + 6,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 28),
                                                    _mm_slli_epi32(w0, 4))));
  _mm_storeu_si128(out + 7, _mm_and_si128(mask, _mm_srli_epi32(w0, 6)));
  _mm_storeu_si128(out + 8, _mm_and_si128(mask, _mm_srli_epi32(w0, 16)));
  w1 = _mm_loadu_si128(compressed + 3);
  _mm_storeu_si128(out + 9,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 26),
                                                    _mm_slli_epi32(w1, 6))));
  _mm_storeu_si128(out + 10, _mm_and_si128(mask, _mm_srli_epi32(w1, 4)));
  _mm_storeu_si128(out + 11, _mm_and_si128(mask, _mm_srli_epi32(w1, 14)));
  w0 = _mm_loadu_si128(compressed + 4);
  _mm_storeu_si128(out + 12,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 24),
                                                    _mm_slli_epi32(w0, 8))));
  _mm_storeu_si128(out + 13, _mm_and_si128(mask, _mm_srli_epi32(w0, 2)));
  _mm_storeu_si128(out + 14, _mm_and_si128(mask, _mm_srli_epi32(w0, 12)));
  _mm_storeu_si128(out + 15, _mm_srli_epi32(w0, 22));
}

/* we packed 64 11-bit values, touching 6 128-bit words, using 96 bytes */
static void simdunpackblock11(const __m128i *compressed, u32 *pout) {
  /* we are going to access  6 128-bit words */
  __m128i w0, w1;
  __m128i *out = (__m128i *)pout;
  const __m128i mask = _mm_set1_epi32(2047);
  w0 = _mm_loadu_si128(compressed);
  _mm_storeu_si128(out + 0, _mm_and_si128(mask, w0));
  _mm_storeu_si128(out + 1, _mm_and_si128(mask, _mm_srli_epi32(w0, 11)));
  w1 = _mm_loadu_si128(compressed + 1);
  _mm_storeu_si128(out + 2,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 22),
                                                    _mm_slli_epi32(w1, 10))));
  _mm_storeu_si128(out + 3, _mm_and_si128(mask, _mm_srli_epi32(w1, 1)));
  _mm_storeu_si128(out + 4, _mm_and_si128(mask, _mm_srli_epi32(w1, 12)));
  w0 = _mm_loadu_si128(compressed + 2);
  _mm_storeu_si128(out + 5,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 23),
                                                    _mm_slli_epi32(w0, 9))));
  _mm_storeu_si128(out + 6, _mm_and_si128(mask, _mm_srli_epi32(w0, 2)));
  _mm_storeu_si128(out + 7, _mm_and_si128(mask, _mm_srli_epi32(w0, 13)));
  w1 = _mm_loadu_si128(compressed + 3);
  _mm_storeu_si128(out + 8,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 24),
                                                    _mm_slli_epi32(w1, 8))));
  _mm_storeu_si128(out + 9, _mm_and_si128(mask, _mm_srli_epi32(w1, 3)));
  _mm_storeu_si128(out + 10, _mm_and_si128(mask, _mm_srli_epi32(w1, 14)));
  w0 = _mm_loadu_si128(compressed + 4);
  _mm_storeu_si128(out + 11,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 25),
                                                    _mm_slli_epi32(w0, 7))));
  _mm_storeu_si128(out + 12, _mm_and_si128(mask, _mm_srli_epi32(w0, 4)));
  _mm_storeu_si128(out + 13, _mm_and_si128(mask, _mm_srli_epi32(w0, 15)));
  w1 = _mm_loadu_si128(compressed + 5);
  _mm_storeu_si128(out + 14,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 26),
                                                    _mm_slli_epi32(w1, 6))));
  _mm_storeu_si128(out + 15, _mm_and_si128(mask, _mm_srli_epi32(w1, 5)));
}

/* we packed 64 12-bit values, touching 6 128-bit words, using 96 bytes */
static void simdunpackblock12(const __m128i *compressed, u32 *pout) {
  /* we are going to access  6 128-bit words */
  __m128i w0, w1;
  __m128i *out = (__m128i *)pout;
  const __m128i mask = _mm_set1_epi32(4095);
  w0 = _mm_loadu_si128(compressed);
  _mm_storeu_si128(out + 0, _mm_and_si128(mask, w0));
  _mm_storeu_si128(out + 1, _mm_and_si128(mask, _mm_srli_epi32(w0, 12)));
  w1 = _mm_loadu_si128(compressed + 1);
  _mm_storeu_si128(out + 2,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 24),
                                                    _mm_slli_epi32(w1, 8))));
  _mm_storeu_si128(out + 3, _mm_and_si128(mask, _mm_srli_epi32(w1, 4)));
  _mm_storeu_si128(out + 4, _mm_and_si128(mask, _mm_srli_epi32(w1, 16)));
  w0 = _mm_loadu_si128(compressed + 2);
  _mm_storeu_si128(out + 5,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 28),
                                                    _mm_slli_epi32(w0, 4))));
  _mm_storeu_si128(out + 6, _mm_and_si128(mask, _mm_srli_epi32(w0, 8)));
  _mm_storeu_si128(out + 7, _mm_srli_epi32(w0, 20));
  w1 = _mm_loadu_si128(compressed + 3);
  _mm_storeu_si128(out + 8, _mm_and_si128(mask, w1));
  _mm_storeu_si128(out + 9, _mm_and_si128(mask, _mm_srli_epi32(w1, 12)));
  w0 = _mm_loadu_si128(compressed + 4);
  _mm_storeu_si128(out + 10,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 24),
                                                    _mm_slli_epi32(w0, 8))));
  _mm_storeu_si128(out + 11, _mm_and_si128(mask, _mm_srli_epi32(w0, 4)));
  _mm_storeu_si128(out + 12, _mm_and_si128(mask, _mm_srli_epi32(w0, 16)));
  w1 = _mm_loadu_si128(compressed + 5);
  _mm_storeu_si128(out + 13,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 28),
                                                    _mm_slli_epi32(w1, 4))));
  _mm_storeu_si128(out + 14, _mm_and_si128(mask, _mm_srli_epi32(w1, 8)));
  _mm_storeu_si128(out + 15, _mm_srli_epi32(w1, 20));
}

/* we packed 64 13-bit values, touching 7 128-bit words, using 112 bytes */
static void simdunpackblock13(const __m128i *compressed, u32 *pout) {
  /* we are going to access  7 128-bit words */
  __m128i w0, w1;
  __m128i *out = (__m128i *)pout;
  const __m128i mask = _mm_set1_epi32(8191);
  w0 = _mm_loadu_si128(compressed);
  _mm_storeu_si128(out + 0, _mm_and_si128(mask, w0));
  _mm_storeu_si128(out + 1, _mm_and_si128(mask, _mm_srli_epi32(w0, 13)));
  w1 = _mm_loadu_si128(compressed + 1);
  _mm_storeu_si128(out + 2,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 26),
                                                    _mm_slli_epi32(w1, 6))));
  _mm_storeu_si128(out + 3, _mm_and_si128(mask, _mm_srli_epi32(w1, 7)));
  w0 = _mm_loadu_si128(compressed + 2);
  _mm_storeu_si128(out + 4,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 20),
                                                    _mm_slli_epi32(w0, 12))));
  _mm_storeu_si128(out + 5, _mm_and_si128(mask, _mm_srli_epi32(w0, 1)));
  _mm_storeu_si128(out + 6, _mm_and_si128(mask, _mm_srli_epi32(w0, 14)));
  w1 = _mm_loadu_si128(compressed + 3);
  _mm_storeu_si128(out + 7,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 27),
                                                    _mm_slli_epi32(w1, 5))));
  _mm_storeu_si128(out + 8, _mm_and_si128(mask, _mm_srli_epi32(w1, 8)));
  w0 = _mm_loadu_si128(compressed + 4);
  _mm_storeu_si128(out + 9,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 21),
                                                    _mm_slli_epi32(w0, 11))));
  _mm_storeu_si128(out + 10, _mm_and_si128(mask, _mm_srli_epi32(w0, 2)));
  _mm_storeu_si128(out + 11, _mm_and_si128(mask, _mm_srli_epi32(w0, 15)));
  w1 = _mm_loadu_si128(compressed + 5);
  _mm_storeu_si128(out + 12,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 28),
                                                    _mm_slli_epi32(w1, 4))));
  _mm_storeu_si128(out + 13, _mm_and_si128(mask, _mm_srli_epi32(w1, 9)));
  w0 = _mm_loadu_si128(compressed + 6);
  _mm_storeu_si128(out + 14,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 22),
                                                    _mm_slli_epi32(w0, 10))));
  _mm_storeu_si128(out + 15, _mm_and_si128(mask, _mm_srli_epi32(w0, 3)));
}

/* we packed 64 14-bit values, touching 7 128-bit words, using 112 bytes */
static void simdunpackblock14(const __m128i *compressed, u32 *pout) {
  /* we are going to access  7 128-bit words */
  __m128i w0, w1;
  __m128i *out = (__m128i *)pout;
  const __m128i mask = _mm_set1_epi32(16383);
  w0 = _mm_loadu_si128(compressed);
  _mm_storeu_si128(out + 0, _mm_and_si128(mask, w0));
  _mm_storeu_si128(out + 1, _mm_and_si128(mask, _mm_srli_epi32(w0, 14)));
  w1 = _mm_loadu_si128(compressed + 1);
  _mm_storeu_si128(out + 2,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 28),
                                                    _mm_slli_epi32(w1, 4))));
  _mm_storeu_si128(out + 3, _mm_and_si128(mask, _mm_srli_epi32(w1, 10)));
  w0 = _mm_loadu_si128(compressed + 2);
  _mm_storeu_si128(out + 4,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 24),
                                                    _mm_slli_epi32(w0, 8))));
  _mm_storeu_si128(out + 5, _mm_and_si128(mask, _mm_srli_epi32(w0, 6)));
  w1 = _mm_loadu_si128(compressed + 3);
  _mm_storeu_si128(out + 6,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 20),
                                                    _mm_slli_epi32(w1, 12))));
  _mm_storeu_si128(out + 7, _mm_and_si128(mask, _mm_srli_epi32(w1, 2)));
  _mm_storeu_si128(out + 8, _mm_and_si128(mask, _mm_srli_epi32(w1, 16)));
  w0 = _mm_loadu_si128(compressed + 4);
  _mm_storeu_si128(out + 9,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 30),
                                                    _mm_slli_epi32(w0, 2))));
  _mm_storeu_si128(out + 10, _mm_and_si128(mask, _mm_srli_epi32(w0, 12)));
  w1 = _mm_loadu_si128(compressed + 5);
  _mm_storeu_si128(out + 11,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 26),
                                                    _mm_slli_epi32(w1, 6))));
  _mm_storeu_si128(out + 12, _mm_and_si128(mask, _mm_srli_epi32(w1, 8)));
  w0 = _mm_loadu_si128(compressed + 6);
  _mm_storeu_si128(out + 13,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 22),
                                                    _mm_slli_epi32(w0, 10))));
  _mm_storeu_si128(out + 14, _mm_and_si128(mask, _mm_srli_epi32(w0, 4)));
  _mm_storeu_si128(out + 15, _mm_srli_epi32(w0, 18));
}

/* we packed 64 15-bit values, touching 8 128-bit words, using 128 bytes */
static void simdunpackblock15(const __m128i *compressed, u32 *pout) {
  /* we are going to access  8 128-bit words */
  __m128i w0, w1;
  __m128i *out = (__m128i *)pout;
  const __m128i mask = _mm_set1_epi32(32767);
  w0 = _mm_loadu_si128(compressed);
  _mm_storeu_si128(out + 0, _mm_and_si128(mask, w0));
  _mm_storeu_si128(out + 1, _mm_and_si128(mask, _mm_srli_epi32(w0, 15)));
  w1 = _mm_loadu_si128(compressed + 1);
  _mm_storeu_si128(out + 2,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 30),
                                                    _mm_slli_epi32(w1, 2))));
  _mm_storeu_si128(out + 3, _mm_and_si128(mask, _mm_srli_epi32(w1, 13)));
  w0 = _mm_loadu_si128(compressed + 2);
  _mm_storeu_si128(out + 4,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 28),
                                                    _mm_slli_epi32(w0, 4))));
  _mm_storeu_si128(out + 5, _mm_and_si128(mask, _mm_srli_epi32(w0, 11)));
  w1 = _mm_loadu_si128(compressed + 3);
  _mm_storeu_si128(out + 6,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 26),
                                                    _mm_slli_epi32(w1, 6))));
  _mm_storeu_si128(out + 7, _mm_and_si128(mask, _mm_srli_epi32(w1, 9)));
  w0 = _mm_loadu_si128(compressed + 4);
  _mm_storeu_si128(out + 8,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 24),
                                                    _mm_slli_epi32(w0, 8))));
  _mm_storeu_si128(out + 9, _mm_and_si128(mask, _mm_srli_epi32(w0, 7)));
  w1 = _mm_loadu_si128(compressed + 5);
  _mm_storeu_si128(out + 10,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 22),
                                                    _mm_slli_epi32(w1, 10))));
  _mm_storeu_si128(out + 11, _mm_and_si128(mask, _mm_srli_epi32(w1, 5)));
  w0 = _mm_loadu_si128(compressed + 6);
  _mm_storeu_si128(out + 12,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 20),
                                                    _mm_slli_epi32(w0, 12))));
  _mm_storeu_si128(out + 13, _mm_and_si128(mask, _mm_srli_epi32(w0, 3)));
  w1 = _mm_loadu_si128(compressed + 7);
  _mm_storeu_si128(out + 14,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 18),
                                                    _mm_slli_epi32(w1, 14))));
  _mm_storeu_si128(out + 15, _mm_and_si128(mask, _mm_srli_epi32(w1, 1)));
}

/* we packed 64 16-bit values, touching 8 128-bit words, using 128 bytes */
static void simdunpackblock16(const __m128i *compressed, u32 *pout) {
  /* we are going to access  8 128-bit words */
  __m128i w0, w1;
  __m128i *out = (__m128i *)pout;
  const __m128i mask = _mm_set1_epi32(65535);
  w0 = _mm_loadu_si128(compressed);
  _mm_storeu_si128(out + 0, _mm_and_si128(mask, w0));
  _mm_storeu_si128(out + 1, _mm_srli_epi32(w0, 16));
  w1 = _mm_loadu_si128(compressed + 1);
  _mm_storeu_si128(out + 2, _mm_and_si128(mask, w1));
  _mm_storeu_si128(out + 3, _mm_srli_epi32(w1, 16));
  w0 = _mm_loadu_si128(compressed + 2);
  _mm_storeu_si128(out + 4, _mm_and_si128(mask, w0));
  _mm_storeu_si128(out + 5, _mm_srli_epi32(w0, 16));
  w1 = _mm_loadu_si128(compressed + 3);
  _mm_storeu_si128(out + 6, _mm_and_si128(mask, w1));
  _mm_storeu_si128(out + 7, _mm_srli_epi32(w1, 16));
  w0 = _mm_loadu_si128(compressed + 4);
  _mm_storeu_si128(out + 8, _mm_and_si128(mask, w0));
  _mm_storeu_si128(out + 9, _mm_srli_epi32(w0, 16));
  w1 = _mm_loadu_si128(compressed + 5);
  _mm_storeu_si128(out + 10, _mm_and_si128(mask, w1));
  _mm_storeu_si128(out + 11, _mm_srli_epi32(w1, 16));
  w0 = _mm_loadu_si128(compressed + 6);
  _mm_storeu_si128(out + 12, _mm_and_si128(mask, w0));
  _mm_storeu_si128(out + 13, _mm_srli_epi32(w0, 16));
  w1 = _mm_loadu_si128(compressed + 7);
  _mm_storeu_si128(out + 14, _mm_and_si128(mask, w1));
  _mm_storeu_si128(out + 15, _mm_srli_epi32(w1, 16));
}

/* we packed 64 17-bit values, touching 9 128-bit words, using 144 bytes */
static void simdunpackblock17(const __m128i *compressed, u32 *pout) {
  /* we are going to access  9 128-bit words */
  __m128i w0, w1;
  __m128i *out = (__m128i *)pout;
  const __m128i mask = _mm_set1_epi32(131071);
  w0 = _mm_loadu_si128(compressed);
  _mm_storeu_si128(out + 0, _mm_and_si128(mask, w0));
  w1 = _mm_loadu_si128(compressed + 1);
  _mm_storeu_si128(out + 1,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 17),
                                                    _mm_slli_epi32(w1, 15))));
  _mm_storeu_si128(out + 2, _mm_and_si128(mask, _mm_srli_epi32(w1, 2)));
  w0 = _mm_loadu_si128(compressed + 2);
  _mm_storeu_si128(out + 3,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 19),
                                                    _mm_slli_epi32(w0, 13))));
  _mm_storeu_si128(out + 4, _mm_and_si128(mask, _mm_srli_epi32(w0, 4)));
  w1 = _mm_loadu_si128(compressed + 3);
  _mm_storeu_si128(out + 5,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 21),
                                                    _mm_slli_epi32(w1, 11))));
  _mm_storeu_si128(out + 6, _mm_and_si128(mask, _mm_srli_epi32(w1, 6)));
  w0 = _mm_loadu_si128(compressed + 4);
  _mm_storeu_si128(out + 7,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 23),
                                                    _mm_slli_epi32(w0, 9))));
  _mm_storeu_si128(out + 8, _mm_and_si128(mask, _mm_srli_epi32(w0, 8)));
  w1 = _mm_loadu_si128(compressed + 5);
  _mm_storeu_si128(out + 9,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 25),
                                                    _mm_slli_epi32(w1, 7))));
  _mm_storeu_si128(out + 10, _mm_and_si128(mask, _mm_srli_epi32(w1, 10)));
  w0 = _mm_loadu_si128(compressed + 6);
  _mm_storeu_si128(out + 11,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 27),
                                                    _mm_slli_epi32(w0, 5))));
  _mm_storeu_si128(out + 12, _mm_and_si128(mask, _mm_srli_epi32(w0, 12)));
  w1 = _mm_loadu_si128(compressed + 7);
  _mm_storeu_si128(out + 13,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 29),
                                                    _mm_slli_epi32(w1, 3))));
  _mm_storeu_si128(out + 14, _mm_and_si128(mask, _mm_srli_epi32(w1, 14)));
  w0 = _mm_loadu_si128(compressed + 8);
  _mm_storeu_si128(out + 15,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 31),
                                                    _mm_slli_epi32(w0, 1))));
}

/* we packed 64 18-bit values, touching 9 128-bit words, using 144 bytes */
static void simdunpackblock18(const __m128i *compressed, u32 *pout) {
  /* we are going to access  9 128-bit words */
  __m128i w0, w1;
  __m128i *out = (__m128i *)pout;
  const __m128i mask = _mm_set1_epi32(262143);
  w0 = _mm_loadu_si128(compressed);
  _mm_storeu_si128(out + 0, _mm_and_si128(mask, w0));
  w1 = _mm_loadu_si128(compressed + 1);
  _mm_storeu_si128(out + 1,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 18),
                                                    _mm_slli_epi32(w1, 14))));
  _mm_storeu_si128(out + 2, _mm_and_si128(mask, _mm_srli_epi32(w1, 4)));
  w0 = _mm_loadu_si128(compressed + 2);
  _mm_storeu_si128(out + 3,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 22),
                                                    _mm_slli_epi32(w0, 10))));
  _mm_storeu_si128(out + 4, _mm_and_si128(mask, _mm_srli_epi32(w0, 8)));
  w1 = _mm_loadu_si128(compressed + 3);
  _mm_storeu_si128(out + 5,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 26),
                                                    _mm_slli_epi32(w1, 6))));
  _mm_storeu_si128(out + 6, _mm_and_si128(mask, _mm_srli_epi32(w1, 12)));
  w0 = _mm_loadu_si128(compressed + 4);
  _mm_storeu_si128(out + 7,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 30),
                                                    _mm_slli_epi32(w0, 2))));
  w1 = _mm_loadu_si128(compressed + 5);
  _mm_storeu_si128(out + 8,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 16),
                                                    _mm_slli_epi32(w1, 16))));
  _mm_storeu_si128(out + 9, _mm_and_si128(mask, _mm_srli_epi32(w1, 2)));
  w0 = _mm_loadu_si128(compressed + 6);
  _mm_storeu_si128(out + 10,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 20),
                                                    _mm_slli_epi32(w0, 12))));
  _mm_storeu_si128(out + 11, _mm_and_si128(mask, _mm_srli_epi32(w0, 6)));
  w1 = _mm_loadu_si128(compressed + 7);
  _mm_storeu_si128(out + 12,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 24),
                                                    _mm_slli_epi32(w1, 8))));
  _mm_storeu_si128(out + 13, _mm_and_si128(mask, _mm_srli_epi32(w1, 10)));
  w0 = _mm_loadu_si128(compressed + 8);
  _mm_storeu_si128(out + 14,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 28),
                                                    _mm_slli_epi32(w0, 4))));
  _mm_storeu_si128(out + 15, _mm_srli_epi32(w0, 14));
}

/* we packed 64 19-bit values, touching 10 128-bit words, using 160 bytes */
static void simdunpackblock19(const __m128i *compressed, u32 *pout) {
  /* we are going to access  10 128-bit words */
  __m128i w0, w1;
  __m128i *out = (__m128i *)pout;
  const __m128i mask = _mm_set1_epi32(524287);
  w0 = _mm_loadu_si128(compressed);
  _mm_storeu_si128(out + 0, _mm_and_si128(mask, w0));
  w1 = _mm_loadu_si128(compressed + 1);
  _mm_storeu_si128(out + 1,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 19),
                                                    _mm_slli_epi32(w1, 13))));
  _mm_storeu_si128(out + 2, _mm_and_si128(mask, _mm_srli_epi32(w1, 6)));
  w0 = _mm_loadu_si128(compressed + 2);
  _mm_storeu_si128(out + 3,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 25),
                                                    _mm_slli_epi32(w0, 7))));
  _mm_storeu_si128(out + 4, _mm_and_si128(mask, _mm_srli_epi32(w0, 12)));
  w1 = _mm_loadu_si128(compressed + 3);
  _mm_storeu_si128(out + 5,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 31),
                                                    _mm_slli_epi32(w1, 1))));
  w0 = _mm_loadu_si128(compressed + 4);
  _mm_storeu_si128(out + 6,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 18),
                                                    _mm_slli_epi32(w0, 14))));
  _mm_storeu_si128(out + 7, _mm_and_si128(mask, _mm_srli_epi32(w0, 5)));
  w1 = _mm_loadu_si128(compressed + 5);
  _mm_storeu_si128(out + 8,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 24),
                                                    _mm_slli_epi32(w1, 8))));
  _mm_storeu_si128(out + 9, _mm_and_si128(mask, _mm_srli_epi32(w1, 11)));
  w0 = _mm_loadu_si128(compressed + 6);
  _mm_storeu_si128(out + 10,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 30),
                                                    _mm_slli_epi32(w0, 2))));
  w1 = _mm_loadu_si128(compressed + 7);
  _mm_storeu_si128(out + 11,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 17),
                                                    _mm_slli_epi32(w1, 15))));
  _mm_storeu_si128(out + 12, _mm_and_si128(mask, _mm_srli_epi32(w1, 4)));
  w0 = _mm_loadu_si128(compressed + 8);
  _mm_storeu_si128(out + 13,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 23),
                                                    _mm_slli_epi32(w0, 9))));
  _mm_storeu_si128(out + 14, _mm_and_si128(mask, _mm_srli_epi32(w0, 10)));
  w1 = _mm_loadu_si128(compressed + 9);
  _mm_storeu_si128(out + 15,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 29),
                                                    _mm_slli_epi32(w1, 3))));
}

/* we packed 64 20-bit values, touching 10 128-bit words, using 160 bytes */
static void simdunpackblock20(const __m128i *compressed, u32 *pout) {
  /* we are going to access  10 128-bit words */
  __m128i w0, w1;
  __m128i *out = (__m128i *)pout;
  const __m128i mask = _mm_set1_epi32(1048575);
  w0 = _mm_loadu_si128(compressed);
  _mm_storeu_si128(out + 0, _mm_and_si128(mask, w0));
  w1 = _mm_loadu_si128(compressed + 1);
  _mm_storeu_si128(out + 1,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 20),
                                                    _mm_slli_epi32(w1, 12))));
  _mm_storeu_si128(out + 2, _mm_and_si128(mask, _mm_srli_epi32(w1, 8)));
  w0 = _mm_loadu_si128(compressed + 2);
  _mm_storeu_si128(out + 3,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 28),
                                                    _mm_slli_epi32(w0, 4))));
  w1 = _mm_loadu_si128(compressed + 3);
  _mm_storeu_si128(out + 4,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 16),
                                                    _mm_slli_epi32(w1, 16))));
  _mm_storeu_si128(out + 5, _mm_and_si128(mask, _mm_srli_epi32(w1, 4)));
  w0 = _mm_loadu_si128(compressed + 4);
  _mm_storeu_si128(out + 6,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 24),
                                                    _mm_slli_epi32(w0, 8))));
  _mm_storeu_si128(out + 7, _mm_srli_epi32(w0, 12));
  w1 = _mm_loadu_si128(compressed + 5);
  _mm_storeu_si128(out + 8, _mm_and_si128(mask, w1));
  w0 = _mm_loadu_si128(compressed + 6);
  _mm_storeu_si128(out + 9,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 20),
                                                    _mm_slli_epi32(w0, 12))));
  _mm_storeu_si128(out + 10, _mm_and_si128(mask, _mm_srli_epi32(w0, 8)));
  w1 = _mm_loadu_si128(compressed + 7);
  _mm_storeu_si128(out + 11,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 28),
                                                    _mm_slli_epi32(w1, 4))));
  w0 = _mm_loadu_si128(compressed + 8);
  _mm_storeu_si128(out + 12,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 16),
                                                    _mm_slli_epi32(w0, 16))));
  _mm_storeu_si128(out + 13, _mm_and_si128(mask, _mm_srli_epi32(w0, 4)));
  w1 = _mm_loadu_si128(compressed + 9);
  _mm_storeu_si128(out + 14,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 24),
                                                    _mm_slli_epi32(w1, 8))));
  _mm_storeu_si128(out + 15, _mm_srli_epi32(w1, 12));
}

/* we packed 64 21-bit values, touching 11 128-bit words, using 176 bytes */
static void simdunpackblock21(const __m128i *compressed, u32 *pout) {
  /* we are going to access  11 128-bit words */
  __m128i w0, w1;
  __m128i *out = (__m128i *)pout;
  const __m128i mask = _mm_set1_epi32(2097151);
  w0 = _mm_loadu_si128(compressed);
  _mm_storeu_si128(out + 0, _mm_and_si128(mask, w0));
  w1 = _mm_loadu_si128(compressed + 1);
  _mm_storeu_si128(out + 1,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 21),
                                                    _mm_slli_epi32(w1, 11))));
  _mm_storeu_si128(out + 2, _mm_and_si128(mask, _mm_srli_epi32(w1, 10)));
  w0 = _mm_loadu_si128(compressed + 2);
  _mm_storeu_si128(out + 3,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 31),
                                                    _mm_slli_epi32(w0, 1))));
  w1 = _mm_loadu_si128(compressed + 3);
  _mm_storeu_si128(out + 4,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 20),
                                                    _mm_slli_epi32(w1, 12))));
  _mm_storeu_si128(out + 5, _mm_and_si128(mask, _mm_srli_epi32(w1, 9)));
  w0 = _mm_loadu_si128(compressed + 4);
  _mm_storeu_si128(out + 6,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 30),
                                                    _mm_slli_epi32(w0, 2))));
  w1 = _mm_loadu_si128(compressed + 5);
  _mm_storeu_si128(out + 7,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 19),
                                                    _mm_slli_epi32(w1, 13))));
  _mm_storeu_si128(out + 8, _mm_and_si128(mask, _mm_srli_epi32(w1, 8)));
  w0 = _mm_loadu_si128(compressed + 6);
  _mm_storeu_si128(out + 9,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 29),
                                                    _mm_slli_epi32(w0, 3))));
  w1 = _mm_loadu_si128(compressed + 7);
  _mm_storeu_si128(out + 10,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 18),
                                                    _mm_slli_epi32(w1, 14))));
  _mm_storeu_si128(out + 11, _mm_and_si128(mask, _mm_srli_epi32(w1, 7)));
  w0 = _mm_loadu_si128(compressed + 8);
  _mm_storeu_si128(out + 12,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 28),
                                                    _mm_slli_epi32(w0, 4))));
  w1 = _mm_loadu_si128(compressed + 9);
  _mm_storeu_si128(out + 13,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 17),
                                                    _mm_slli_epi32(w1, 15))));
  _mm_storeu_si128(out + 14, _mm_and_si128(mask, _mm_srli_epi32(w1, 6)));
  w0 = _mm_loadu_si128(compressed + 10);
  _mm_storeu_si128(out + 15,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 27),
                                                    _mm_slli_epi32(w0, 5))));
}

/* we packed 64 22-bit values, touching 11 128-bit words, using 176 bytes */
static void simdunpackblock22(const __m128i *compressed, u32 *pout) {
  /* we are going to access  11 128-bit words */
  __m128i w0, w1;
  __m128i *out = (__m128i *)pout;
  const __m128i mask = _mm_set1_epi32(4194303);
  w0 = _mm_loadu_si128(compressed);
  _mm_storeu_si128(out + 0, _mm_and_si128(mask, w0));
  w1 = _mm_loadu_si128(compressed + 1);
  _mm_storeu_si128(out + 1,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 22),
                                                    _mm_slli_epi32(w1, 10))));
  w0 = _mm_loadu_si128(compressed + 2);
  _mm_storeu_si128(out + 2,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 12),
                                                    _mm_slli_epi32(w0, 20))));
  _mm_storeu_si128(out + 3, _mm_and_si128(mask, _mm_srli_epi32(w0, 2)));
  w1 = _mm_loadu_si128(compressed + 3);
  _mm_storeu_si128(out + 4,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 24),
                                                    _mm_slli_epi32(w1, 8))));
  w0 = _mm_loadu_si128(compressed + 4);
  _mm_storeu_si128(out + 5,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 14),
                                                    _mm_slli_epi32(w0, 18))));
  _mm_storeu_si128(out + 6, _mm_and_si128(mask, _mm_srli_epi32(w0, 4)));
  w1 = _mm_loadu_si128(compressed + 5);
  _mm_storeu_si128(out + 7,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 26),
                                                    _mm_slli_epi32(w1, 6))));
  w0 = _mm_loadu_si128(compressed + 6);
  _mm_storeu_si128(out + 8,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 16),
                                                    _mm_slli_epi32(w0, 16))));
  _mm_storeu_si128(out + 9, _mm_and_si128(mask, _mm_srli_epi32(w0, 6)));
  w1 = _mm_loadu_si128(compressed + 7);
  _mm_storeu_si128(out + 10,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 28),
                                                    _mm_slli_epi32(w1, 4))));
  w0 = _mm_loadu_si128(compressed + 8);
  _mm_storeu_si128(out + 11,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 18),
                                                    _mm_slli_epi32(w0, 14))));
  _mm_storeu_si128(out + 12, _mm_and_si128(mask, _mm_srli_epi32(w0, 8)));
  w1 = _mm_loadu_si128(compressed + 9);
  _mm_storeu_si128(out + 13,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 30),
                                                    _mm_slli_epi32(w1, 2))));
  w0 = _mm_loadu_si128(compressed + 10);
  _mm_storeu_si128(out + 14,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 20),
                                                    _mm_slli_epi32(w0, 12))));
  _mm_storeu_si128(out + 15, _mm_srli_epi32(w0, 10));
}

/* we packed 64 23-bit values, touching 12 128-bit words, using 192 bytes */
static void simdunpackblock23(const __m128i *compressed, u32 *pout) {
  /* we are going to access  12 128-bit words */
  __m128i w0, w1;
  __m128i *out = (__m128i *)pout;
  const __m128i mask = _mm_set1_epi32(8388607);
  w0 = _mm_loadu_si128(compressed);
  _mm_storeu_si128(out + 0, _mm_and_si128(mask, w0));
  w1 = _mm_loadu_si128(compressed + 1);
  _mm_storeu_si128(out + 1,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 23),
                                                    _mm_slli_epi32(w1, 9))));
  w0 = _mm_loadu_si128(compressed + 2);
  _mm_storeu_si128(out + 2,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 14),
                                                    _mm_slli_epi32(w0, 18))));
  _mm_storeu_si128(out + 3, _mm_and_si128(mask, _mm_srli_epi32(w0, 5)));
  w1 = _mm_loadu_si128(compressed + 3);
  _mm_storeu_si128(out + 4,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 28),
                                                    _mm_slli_epi32(w1, 4))));
  w0 = _mm_loadu_si128(compressed + 4);
  _mm_storeu_si128(out + 5,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 19),
                                                    _mm_slli_epi32(w0, 13))));
  w1 = _mm_loadu_si128(compressed + 5);
  _mm_storeu_si128(out + 6,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 10),
                                                    _mm_slli_epi32(w1, 22))));
  _mm_storeu_si128(out + 7, _mm_and_si128(mask, _mm_srli_epi32(w1, 1)));
  w0 = _mm_loadu_si128(compressed + 6);
  _mm_storeu_si128(out + 8,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 24),
                                                    _mm_slli_epi32(w0, 8))));
  w1 = _mm_loadu_si128(compressed + 7);
  _mm_storeu_si128(out + 9,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 15),
                                                    _mm_slli_epi32(w1, 17))));
  _mm_storeu_si128(out + 10, _mm_and_si128(mask, _mm_srli_epi32(w1, 6)));
  w0 = _mm_loadu_si128(compressed + 8);
  _mm_storeu_si128(out + 11,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 29),
                                                    _mm_slli_epi32(w0, 3))));
  w1 = _mm_loadu_si128(compressed + 9);
  _mm_storeu_si128(out + 12,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 20),
                                                    _mm_slli_epi32(w1, 12))));
  w0 = _mm_loadu_si128(compressed + 10);
  _mm_storeu_si128(out + 13,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 11),
                                                    _mm_slli_epi32(w0, 21))));
  _mm_storeu_si128(out + 14, _mm_and_si128(mask, _mm_srli_epi32(w0, 2)));
  w1 = _mm_loadu_si128(compressed + 11);
  _mm_storeu_si128(out + 15,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 25),
                                                    _mm_slli_epi32(w1, 7))));
}

/* we packed 64 24-bit values, touching 12 128-bit words, using 192 bytes */
static void simdunpackblock24(const __m128i *compressed, u32 *pout) {
  /* we are going to access  12 128-bit words */
  __m128i w0, w1;
  __m128i *out = (__m128i *)pout;
  const __m128i mask = _mm_set1_epi32(16777215);
  w0 = _mm_loadu_si128(compressed);
  _mm_storeu_si128(out + 0, _mm_and_si128(mask, w0));
  w1 = _mm_loadu_si128(compressed + 1);
  _mm_storeu_si128(out + 1,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 24),
                                                    _mm_slli_epi32(w1, 8))));
  w0 = _mm_loadu_si128(compressed + 2);
  _mm_storeu_si128(out + 2,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 16),
                                                    _mm_slli_epi32(w0, 16))));
  _mm_storeu_si128(out + 3, _mm_srli_epi32(w0, 8));
  w1 = _mm_loadu_si128(compressed + 3);
  _mm_storeu_si128(out + 4, _mm_and_si128(mask, w1));
  w0 = _mm_loadu_si128(compressed + 4);
  _mm_storeu_si128(out + 5,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 24),
                                                    _mm_slli_epi32(w0, 8))));
  w1 = _mm_loadu_si128(compressed + 5);
  _mm_storeu_si128(out + 6,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 16),
                                                    _mm_slli_epi32(w1, 16))));
  _mm_storeu_si128(out + 7, _mm_srli_epi32(w1, 8));
  w0 = _mm_loadu_si128(compressed + 6);
  _mm_storeu_si128(out + 8, _mm_and_si128(mask, w0));
  w1 = _mm_loadu_si128(compressed + 7);
  _mm_storeu_si128(out + 9,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 24),
                                                    _mm_slli_epi32(w1, 8))));
  w0 = _mm_loadu_si128(compressed + 8);
  _mm_storeu_si128(out + 10,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 16),
                                                    _mm_slli_epi32(w0, 16))));
  _mm_storeu_si128(out + 11, _mm_srli_epi32(w0, 8));
  w1 = _mm_loadu_si128(compressed + 9);
  _mm_storeu_si128(out + 12, _mm_and_si128(mask, w1));
  w0 = _mm_loadu_si128(compressed + 10);
  _mm_storeu_si128(out + 13,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 24),
                                                    _mm_slli_epi32(w0, 8))));
  w1 = _mm_loadu_si128(compressed + 11);
  _mm_storeu_si128(out + 14,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 16),
                                                    _mm_slli_epi32(w1, 16))));
  _mm_storeu_si128(out + 15, _mm_srli_epi32(w1, 8));
}

/* we packed 64 25-bit values, touching 13 128-bit words, using 208 bytes */
static void simdunpackblock25(const __m128i *compressed, u32 *pout) {
  /* we are going to access  13 128-bit words */
  __m128i w0, w1;
  __m128i *out = (__m128i *)pout;
  const __m128i mask = _mm_set1_epi32(33554431);
  w0 = _mm_loadu_si128(compressed);
  _mm_storeu_si128(out + 0, _mm_and_si128(mask, w0));
  w1 = _mm_loadu_si128(compressed + 1);
  _mm_storeu_si128(out + 1,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 25),
                                                    _mm_slli_epi32(w1, 7))));
  w0 = _mm_loadu_si128(compressed + 2);
  _mm_storeu_si128(out + 2,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 18),
                                                    _mm_slli_epi32(w0, 14))));
  w1 = _mm_loadu_si128(compressed + 3);
  _mm_storeu_si128(out + 3,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 11),
                                                    _mm_slli_epi32(w1, 21))));
  _mm_storeu_si128(out + 4, _mm_and_si128(mask, _mm_srli_epi32(w1, 4)));
  w0 = _mm_loadu_si128(compressed + 4);
  _mm_storeu_si128(out + 5,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 29),
                                                    _mm_slli_epi32(w0, 3))));
  w1 = _mm_loadu_si128(compressed + 5);
  _mm_storeu_si128(out + 6,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 22),
                                                    _mm_slli_epi32(w1, 10))));
  w0 = _mm_loadu_si128(compressed + 6);
  _mm_storeu_si128(out + 7,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 15),
                                                    _mm_slli_epi32(w0, 17))));
  w1 = _mm_loadu_si128(compressed + 7);
  _mm_storeu_si128(out + 8,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 8),
                                                    _mm_slli_epi32(w1, 24))));
  _mm_storeu_si128(out + 9, _mm_and_si128(mask, _mm_srli_epi32(w1, 1)));
  w0 = _mm_loadu_si128(compressed + 8);
  _mm_storeu_si128(out + 10,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 26),
                                                    _mm_slli_epi32(w0, 6))));
  w1 = _mm_loadu_si128(compressed + 9);
  _mm_storeu_si128(out + 11,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 19),
                                                    _mm_slli_epi32(w1, 13))));
  w0 = _mm_loadu_si128(compressed + 10);
  _mm_storeu_si128(out + 12,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 12),
                                                    _mm_slli_epi32(w0, 20))));
  _mm_storeu_si128(out + 13, _mm_and_si128(mask, _mm_srli_epi32(w0, 5)));
  w1 = _mm_loadu_si128(compressed + 11);
  _mm_storeu_si128(out + 14,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 30),
                                                    _mm_slli_epi32(w1, 2))));
  w0 = _mm_loadu_si128(compressed + 12);
  _mm_storeu_si128(out + 15,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 23),
                                                    _mm_slli_epi32(w0, 9))));
}

/* we packed 64 26-bit values, touching 13 128-bit words, using 208 bytes */
static void simdunpackblock26(const __m128i *compressed, u32 *pout) {
  /* we are going to access  13 128-bit words */
  __m128i w0, w1;
  __m128i *out = (__m128i *)pout;
  const __m128i mask = _mm_set1_epi32(67108863);
  w0 = _mm_loadu_si128(compressed);
  _mm_storeu_si128(out + 0, _mm_and_si128(mask, w0));
  w1 = _mm_loadu_si128(compressed + 1);
  _mm_storeu_si128(out + 1,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 26),
                                                    _mm_slli_epi32(w1, 6))));
  w0 = _mm_loadu_si128(compressed + 2);
  _mm_storeu_si128(out + 2,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 20),
                                                    _mm_slli_epi32(w0, 12))));
  w1 = _mm_loadu_si128(compressed + 3);
  _mm_storeu_si128(out + 3,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 14),
                                                    _mm_slli_epi32(w1, 18))));
  w0 = _mm_loadu_si128(compressed + 4);
  _mm_storeu_si128(out + 4,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 8),
                                                    _mm_slli_epi32(w0, 24))));
  _mm_storeu_si128(out + 5, _mm_and_si128(mask, _mm_srli_epi32(w0, 2)));
  w1 = _mm_loadu_si128(compressed + 5);
  _mm_storeu_si128(out + 6,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 28),
                                                    _mm_slli_epi32(w1, 4))));
  w0 = _mm_loadu_si128(compressed + 6);
  _mm_storeu_si128(out + 7,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 22),
                                                    _mm_slli_epi32(w0, 10))));
  w1 = _mm_loadu_si128(compressed + 7);
  _mm_storeu_si128(out + 8,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 16),
                                                    _mm_slli_epi32(w1, 16))));
  w0 = _mm_loadu_si128(compressed + 8);
  _mm_storeu_si128(out + 9,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 10),
                                                    _mm_slli_epi32(w0, 22))));
  _mm_storeu_si128(out + 10, _mm_and_si128(mask, _mm_srli_epi32(w0, 4)));
  w1 = _mm_loadu_si128(compressed + 9);
  _mm_storeu_si128(out + 11,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 30),
                                                    _mm_slli_epi32(w1, 2))));
  w0 = _mm_loadu_si128(compressed + 10);
  _mm_storeu_si128(out + 12,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 24),
                                                    _mm_slli_epi32(w0, 8))));
  w1 = _mm_loadu_si128(compressed + 11);
  _mm_storeu_si128(out + 13,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 18),
                                                    _mm_slli_epi32(w1, 14))));
  w0 = _mm_loadu_si128(compressed + 12);
  _mm_storeu_si128(out + 14,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 12),
                                                    _mm_slli_epi32(w0, 20))));
  _mm_storeu_si128(out + 15, _mm_srli_epi32(w0, 6));
}

/* we packed 64 27-bit values, touching 14 128-bit words, using 224 bytes */
static void simdunpackblock27(const __m128i *compressed, u32 *pout) {
  /* we are going to access  14 128-bit words */
  __m128i w0, w1;
  __m128i *out = (__m128i *)pout;
  const __m128i mask = _mm_set1_epi32(134217727);
  w0 = _mm_loadu_si128(compressed);
  _mm_storeu_si128(out + 0, _mm_and_si128(mask, w0));
  w1 = _mm_loadu_si128(compressed + 1);
  _mm_storeu_si128(out + 1,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 27),
                                                    _mm_slli_epi32(w1, 5))));
  w0 = _mm_loadu_si128(compressed + 2);
  _mm_storeu_si128(out + 2,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 22),
                                                    _mm_slli_epi32(w0, 10))));
  w1 = _mm_loadu_si128(compressed + 3);
  _mm_storeu_si128(out + 3,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 17),
                                                    _mm_slli_epi32(w1, 15))));
  w0 = _mm_loadu_si128(compressed + 4);
  _mm_storeu_si128(out + 4,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 12),
                                                    _mm_slli_epi32(w0, 20))));
  w1 = _mm_loadu_si128(compressed + 5);
  _mm_storeu_si128(out + 5,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 7),
                                                    _mm_slli_epi32(w1, 25))));
  _mm_storeu_si128(out + 6, _mm_and_si128(mask, _mm_srli_epi32(w1, 2)));
  w0 = _mm_loadu_si128(compressed + 6);
  _mm_storeu_si128(out + 7,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 29),
                                                    _mm_slli_epi32(w0, 3))));
  w1 = _mm_loadu_si128(compressed + 7);
  _mm_storeu_si128(out + 8,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 24),
                                                    _mm_slli_epi32(w1, 8))));
  w0 = _mm_loadu_si128(compressed + 8);
  _mm_storeu_si128(out + 9,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 19),
                                                    _mm_slli_epi32(w0, 13))));
  w1 = _mm_loadu_si128(compressed + 9);
  _mm_storeu_si128(out + 10,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 14),
                                                    _mm_slli_epi32(w1, 18))));
  w0 = _mm_loadu_si128(compressed + 10);
  _mm_storeu_si128(out + 11,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 9),
                                                    _mm_slli_epi32(w0, 23))));
  _mm_storeu_si128(out + 12, _mm_and_si128(mask, _mm_srli_epi32(w0, 4)));
  w1 = _mm_loadu_si128(compressed + 11);
  _mm_storeu_si128(out + 13,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 31),
                                                    _mm_slli_epi32(w1, 1))));
  w0 = _mm_loadu_si128(compressed + 12);
  _mm_storeu_si128(out + 14,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 26),
                                                    _mm_slli_epi32(w0, 6))));
  w1 = _mm_loadu_si128(compressed + 13);
  _mm_storeu_si128(out + 15,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 21),
                                                    _mm_slli_epi32(w1, 11))));
}

/* we packed 64 28-bit values, touching 14 128-bit words, using 224 bytes */
static void simdunpackblock28(const __m128i *compressed, u32 *pout) {
  /* we are going to access  14 128-bit words */
  __m128i w0, w1;
  __m128i *out = (__m128i *)pout;
  const __m128i mask = _mm_set1_epi32(268435455);
  w0 = _mm_loadu_si128(compressed);
  _mm_storeu_si128(out + 0, _mm_and_si128(mask, w0));
  w1 = _mm_loadu_si128(compressed + 1);
  _mm_storeu_si128(out + 1,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 28),
                                                    _mm_slli_epi32(w1, 4))));
  w0 = _mm_loadu_si128(compressed + 2);
  _mm_storeu_si128(out + 2,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 24),
                                                    _mm_slli_epi32(w0, 8))));
  w1 = _mm_loadu_si128(compressed + 3);
  _mm_storeu_si128(out + 3,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 20),
                                                    _mm_slli_epi32(w1, 12))));
  w0 = _mm_loadu_si128(compressed + 4);
  _mm_storeu_si128(out + 4,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 16),
                                                    _mm_slli_epi32(w0, 16))));
  w1 = _mm_loadu_si128(compressed + 5);
  _mm_storeu_si128(out + 5,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 12),
                                                    _mm_slli_epi32(w1, 20))));
  w0 = _mm_loadu_si128(compressed + 6);
  _mm_storeu_si128(out + 6,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 8),
                                                    _mm_slli_epi32(w0, 24))));
  _mm_storeu_si128(out + 7, _mm_srli_epi32(w0, 4));
  w1 = _mm_loadu_si128(compressed + 7);
  _mm_storeu_si128(out + 8, _mm_and_si128(mask, w1));
  w0 = _mm_loadu_si128(compressed + 8);
  _mm_storeu_si128(out + 9,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 28),
                                                    _mm_slli_epi32(w0, 4))));
  w1 = _mm_loadu_si128(compressed + 9);
  _mm_storeu_si128(out + 10,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 24),
                                                    _mm_slli_epi32(w1, 8))));
  w0 = _mm_loadu_si128(compressed + 10);
  _mm_storeu_si128(out + 11,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 20),
                                                    _mm_slli_epi32(w0, 12))));
  w1 = _mm_loadu_si128(compressed + 11);
  _mm_storeu_si128(out + 12,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 16),
                                                    _mm_slli_epi32(w1, 16))));
  w0 = _mm_loadu_si128(compressed + 12);
  _mm_storeu_si128(out + 13,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 12),
                                                    _mm_slli_epi32(w0, 20))));
  w1 = _mm_loadu_si128(compressed + 13);
  _mm_storeu_si128(out + 14,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 8),
                                                    _mm_slli_epi32(w1, 24))));
  _mm_storeu_si128(out + 15, _mm_srli_epi32(w1, 4));
}

/* we packed 64 29-bit values, touching 15 128-bit words, using 240 bytes */
static void simdunpackblock29(const __m128i *compressed, u32 *pout) {
  /* we are going to access  15 128-bit words */
  __m128i w0, w1;
  __m128i *out = (__m128i *)pout;
  const __m128i mask = _mm_set1_epi32(536870911);
  w0 = _mm_loadu_si128(compressed);
  _mm_storeu_si128(out + 0, _mm_and_si128(mask, w0));
  w1 = _mm_loadu_si128(compressed + 1);
  _mm_storeu_si128(out + 1,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 29),
                                                    _mm_slli_epi32(w1, 3))));
  w0 = _mm_loadu_si128(compressed + 2);
  _mm_storeu_si128(out + 2,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 26),
                                                    _mm_slli_epi32(w0, 6))));
  w1 = _mm_loadu_si128(compressed + 3);
  _mm_storeu_si128(out + 3,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 23),
                                                    _mm_slli_epi32(w1, 9))));
  w0 = _mm_loadu_si128(compressed + 4);
  _mm_storeu_si128(out + 4,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 20),
                                                    _mm_slli_epi32(w0, 12))));
  w1 = _mm_loadu_si128(compressed + 5);
  _mm_storeu_si128(out + 5,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 17),
                                                    _mm_slli_epi32(w1, 15))));
  w0 = _mm_loadu_si128(compressed + 6);
  _mm_storeu_si128(out + 6,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 14),
                                                    _mm_slli_epi32(w0, 18))));
  w1 = _mm_loadu_si128(compressed + 7);
  _mm_storeu_si128(out + 7,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 11),
                                                    _mm_slli_epi32(w1, 21))));
  w0 = _mm_loadu_si128(compressed + 8);
  _mm_storeu_si128(out + 8,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 8),
                                                    _mm_slli_epi32(w0, 24))));
  w1 = _mm_loadu_si128(compressed + 9);
  _mm_storeu_si128(out + 9,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 5),
                                                    _mm_slli_epi32(w1, 27))));
  _mm_storeu_si128(out + 10, _mm_and_si128(mask, _mm_srli_epi32(w1, 2)));
  w0 = _mm_loadu_si128(compressed + 10);
  _mm_storeu_si128(out + 11,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 31),
                                                    _mm_slli_epi32(w0, 1))));
  w1 = _mm_loadu_si128(compressed + 11);
  _mm_storeu_si128(out + 12,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 28),
                                                    _mm_slli_epi32(w1, 4))));
  w0 = _mm_loadu_si128(compressed + 12);
  _mm_storeu_si128(out + 13,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 25),
                                                    _mm_slli_epi32(w0, 7))));
  w1 = _mm_loadu_si128(compressed + 13);
  _mm_storeu_si128(out + 14,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 22),
                                                    _mm_slli_epi32(w1, 10))));
  w0 = _mm_loadu_si128(compressed + 14);
  _mm_storeu_si128(out + 15,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 19),
                                                    _mm_slli_epi32(w0, 13))));
}

/* we packed 64 30-bit values, touching 15 128-bit words, using 240 bytes */
static void simdunpackblock30(const __m128i *compressed, u32 *pout) {
  /* we are going to access  15 128-bit words */
  __m128i w0, w1;
  __m128i *out = (__m128i *)pout;
  const __m128i mask = _mm_set1_epi32(1073741823);
  w0 = _mm_loadu_si128(compressed);
  _mm_storeu_si128(out + 0, _mm_and_si128(mask, w0));
  w1 = _mm_loadu_si128(compressed + 1);
  _mm_storeu_si128(out + 1,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 30),
                                                    _mm_slli_epi32(w1, 2))));
  w0 = _mm_loadu_si128(compressed + 2);
  _mm_storeu_si128(out + 2,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 28),
                                                    _mm_slli_epi32(w0, 4))));
  w1 = _mm_loadu_si128(compressed + 3);
  _mm_storeu_si128(out + 3,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 26),
                                                    _mm_slli_epi32(w1, 6))));
  w0 = _mm_loadu_si128(compressed + 4);
  _mm_storeu_si128(out + 4,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 24),
                                                    _mm_slli_epi32(w0, 8))));
  w1 = _mm_loadu_si128(compressed + 5);
  _mm_storeu_si128(out + 5,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 22),
                                                    _mm_slli_epi32(w1, 10))));
  w0 = _mm_loadu_si128(compressed + 6);
  _mm_storeu_si128(out + 6,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 20),
                                                    _mm_slli_epi32(w0, 12))));
  w1 = _mm_loadu_si128(compressed + 7);
  _mm_storeu_si128(out + 7,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 18),
                                                    _mm_slli_epi32(w1, 14))));
  w0 = _mm_loadu_si128(compressed + 8);
  _mm_storeu_si128(out + 8,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 16),
                                                    _mm_slli_epi32(w0, 16))));
  w1 = _mm_loadu_si128(compressed + 9);
  _mm_storeu_si128(out + 9,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 14),
                                                    _mm_slli_epi32(w1, 18))));
  w0 = _mm_loadu_si128(compressed + 10);
  _mm_storeu_si128(out + 10,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 12),
                                                    _mm_slli_epi32(w0, 20))));
  w1 = _mm_loadu_si128(compressed + 11);
  _mm_storeu_si128(out + 11,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 10),
                                                    _mm_slli_epi32(w1, 22))));
  w0 = _mm_loadu_si128(compressed + 12);
  _mm_storeu_si128(out + 12,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 8),
                                                    _mm_slli_epi32(w0, 24))));
  w1 = _mm_loadu_si128(compressed + 13);
  _mm_storeu_si128(out + 13,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 6),
                                                    _mm_slli_epi32(w1, 26))));
  w0 = _mm_loadu_si128(compressed + 14);
  _mm_storeu_si128(out + 14,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 4),
                                                    _mm_slli_epi32(w0, 28))));
  _mm_storeu_si128(out + 15, _mm_srli_epi32(w0, 2));
}

/* we packed 64 31-bit values, touching 16 128-bit words, using 256 bytes */
static void simdunpackblock31(const __m128i *compressed, u32 *pout) {
  /* we are going to access  16 128-bit words */
  __m128i w0, w1;
  __m128i *out = (__m128i *)pout;
  const __m128i mask = _mm_set1_epi32(2147483647);
  w0 = _mm_loadu_si128(compressed);
  _mm_storeu_si128(out + 0, _mm_and_si128(mask, w0));
  w1 = _mm_loadu_si128(compressed + 1);
  _mm_storeu_si128(out + 1,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 31),
                                                    _mm_slli_epi32(w1, 1))));
  w0 = _mm_loadu_si128(compressed + 2);
  _mm_storeu_si128(out + 2,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 30),
                                                    _mm_slli_epi32(w0, 2))));
  w1 = _mm_loadu_si128(compressed + 3);
  _mm_storeu_si128(out + 3,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 29),
                                                    _mm_slli_epi32(w1, 3))));
  w0 = _mm_loadu_si128(compressed + 4);
  _mm_storeu_si128(out + 4,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 28),
                                                    _mm_slli_epi32(w0, 4))));
  w1 = _mm_loadu_si128(compressed + 5);
  _mm_storeu_si128(out + 5,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 27),
                                                    _mm_slli_epi32(w1, 5))));
  w0 = _mm_loadu_si128(compressed + 6);
  _mm_storeu_si128(out + 6,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 26),
                                                    _mm_slli_epi32(w0, 6))));
  w1 = _mm_loadu_si128(compressed + 7);
  _mm_storeu_si128(out + 7,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 25),
                                                    _mm_slli_epi32(w1, 7))));
  w0 = _mm_loadu_si128(compressed + 8);
  _mm_storeu_si128(out + 8,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 24),
                                                    _mm_slli_epi32(w0, 8))));
  w1 = _mm_loadu_si128(compressed + 9);
  _mm_storeu_si128(out + 9,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 23),
                                                    _mm_slli_epi32(w1, 9))));
  w0 = _mm_loadu_si128(compressed + 10);
  _mm_storeu_si128(out + 10,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 22),
                                                    _mm_slli_epi32(w0, 10))));
  w1 = _mm_loadu_si128(compressed + 11);
  _mm_storeu_si128(out + 11,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 21),
                                                    _mm_slli_epi32(w1, 11))));
  w0 = _mm_loadu_si128(compressed + 12);
  _mm_storeu_si128(out + 12,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 20),
                                                    _mm_slli_epi32(w0, 12))));
  w1 = _mm_loadu_si128(compressed + 13);
  _mm_storeu_si128(out + 13,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 19),
                                                    _mm_slli_epi32(w1, 13))));
  w0 = _mm_loadu_si128(compressed + 14);
  _mm_storeu_si128(out + 14,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w1, 18),
                                                    _mm_slli_epi32(w0, 14))));
  w1 = _mm_loadu_si128(compressed + 15);
  _mm_storeu_si128(out + 15,
                   _mm_and_si128(mask, _mm_or_si128(_mm_srli_epi32(w0, 17),
                                                    _mm_slli_epi32(w1, 15))));
}

/* we packed 64 32-bit values, touching 16 128-bit words, using 256 bytes */
static void simdunpackblock32(const __m128i *compressed, u32 *pout) {
  /* we are going to access  16 128-bit words */
  __m128i w0, w1;
  __m128i *out = (__m128i *)pout;
  w0 = _mm_loadu_si128(compressed);
  _mm_storeu_si128(out + 0, w0);
  w1 = _mm_loadu_si128(compressed + 1);
  _mm_storeu_si128(out + 1, w1);
  w0 = _mm_loadu_si128(compressed + 2);
  _mm_storeu_si128(out + 2, w0);
  w1 = _mm_loadu_si128(compressed + 3);
  _mm_storeu_si128(out + 3, w1);
  w0 = _mm_loadu_si128(compressed + 4);
  _mm_storeu_si128(out + 4, w0);
  w1 = _mm_loadu_si128(compressed + 5);
  _mm_storeu_si128(out + 5, w1);
  w0 = _mm_loadu_si128(compressed + 6);
  _mm_storeu_si128(out + 6, w0);
  w1 = _mm_loadu_si128(compressed + 7);
  _mm_storeu_si128(out + 7, w1);
  w0 = _mm_loadu_si128(compressed + 8);
  _mm_storeu_si128(out + 8, w0);
  w1 = _mm_loadu_si128(compressed + 9);
  _mm_storeu_si128(out + 9, w1);
  w0 = _mm_loadu_si128(compressed + 10);
  _mm_storeu_si128(out + 10, w0);
  w1 = _mm_loadu_si128(compressed + 11);
  _mm_storeu_si128(out + 11, w1);
  w0 = _mm_loadu_si128(compressed + 12);
  _mm_storeu_si128(out + 12, w0);
  w1 = _mm_loadu_si128(compressed + 13);
  _mm_storeu_si128(out + 13, w1);
  w0 = _mm_loadu_si128(compressed + 14);
  _mm_storeu_si128(out + 14, w0);
  w1 = _mm_loadu_si128(compressed + 15);
  _mm_storeu_si128(out + 15, w1);
}

static simdpackblockfnc simdfuncPackArr[] = {
    &simdpackblock0,  &simdpackblock1,  &simdpackblock2,  &simdpackblock3,
    &simdpackblock4,  &simdpackblock5,  &simdpackblock6,  &simdpackblock7,
    &simdpackblock8,  &simdpackblock9,  &simdpackblock10, &simdpackblock11,
    &simdpackblock12, &simdpackblock13, &simdpackblock14, &simdpackblock15,
    &simdpackblock16, &simdpackblock17, &simdpackblock18, &simdpackblock19,
    &simdpackblock20, &simdpackblock21, &simdpackblock22, &simdpackblock23,
    &simdpackblock24, &simdpackblock25, &simdpackblock26, &simdpackblock27,
    &simdpackblock28, &simdpackblock29, &simdpackblock30, &simdpackblock31,
    &simdpackblock32};
static simdpackblockfnc simdfuncPackMaskArr[] = {
    &simdpackblockmask0,  &simdpackblockmask1,  &simdpackblockmask2,
    &simdpackblockmask3,  &simdpackblockmask4,  &simdpackblockmask5,
    &simdpackblockmask6,  &simdpackblockmask7,  &simdpackblockmask8,
    &simdpackblockmask9,  &simdpackblockmask10, &simdpackblockmask11,
    &simdpackblockmask12, &simdpackblockmask13, &simdpackblockmask14,
    &simdpackblockmask15, &simdpackblockmask16, &simdpackblockmask17,
    &simdpackblockmask18, &simdpackblockmask19, &simdpackblockmask20,
    &simdpackblockmask21, &simdpackblockmask22, &simdpackblockmask23,
    &simdpackblockmask24, &simdpackblockmask25, &simdpackblockmask26,
    &simdpackblockmask27, &simdpackblockmask28, &simdpackblockmask29,
    &simdpackblockmask30, &simdpackblockmask31, &simdpackblockmask32};
static simdunpackblockfnc simdfuncUnpackArr[] = {
    &simdunpackblock0,  &simdunpackblock1,  &simdunpackblock2,
    &simdunpackblock3,  &simdunpackblock4,  &simdunpackblock5,
    &simdunpackblock6,  &simdunpackblock7,  &simdunpackblock8,
    &simdunpackblock9,  &simdunpackblock10, &simdunpackblock11,
    &simdunpackblock12, &simdunpackblock13, &simdunpackblock14,
    &simdunpackblock15, &simdunpackblock16, &simdunpackblock17,
    &simdunpackblock18, &simdunpackblock19, &simdunpackblock20,
    &simdunpackblock21, &simdunpackblock22, &simdunpackblock23,
    &simdunpackblock24, &simdunpackblock25, &simdunpackblock26,
    &simdunpackblock27, &simdunpackblock28, &simdunpackblock29,
    &simdunpackblock30, &simdunpackblock31, &simdunpackblock32};
//---------------------------------------------------------------------------
void pack(const u32 *in, __m128i *out, const u8 bit) {
  simdfuncPackArr[bit](in, out);
}
//---------------------------------------------------------------------------
void packmask(const u32 *in, __m128i *out, const u8 bit) {
  simdfuncPackMaskArr[bit](in, out);
}
//---------------------------------------------------------------------------
void unpack(const __m128i *in, u32 *out, const u8 bit) {
  simdfuncUnpackArr[bit](in, out);
}
//---------------------------------------------------------------------------
} // namespace block64
//---------------------------------------------------------------------------
} // namespace simd32
//---------------------------------------------------------------------------
} // namespace bitpacking
//---------------------------------------------------------------------------
} // namespace compression