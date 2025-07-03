#include <cstring>
#include <simdcomp.h>
//---------------------------------------------------------------------------
#include "Block256BitPacking.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
namespace bitpacking {
//---------------------------------------------------------------------------
namespace simd32 {
//---------------------------------------------------------------------------
namespace avx {
//---------------------------------------------------------------------------

void pack(const u32 *in, __m256i *out, const u8 bit) {
  return avxpackwithoutmask(in, out, bit);
}

void packmask(const u32 *in, __m256i *out, const u8 bit) {
  return avxpack(in, out, bit);
}

void unpack(const __m256i *in, u32 *out, const u8 bit) {
  avxunpack(in, out, bit);
}

static void filtereq0(const __m256i *in, u32 *matches, const INTEGER comp) {
  if (comp == 0)
    memset(matches, 1, 256 * sizeof(*matches));
  else
    memset(matches, 0, 256 * sizeof(*matches));
}

/* we packed 256 1-bit values, touching 1 256-bit words, using 16 bytes */
static void filtereq1(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  1 256-bit word */
  __m256i w0;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(1);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 1)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 2)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 3)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 5)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 6)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 7)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 9)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 10)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 11)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 12)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 13)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 14)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 15)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 17)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 18)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 19)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 20)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 21)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 22)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 23)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 24)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 25)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 26)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 27)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 28)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 29)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 30)),
                         broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 31), broadcomp));
}

/* we packed 256 2-bit values, touching 2 256-bit words, using 32 bytes */
static void filtereq2(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  2 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(3);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 2)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 6)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 10)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 12)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 14)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 18)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 20)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 22)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 24)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 26)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 28)),
                         broadcomp));
  _mm256_storeu_si256(out + 15,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 30), broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 16, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp));
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 2)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 6)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 10)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 12)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 14)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 18)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 20)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 22)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 24)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 26)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 28)),
                         broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 30), broadcomp));
}

/* we packed 256 3-bit values, touching 3 256-bit words, using 48 bytes */
static void filtereq3(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  3 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(7);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 3)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 6)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 9)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 12)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 15)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 18)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 21)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 24)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 27)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                 _mm256_slli_epi32(w1, 2))),
          broadcomp));
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 1)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 7)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 10)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 13)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 19)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 22)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 25)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 28)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 31),
                                                 _mm256_slli_epi32(w0, 1))),
          broadcomp));
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 2)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 5)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 11)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 14)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 17)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 20)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 23)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 26)),
                         broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 29), broadcomp));
}

/* we packed 256 4-bit values, touching 4 256-bit words, using 64 bytes */
static void filtereq4(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  4 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(15);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 12)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 20)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 24)),
                         broadcomp));
  _mm256_storeu_si256(out + 7,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 28), broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 8, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp));
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 12)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 20)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 24)),
                         broadcomp));
  _mm256_storeu_si256(out + 15,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 28), broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 16, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 12)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 20)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 24)),
                         broadcomp));
  _mm256_storeu_si256(out + 23,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 28), broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 24, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp));
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 12)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 20)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 24)),
                         broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 28), broadcomp));
}

/* we packed 256 5-bit values, touching 5 256-bit words, using 80 bytes */
static void filtereq5(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  5 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(31);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 5)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 10)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 15)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 20)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 25)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                 _mm256_slli_epi32(w1, 2))),
          broadcomp));
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 3)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 13)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 18)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 23)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4))),
          broadcomp));
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 1)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 6)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 11)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 21)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 26)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 31),
                                                 _mm256_slli_epi32(w1, 1))),
          broadcomp));
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 9)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 14)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 19)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 24)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 29),
                                                 _mm256_slli_epi32(w0, 3))),
          broadcomp));
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 2)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 7)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 12)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 17)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 22)),
                         broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 27), broadcomp));
}

/* we packed 256 6-bit values, touching 6 256-bit words, using 96 bytes */
static void filtereq6(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  6 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(63);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 6)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 12)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 18)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 24)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                 _mm256_slli_epi32(w1, 2))),
          broadcomp));
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 10)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 22)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4))),
          broadcomp));
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 2)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 14)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 20)),
                         broadcomp));
  _mm256_storeu_si256(out + 15,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 26), broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 16, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp));
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 6)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 12)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 18)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 24)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                 _mm256_slli_epi32(w0, 2))),
          broadcomp));
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 10)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 22)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4))),
          broadcomp));
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 2)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 14)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 20)),
                         broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 26), broadcomp));
}

/* we packed 256 7-bit values, touching 7 256-bit words, using 112 bytes */
static void filtereq7(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  7 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(127);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 7)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 14)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 21)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4))),
          broadcomp));
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 3)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 10)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 17)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 24)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 31),
                                                 _mm256_slli_epi32(w0, 1))),
          broadcomp));
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 6)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 13)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 20)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 27),
                                                 _mm256_slli_epi32(w1, 5))),
          broadcomp));
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 2)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 9)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 23)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                 _mm256_slli_epi32(w0, 2))),
          broadcomp));
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 5)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 12)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 19)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 26),
                                                 _mm256_slli_epi32(w1, 6))),
          broadcomp));
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 1)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 15)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 22)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 29),
                                                 _mm256_slli_epi32(w0, 3))),
          broadcomp));
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 11)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 18)),
                         broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 25), broadcomp));
}

/* we packed 256 8-bit values, touching 8 256-bit words, using 128 bytes */
static void filtereq8(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  8 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(255);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                         broadcomp));
  _mm256_storeu_si256(out + 3,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 24), broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 4, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp));
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                         broadcomp));
  _mm256_storeu_si256(out + 7,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 24), broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 8, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                         broadcomp));
  _mm256_storeu_si256(out + 11,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 24), broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 12, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp));
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                         broadcomp));
  _mm256_storeu_si256(out + 15,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 24), broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 16, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                         broadcomp));
  _mm256_storeu_si256(out + 19,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 24), broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 20, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp));
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                         broadcomp));
  _mm256_storeu_si256(out + 23,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 24), broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 24, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                         broadcomp));
  _mm256_storeu_si256(out + 27,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 24), broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 28, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp));
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                         broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 24), broadcomp));
}

/* we packed 256 9-bit values, touching 9 256-bit words, using 144 bytes */
static void filtereq9(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  9 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(511);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 9)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 18)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 27),
                                                 _mm256_slli_epi32(w1, 5))),
          broadcomp));
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 13)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 22)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 31),
                                                 _mm256_slli_epi32(w0, 1))),
          broadcomp));
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 17)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 26),
                                                 _mm256_slli_epi32(w1, 6))),
          broadcomp));
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 3)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 12)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 21)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                 _mm256_slli_epi32(w0, 2))),
          broadcomp));
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 7)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 25),
                                                 _mm256_slli_epi32(w1, 7))),
          broadcomp));
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 2)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 11)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 20)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 29),
                                                 _mm256_slli_epi32(w0, 3))),
          broadcomp));
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 6)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 15)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8))),
          broadcomp));
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 1)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 10)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 19)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4))),
          broadcomp));
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 5)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 14)),
                         broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 23), broadcomp));
}

/* we packed 256 10-bit values, touching 10 256-bit words, using 160 bytes */
static void filtereq10(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  10 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(1023);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 10)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 20)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                 _mm256_slli_epi32(w1, 2))),
          broadcomp));
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 18)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4))),
          broadcomp));
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 6)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 26),
                                                 _mm256_slli_epi32(w1, 6))),
          broadcomp));
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 14)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8))),
          broadcomp));
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 2)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 12)),
                         broadcomp));
  _mm256_storeu_si256(out + 15,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 22), broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 16, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp));
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 10)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 20)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                 _mm256_slli_epi32(w0, 2))),
          broadcomp));
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 18)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4))),
          broadcomp));
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 6)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                 _mm256_slli_epi32(w0, 6))),
          broadcomp));
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 14)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8))),
          broadcomp));
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 2)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 12)),
                         broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 22), broadcomp));
}

/* we packed 256 11-bit values, touching 11 256-bit words, using 176 bytes */
static void filtereq11(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  11 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(2047);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 11)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 22),
                                                 _mm256_slli_epi32(w1, 10))),
          broadcomp));
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 1)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 12)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 23),
                                                 _mm256_slli_epi32(w0, 9))),
          broadcomp));
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 2)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 13)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8))),
          broadcomp));
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 3)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 14)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 25),
                                                 _mm256_slli_epi32(w0, 7))),
          broadcomp));
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 15)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 26),
                                                 _mm256_slli_epi32(w1, 6))),
          broadcomp));
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 5)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 27),
                                                 _mm256_slli_epi32(w0, 5))),
          broadcomp));
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 6)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 17)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4))),
          broadcomp));
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 7)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 18)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 29),
                                                 _mm256_slli_epi32(w0, 3))),
          broadcomp));
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 19)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                 _mm256_slli_epi32(w1, 2))),
          broadcomp));
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 9)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 20)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 31),
                                                 _mm256_slli_epi32(w0, 1))),
          broadcomp));
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 10)),
                         broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 21), broadcomp));
}

/* we packed 256 12-bit values, touching 12 256-bit words, using 192 bytes */
static void filtereq12(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  12 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(4095);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 12)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8))),
          broadcomp));
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4))),
          broadcomp));
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                         broadcomp));
  _mm256_storeu_si256(out + 7,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 20), broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 8, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp));
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 12)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8))),
          broadcomp));
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4))),
          broadcomp));
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                         broadcomp));
  _mm256_storeu_si256(out + 15,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 20), broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 16, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 12)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8))),
          broadcomp));
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4))),
          broadcomp));
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                         broadcomp));
  _mm256_storeu_si256(out + 23,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 20), broadcomp));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 24, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp));
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 12)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8))),
          broadcomp));
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4))),
          broadcomp));
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                         broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 20), broadcomp));
}

/* we packed 256 13-bit values, touching 13 256-bit words, using 208 bytes */
static void filtereq13(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  13 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(8191);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 13)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 26),
                                                 _mm256_slli_epi32(w1, 6))),
          broadcomp));
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 7)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12))),
          broadcomp));
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 1)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 14)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 27),
                                                 _mm256_slli_epi32(w1, 5))),
          broadcomp));
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 21),
                                                 _mm256_slli_epi32(w0, 11))),
          broadcomp));
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 2)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 15)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4))),
          broadcomp));
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 9)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 22),
                                                 _mm256_slli_epi32(w0, 10))),
          broadcomp));
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 3)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 29),
                                                 _mm256_slli_epi32(w1, 3))),
          broadcomp));
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 10)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 23),
                                                 _mm256_slli_epi32(w0, 9))),
          broadcomp));
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 17)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                 _mm256_slli_epi32(w1, 2))),
          broadcomp));
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 11)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8))),
          broadcomp));
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 5)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 18)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 31),
                                                 _mm256_slli_epi32(w1, 1))),
          broadcomp));
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 12)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 25),
                                                 _mm256_slli_epi32(w0, 7))),
          broadcomp));
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 6)),
                         broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 19), broadcomp));
}

/* we packed 256 14-bit values, touching 14 256-bit words, using 224 bytes */
static void filtereq14(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  14 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(16383);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 14)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4))),
          broadcomp));
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 10)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8))),
          broadcomp));
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 6)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                 _mm256_slli_epi32(w1, 12))),
          broadcomp));
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 2)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                 _mm256_slli_epi32(w0, 2))),
          broadcomp));
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 12)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 26),
                                                 _mm256_slli_epi32(w1, 6))),
          broadcomp));
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 22),
                                                 _mm256_slli_epi32(w0, 10))),
          broadcomp));
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                         broadcomp));
  _mm256_storeu_si256(out + 15,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 18), broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 16, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp));
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 14)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4))),
          broadcomp));
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 10)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8))),
          broadcomp));
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 6)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12))),
          broadcomp));
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 2)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                 _mm256_slli_epi32(w1, 2))),
          broadcomp));
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 12)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                 _mm256_slli_epi32(w0, 6))),
          broadcomp));
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 22),
                                                 _mm256_slli_epi32(w1, 10))),
          broadcomp));
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                         broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 18), broadcomp));
}

/* we packed 256 15-bit values, touching 15 256-bit words, using 240 bytes */
static void filtereq15(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  15 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(32767);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 15)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                 _mm256_slli_epi32(w1, 2))),
          broadcomp));
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 13)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4))),
          broadcomp));
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 11)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 26),
                                                 _mm256_slli_epi32(w1, 6))),
          broadcomp));
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 9)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8))),
          broadcomp));
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 7)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 22),
                                                 _mm256_slli_epi32(w1, 10))),
          broadcomp));
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 5)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12))),
          broadcomp));
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 3)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 18),
                                                 _mm256_slli_epi32(w1, 14))),
          broadcomp));
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 1)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 31),
                                                 _mm256_slli_epi32(w0, 1))),
          broadcomp));
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 14)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 29),
                                                 _mm256_slli_epi32(w1, 3))),
          broadcomp));
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 12)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 27),
                                                 _mm256_slli_epi32(w0, 5))),
          broadcomp));
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 10)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 25),
                                                 _mm256_slli_epi32(w1, 7))),
          broadcomp));
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 23),
                                                 _mm256_slli_epi32(w0, 9))),
          broadcomp));
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 6)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 21),
                                                 _mm256_slli_epi32(w1, 11))),
          broadcomp));
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 19),
                                                 _mm256_slli_epi32(w0, 13))),
          broadcomp));
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 2)),
                         broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 17), broadcomp));
}

/* we packed 256 16-bit values, touching 16 256-bit words, using 256 bytes */
static void filtereq16(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  16 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(65535);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(out + 1,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 16), broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 2, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp));
  _mm256_storeu_si256(out + 3,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 16), broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 4, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(out + 5,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 16), broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 6, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp));
  _mm256_storeu_si256(out + 7,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 16), broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 8, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(out + 9,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 16), broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 10, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp));
  _mm256_storeu_si256(out + 11,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 16), broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 12, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(out + 13,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 16), broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 14, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp));
  _mm256_storeu_si256(out + 15,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 16), broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 16, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(out + 17,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 16), broadcomp));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 18, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp));
  _mm256_storeu_si256(out + 19,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 16), broadcomp));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 20, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(out + 21,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 16), broadcomp));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 22, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp));
  _mm256_storeu_si256(out + 23,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 16), broadcomp));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 24, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(out + 25,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 16), broadcomp));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 26, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp));
  _mm256_storeu_si256(out + 27,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 16), broadcomp));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 28, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(out + 29,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 16), broadcomp));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 30, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 16), broadcomp));
}

/* we packed 256 17-bit values, touching 17 256-bit words, using 272 bytes */
static void filtereq17(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  17 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(131071);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 17),
                                                 _mm256_slli_epi32(w1, 15))),
          broadcomp));
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 2)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 19),
                                                 _mm256_slli_epi32(w0, 13))),
          broadcomp));
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 21),
                                                 _mm256_slli_epi32(w1, 11))),
          broadcomp));
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 6)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 23),
                                                 _mm256_slli_epi32(w0, 9))),
          broadcomp));
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 25),
                                                 _mm256_slli_epi32(w1, 7))),
          broadcomp));
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 10)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 27),
                                                 _mm256_slli_epi32(w0, 5))),
          broadcomp));
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 12)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 29),
                                                 _mm256_slli_epi32(w1, 3))),
          broadcomp));
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 14)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 31),
                                                 _mm256_slli_epi32(w0, 1))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16))),
          broadcomp));
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 1)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 18),
                                                 _mm256_slli_epi32(w0, 14))),
          broadcomp));
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 3)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                 _mm256_slli_epi32(w1, 12))),
          broadcomp));
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 5)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 22),
                                                 _mm256_slli_epi32(w0, 10))),
          broadcomp));
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 7)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8))),
          broadcomp));
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 9)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                 _mm256_slli_epi32(w0, 6))),
          broadcomp));
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 11)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4))),
          broadcomp));
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 13)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                 _mm256_slli_epi32(w0, 2))),
          broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 15), broadcomp));
}

/* we packed 256 18-bit values, touching 18 256-bit words, using 288 bytes */
static void filtereq18(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  18 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(262143);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 18),
                                                 _mm256_slli_epi32(w1, 14))),
          broadcomp));
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 22),
                                                 _mm256_slli_epi32(w0, 10))),
          broadcomp));
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 26),
                                                 _mm256_slli_epi32(w1, 6))),
          broadcomp));
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 12)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                 _mm256_slli_epi32(w0, 2))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16))),
          broadcomp));
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 2)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12))),
          broadcomp));
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 6)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8))),
          broadcomp));
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 10)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4))),
          broadcomp));
  _mm256_storeu_si256(out + 15,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 14), broadcomp));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 16, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 18),
                                                 _mm256_slli_epi32(w0, 14))),
          broadcomp));
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 22),
                                                 _mm256_slli_epi32(w1, 10))),
          broadcomp));
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                 _mm256_slli_epi32(w0, 6))),
          broadcomp));
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 12)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                 _mm256_slli_epi32(w1, 2))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16))),
          broadcomp));
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 2)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                 _mm256_slli_epi32(w1, 12))),
          broadcomp));
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 6)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8))),
          broadcomp));
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 10)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4))),
          broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 14), broadcomp));
}

/* we packed 256 19-bit values, touching 19 256-bit words, using 304 bytes */
static void filtereq19(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  19 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(524287);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 19),
                                                 _mm256_slli_epi32(w1, 13))),
          broadcomp));
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 6)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 25),
                                                 _mm256_slli_epi32(w0, 7))),
          broadcomp));
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 12)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 31),
                                                 _mm256_slli_epi32(w1, 1))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 18),
                                                 _mm256_slli_epi32(w0, 14))),
          broadcomp));
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 5)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8))),
          broadcomp));
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 11)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                 _mm256_slli_epi32(w0, 2))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 17),
                                                 _mm256_slli_epi32(w1, 15))),
          broadcomp));
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 23),
                                                 _mm256_slli_epi32(w0, 9))),
          broadcomp));
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 10)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 29),
                                                 _mm256_slli_epi32(w1, 3))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16))),
          broadcomp));
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 3)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 22),
                                                 _mm256_slli_epi32(w1, 10))),
          broadcomp));
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 9)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 15),
                                                 _mm256_slli_epi32(w1, 17))),
          broadcomp));
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 2)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 21),
                                                 _mm256_slli_epi32(w0, 11))),
          broadcomp));
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 27),
                                                 _mm256_slli_epi32(w1, 5))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 14),
                                                 _mm256_slli_epi32(w0, 18))),
          broadcomp));
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 1)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                 _mm256_slli_epi32(w1, 12))),
          broadcomp));
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 7)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                 _mm256_slli_epi32(w0, 6))),
          broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 13), broadcomp));
}

/* we packed 256 20-bit values, touching 20 256-bit words, using 320 bytes */
static void filtereq20(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  20 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(1048575);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                 _mm256_slli_epi32(w1, 12))),
          broadcomp));
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16))),
          broadcomp));
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8))),
          broadcomp));
  _mm256_storeu_si256(out + 7,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 12), broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 8, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12))),
          broadcomp));
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16))),
          broadcomp));
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8))),
          broadcomp));
  _mm256_storeu_si256(out + 15,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 12), broadcomp));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 16, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                 _mm256_slli_epi32(w1, 12))),
          broadcomp));
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16))),
          broadcomp));
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8))),
          broadcomp));
  _mm256_storeu_si256(out + 23,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 12), broadcomp));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 24, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12))),
          broadcomp));
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16))),
          broadcomp));
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8))),
          broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 12), broadcomp));
}

/* we packed 256 21-bit values, touching 21 256-bit words, using 336 bytes */
static void filtereq21(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  21 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(2097151);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 21),
                                                 _mm256_slli_epi32(w1, 11))),
          broadcomp));
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 10)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 31),
                                                 _mm256_slli_epi32(w0, 1))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                 _mm256_slli_epi32(w1, 12))),
          broadcomp));
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 9)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                 _mm256_slli_epi32(w0, 2))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 19),
                                                 _mm256_slli_epi32(w1, 13))),
          broadcomp));
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 29),
                                                 _mm256_slli_epi32(w0, 3))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 18),
                                                 _mm256_slli_epi32(w1, 14))),
          broadcomp));
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 7)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 17),
                                                 _mm256_slli_epi32(w1, 15))),
          broadcomp));
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 6)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 27),
                                                 _mm256_slli_epi32(w0, 5))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16))),
          broadcomp));
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 5)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                 _mm256_slli_epi32(w0, 6))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 15),
                                                 _mm256_slli_epi32(w1, 17))),
          broadcomp));
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 25),
                                                 _mm256_slli_epi32(w0, 7))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 14),
                                                 _mm256_slli_epi32(w1, 18))),
          broadcomp));
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 3)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 13),
                                                 _mm256_slli_epi32(w1, 19))),
          broadcomp));
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 2)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 23),
                                                 _mm256_slli_epi32(w0, 9))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 12),
                                                 _mm256_slli_epi32(w1, 20))),
          broadcomp));
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 1)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 22),
                                                 _mm256_slli_epi32(w0, 10))),
          broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 11), broadcomp));
}

/* we packed 256 22-bit values, touching 22 256-bit words, using 352 bytes */
static void filtereq22(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  22 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(4194303);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 22),
                                                 _mm256_slli_epi32(w1, 10))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 12),
                                                 _mm256_slli_epi32(w0, 20))),
          broadcomp));
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 2)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 14),
                                                 _mm256_slli_epi32(w0, 18))),
          broadcomp));
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 26),
                                                 _mm256_slli_epi32(w1, 6))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16))),
          broadcomp));
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 6)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 18),
                                                 _mm256_slli_epi32(w0, 14))),
          broadcomp));
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                 _mm256_slli_epi32(w1, 2))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12))),
          broadcomp));
  _mm256_storeu_si256(out + 15,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 10), broadcomp));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 16, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 22),
                                                 _mm256_slli_epi32(w0, 10))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 12),
                                                 _mm256_slli_epi32(w1, 20))),
          broadcomp));
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 2)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 14),
                                                 _mm256_slli_epi32(w1, 18))),
          broadcomp));
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                 _mm256_slli_epi32(w0, 6))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16))),
          broadcomp));
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 6)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 18),
                                                 _mm256_slli_epi32(w1, 14))),
          broadcomp));
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                 _mm256_slli_epi32(w0, 2))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 21);
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                 _mm256_slli_epi32(w1, 12))),
          broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 10), broadcomp));
}

/* we packed 256 23-bit values, touching 23 256-bit words, using 368 bytes */
static void filtereq23(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  23 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(8388607);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 23),
                                                 _mm256_slli_epi32(w1, 9))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 14),
                                                 _mm256_slli_epi32(w0, 18))),
          broadcomp));
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 5)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 19),
                                                 _mm256_slli_epi32(w0, 13))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 10),
                                                 _mm256_slli_epi32(w1, 22))),
          broadcomp));
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 1)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 15),
                                                 _mm256_slli_epi32(w1, 17))),
          broadcomp));
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 6)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 29),
                                                 _mm256_slli_epi32(w0, 3))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                 _mm256_slli_epi32(w1, 12))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 11),
                                                 _mm256_slli_epi32(w0, 21))),
          broadcomp));
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 2)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 25),
                                                 _mm256_slli_epi32(w1, 7))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16))),
          broadcomp));
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 7)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                 _mm256_slli_epi32(w1, 2))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 21),
                                                 _mm256_slli_epi32(w0, 11))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 12),
                                                 _mm256_slli_epi32(w1, 20))),
          broadcomp));
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 3)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                 _mm256_slli_epi32(w0, 6))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 17),
                                                 _mm256_slli_epi32(w1, 15))),
          broadcomp));
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 31),
                                                 _mm256_slli_epi32(w0, 1))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 22),
                                                 _mm256_slli_epi32(w1, 10))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 13),
                                                 _mm256_slli_epi32(w0, 19))),
          broadcomp));
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 21);
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 27),
                                                 _mm256_slli_epi32(w1, 5))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 22);
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 18),
                                                 _mm256_slli_epi32(w0, 14))),
          broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 9), broadcomp));
}

/* we packed 256 24-bit values, touching 24 256-bit words, using 384 bytes */
static void filtereq24(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  24 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(16777215);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16))),
          broadcomp));
  _mm256_storeu_si256(out + 3,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 8), broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 4, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16))),
          broadcomp));
  _mm256_storeu_si256(out + 7,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 8), broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 8, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16))),
          broadcomp));
  _mm256_storeu_si256(out + 11,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 8), broadcomp));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 12, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16))),
          broadcomp));
  _mm256_storeu_si256(out + 15,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 8), broadcomp));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 16, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16))),
          broadcomp));
  _mm256_storeu_si256(out + 19,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 8), broadcomp));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 20, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16))),
          broadcomp));
  _mm256_storeu_si256(out + 23,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 8), broadcomp));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 24, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16))),
          broadcomp));
  _mm256_storeu_si256(out + 27,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 8), broadcomp));
  w1 = _mm256_lddqu_si256(in + 21);
  _mm256_storeu_si256(
      out + 28, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp));
  w0 = _mm256_lddqu_si256(in + 22);
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 23);
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16))),
          broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 8), broadcomp));
}

/* we packed 256 25-bit values, touching 25 256-bit words, using 400 bytes */
static void filtereq25(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  25 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(33554431);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 25),
                                                 _mm256_slli_epi32(w1, 7))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 18),
                                                 _mm256_slli_epi32(w0, 14))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 11),
                                                 _mm256_slli_epi32(w1, 21))),
          broadcomp));
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 29),
                                                 _mm256_slli_epi32(w0, 3))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 22),
                                                 _mm256_slli_epi32(w1, 10))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 15),
                                                 _mm256_slli_epi32(w0, 17))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 8),
                                                 _mm256_slli_epi32(w1, 24))),
          broadcomp));
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 1)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                 _mm256_slli_epi32(w0, 6))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 19),
                                                 _mm256_slli_epi32(w1, 13))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 12),
                                                 _mm256_slli_epi32(w0, 20))),
          broadcomp));
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 5)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                 _mm256_slli_epi32(w1, 2))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 23),
                                                 _mm256_slli_epi32(w0, 9))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 9),
                                                 _mm256_slli_epi32(w0, 23))),
          broadcomp));
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 2)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 27),
                                                 _mm256_slli_epi32(w1, 5))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 13),
                                                 _mm256_slli_epi32(w1, 19))),
          broadcomp));
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 6)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 31),
                                                 _mm256_slli_epi32(w0, 1))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 17),
                                                 _mm256_slli_epi32(w0, 15))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 21);
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 10),
                                                 _mm256_slli_epi32(w1, 22))),
          broadcomp));
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 3)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 22);
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 23);
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 21),
                                                 _mm256_slli_epi32(w1, 11))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 24);
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 14),
                                                 _mm256_slli_epi32(w0, 18))),
          broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 7), broadcomp));
}

/* we packed 256 26-bit values, touching 26 256-bit words, using 416 bytes */
static void filtereq26(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  26 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(67108863);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 26),
                                                 _mm256_slli_epi32(w1, 6))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 14),
                                                 _mm256_slli_epi32(w1, 18))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 8),
                                                 _mm256_slli_epi32(w0, 24))),
          broadcomp));
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 2)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 22),
                                                 _mm256_slli_epi32(w0, 10))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 10),
                                                 _mm256_slli_epi32(w0, 22))),
          broadcomp));
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                 _mm256_slli_epi32(w1, 2))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 18),
                                                 _mm256_slli_epi32(w1, 14))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 12),
                                                 _mm256_slli_epi32(w0, 20))),
          broadcomp));
  _mm256_storeu_si256(out + 15,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 6), broadcomp));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 16, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                 _mm256_slli_epi32(w0, 6))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                 _mm256_slli_epi32(w1, 12))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 14),
                                                 _mm256_slli_epi32(w0, 18))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 8),
                                                 _mm256_slli_epi32(w1, 24))),
          broadcomp));
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 2)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 22),
                                                 _mm256_slli_epi32(w1, 10))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 21);
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 10),
                                                 _mm256_slli_epi32(w1, 22))),
          broadcomp));
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 22);
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                 _mm256_slli_epi32(w0, 2))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 23);
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 24);
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 18),
                                                 _mm256_slli_epi32(w0, 14))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 25);
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 12),
                                                 _mm256_slli_epi32(w1, 20))),
          broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 6), broadcomp));
}

/* we packed 256 27-bit values, touching 27 256-bit words, using 432 bytes */
static void filtereq27(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  27 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(134217727);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 27),
                                                 _mm256_slli_epi32(w1, 5))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 22),
                                                 _mm256_slli_epi32(w0, 10))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 17),
                                                 _mm256_slli_epi32(w1, 15))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 12),
                                                 _mm256_slli_epi32(w0, 20))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 7),
                                                 _mm256_slli_epi32(w1, 25))),
          broadcomp));
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 2)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 29),
                                                 _mm256_slli_epi32(w0, 3))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 19),
                                                 _mm256_slli_epi32(w0, 13))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 14),
                                                 _mm256_slli_epi32(w1, 18))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 9),
                                                 _mm256_slli_epi32(w0, 23))),
          broadcomp));
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 31),
                                                 _mm256_slli_epi32(w1, 1))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                 _mm256_slli_epi32(w0, 6))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 21),
                                                 _mm256_slli_epi32(w1, 11))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 11),
                                                 _mm256_slli_epi32(w1, 21))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 6),
                                                 _mm256_slli_epi32(w0, 26))),
          broadcomp));
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 1)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 23),
                                                 _mm256_slli_epi32(w0, 9))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 18),
                                                 _mm256_slli_epi32(w1, 14))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 13),
                                                 _mm256_slli_epi32(w0, 19))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 21);
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 8),
                                                 _mm256_slli_epi32(w1, 24))),
          broadcomp));
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 3)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 22);
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                 _mm256_slli_epi32(w0, 2))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 23);
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 25),
                                                 _mm256_slli_epi32(w1, 7))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 24);
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 25);
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 15),
                                                 _mm256_slli_epi32(w1, 17))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 26);
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 10),
                                                 _mm256_slli_epi32(w0, 22))),
          broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 5), broadcomp));
}

/* we packed 256 28-bit values, touching 28 256-bit words, using 448 bytes */
static void filtereq28(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  28 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(268435455);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                 _mm256_slli_epi32(w1, 12))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 12),
                                                 _mm256_slli_epi32(w1, 20))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 8),
                                                 _mm256_slli_epi32(w0, 24))),
          broadcomp));
  _mm256_storeu_si256(out + 7,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 4), broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 8, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 12),
                                                 _mm256_slli_epi32(w0, 20))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 8),
                                                 _mm256_slli_epi32(w1, 24))),
          broadcomp));
  _mm256_storeu_si256(out + 15,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 4), broadcomp));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 16, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                 _mm256_slli_epi32(w1, 12))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 12),
                                                 _mm256_slli_epi32(w1, 20))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 8),
                                                 _mm256_slli_epi32(w0, 24))),
          broadcomp));
  _mm256_storeu_si256(out + 23,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 4), broadcomp));
  w1 = _mm256_lddqu_si256(in + 21);
  _mm256_storeu_si256(
      out + 24, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp));
  w0 = _mm256_lddqu_si256(in + 22);
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 23);
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 24);
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 25);
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 26);
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 12),
                                                 _mm256_slli_epi32(w0, 20))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 27);
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 8),
                                                 _mm256_slli_epi32(w1, 24))),
          broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 4), broadcomp));
}

/* we packed 256 29-bit values, touching 29 256-bit words, using 464 bytes */
static void filtereq29(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  29 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(536870911);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 29),
                                                 _mm256_slli_epi32(w1, 3))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                 _mm256_slli_epi32(w0, 6))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 23),
                                                 _mm256_slli_epi32(w1, 9))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 17),
                                                 _mm256_slli_epi32(w1, 15))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 14),
                                                 _mm256_slli_epi32(w0, 18))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 11),
                                                 _mm256_slli_epi32(w1, 21))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 8),
                                                 _mm256_slli_epi32(w0, 24))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 5),
                                                 _mm256_slli_epi32(w1, 27))),
          broadcomp));
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 2)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 31),
                                                 _mm256_slli_epi32(w0, 1))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 25),
                                                 _mm256_slli_epi32(w0, 7))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 22),
                                                 _mm256_slli_epi32(w1, 10))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 19),
                                                 _mm256_slli_epi32(w0, 13))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 13),
                                                 _mm256_slli_epi32(w0, 19))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 10),
                                                 _mm256_slli_epi32(w1, 22))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 7),
                                                 _mm256_slli_epi32(w0, 25))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 4),
                                                 _mm256_slli_epi32(w1, 28))),
          broadcomp));
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 1)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                 _mm256_slli_epi32(w0, 2))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 21);
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 27),
                                                 _mm256_slli_epi32(w1, 5))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 22);
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 23);
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 21),
                                                 _mm256_slli_epi32(w1, 11))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 24);
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 18),
                                                 _mm256_slli_epi32(w0, 14))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 25);
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 15),
                                                 _mm256_slli_epi32(w1, 17))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 26);
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 12),
                                                 _mm256_slli_epi32(w0, 20))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 27);
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 9),
                                                 _mm256_slli_epi32(w1, 23))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 28);
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 6),
                                                 _mm256_slli_epi32(w0, 26))),
          broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 3), broadcomp));
}

/* we packed 256 30-bit values, touching 30 256-bit words, using 480 bytes */
static void filtereq30(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  30 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(1073741823);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                 _mm256_slli_epi32(w1, 2))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 26),
                                                 _mm256_slli_epi32(w1, 6))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 22),
                                                 _mm256_slli_epi32(w1, 10))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 18),
                                                 _mm256_slli_epi32(w1, 14))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 14),
                                                 _mm256_slli_epi32(w1, 18))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 12),
                                                 _mm256_slli_epi32(w0, 20))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 10),
                                                 _mm256_slli_epi32(w1, 22))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 8),
                                                 _mm256_slli_epi32(w0, 24))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 6),
                                                 _mm256_slli_epi32(w1, 26))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 4),
                                                 _mm256_slli_epi32(w0, 28))),
          broadcomp));
  _mm256_storeu_si256(out + 15,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 2), broadcomp));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 16, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                 _mm256_slli_epi32(w0, 2))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                 _mm256_slli_epi32(w0, 6))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 22),
                                                 _mm256_slli_epi32(w0, 10))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 21);
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                 _mm256_slli_epi32(w1, 12))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 22);
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 18),
                                                 _mm256_slli_epi32(w0, 14))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 23);
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 24);
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 14),
                                                 _mm256_slli_epi32(w0, 18))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 25);
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 12),
                                                 _mm256_slli_epi32(w1, 20))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 26);
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 10),
                                                 _mm256_slli_epi32(w0, 22))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 27);
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 8),
                                                 _mm256_slli_epi32(w1, 24))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 28);
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 6),
                                                 _mm256_slli_epi32(w0, 26))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 29);
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 4),
                                                 _mm256_slli_epi32(w1, 28))),
          broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 2), broadcomp));
}

/* we packed 256 31-bit values, touching 31 256-bit words, using 496 bytes */
static void filtereq31(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  31 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(2147483647);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 31),
                                                 _mm256_slli_epi32(w1, 1))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                 _mm256_slli_epi32(w0, 2))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 29),
                                                 _mm256_slli_epi32(w1, 3))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 27),
                                                 _mm256_slli_epi32(w1, 5))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                 _mm256_slli_epi32(w0, 6))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 25),
                                                 _mm256_slli_epi32(w1, 7))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 23),
                                                 _mm256_slli_epi32(w1, 9))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 22),
                                                 _mm256_slli_epi32(w0, 10))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 21),
                                                 _mm256_slli_epi32(w1, 11))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 19),
                                                 _mm256_slli_epi32(w1, 13))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 18),
                                                 _mm256_slli_epi32(w0, 14))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 17),
                                                 _mm256_slli_epi32(w1, 15))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 15),
                                                 _mm256_slli_epi32(w1, 17))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 14),
                                                 _mm256_slli_epi32(w0, 18))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 13),
                                                 _mm256_slli_epi32(w1, 19))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 12),
                                                 _mm256_slli_epi32(w0, 20))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 21);
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 11),
                                                 _mm256_slli_epi32(w1, 21))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 22);
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 10),
                                                 _mm256_slli_epi32(w0, 22))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 23);
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 9),
                                                 _mm256_slli_epi32(w1, 23))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 24);
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 8),
                                                 _mm256_slli_epi32(w0, 24))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 25);
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 7),
                                                 _mm256_slli_epi32(w1, 25))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 26);
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 6),
                                                 _mm256_slli_epi32(w0, 26))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 27);
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 5),
                                                 _mm256_slli_epi32(w1, 27))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 28);
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 4),
                                                 _mm256_slli_epi32(w0, 28))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 29);
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 3),
                                                 _mm256_slli_epi32(w1, 29))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 30);
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpeq_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 2),
                                                 _mm256_slli_epi32(w0, 30))),
          broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 1), broadcomp));
}

/* we packed 256 32-bit values, touching 32 256-bit words, using 512 bytes */
static void filtereq32(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  32 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(out + 0, _mm256_cmpeq_epi32(w0, broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(out + 1, _mm256_cmpeq_epi32(w1, broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(out + 2, _mm256_cmpeq_epi32(w0, broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(out + 3, _mm256_cmpeq_epi32(w1, broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(out + 4, _mm256_cmpeq_epi32(w0, broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(out + 5, _mm256_cmpeq_epi32(w1, broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(out + 6, _mm256_cmpeq_epi32(w0, broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(out + 7, _mm256_cmpeq_epi32(w1, broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(out + 8, _mm256_cmpeq_epi32(w0, broadcomp));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(out + 9, _mm256_cmpeq_epi32(w1, broadcomp));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(out + 10, _mm256_cmpeq_epi32(w0, broadcomp));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(out + 11, _mm256_cmpeq_epi32(w1, broadcomp));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(out + 12, _mm256_cmpeq_epi32(w0, broadcomp));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(out + 13, _mm256_cmpeq_epi32(w1, broadcomp));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(out + 14, _mm256_cmpeq_epi32(w0, broadcomp));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(out + 15, _mm256_cmpeq_epi32(w1, broadcomp));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(out + 16, _mm256_cmpeq_epi32(w0, broadcomp));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(out + 17, _mm256_cmpeq_epi32(w1, broadcomp));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(out + 18, _mm256_cmpeq_epi32(w0, broadcomp));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(out + 19, _mm256_cmpeq_epi32(w1, broadcomp));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(out + 20, _mm256_cmpeq_epi32(w0, broadcomp));
  w1 = _mm256_lddqu_si256(in + 21);
  _mm256_storeu_si256(out + 21, _mm256_cmpeq_epi32(w1, broadcomp));
  w0 = _mm256_lddqu_si256(in + 22);
  _mm256_storeu_si256(out + 22, _mm256_cmpeq_epi32(w0, broadcomp));
  w1 = _mm256_lddqu_si256(in + 23);
  _mm256_storeu_si256(out + 23, _mm256_cmpeq_epi32(w1, broadcomp));
  w0 = _mm256_lddqu_si256(in + 24);
  _mm256_storeu_si256(out + 24, _mm256_cmpeq_epi32(w0, broadcomp));
  w1 = _mm256_lddqu_si256(in + 25);
  _mm256_storeu_si256(out + 25, _mm256_cmpeq_epi32(w1, broadcomp));
  w0 = _mm256_lddqu_si256(in + 26);
  _mm256_storeu_si256(out + 26, _mm256_cmpeq_epi32(w0, broadcomp));
  w1 = _mm256_lddqu_si256(in + 27);
  _mm256_storeu_si256(out + 27, _mm256_cmpeq_epi32(w1, broadcomp));
  w0 = _mm256_lddqu_si256(in + 28);
  _mm256_storeu_si256(out + 28, _mm256_cmpeq_epi32(w0, broadcomp));
  w1 = _mm256_lddqu_si256(in + 29);
  _mm256_storeu_si256(out + 29, _mm256_cmpeq_epi32(w1, broadcomp));
  w0 = _mm256_lddqu_si256(in + 30);
  _mm256_storeu_si256(out + 30, _mm256_cmpeq_epi32(w0, broadcomp));
  w1 = _mm256_lddqu_si256(in + 31);
  _mm256_storeu_si256(out + 31, _mm256_cmpeq_epi32(w1, broadcomp));
}

static void filterneq0(const __m256i *in, u32 *matches, const INTEGER comp) {
  if (comp == 0)
    memset(matches, 0, 256 * sizeof(*matches));
  else
    memset(matches, 1, 256 * sizeof(*matches));
}

/* we packed 256 1-bit values, touching 1 256-bit words, using 16 bytes */
static void filterneq1(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  1 256-bit word */
  __m256i w0;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(1);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 1,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 1)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 2,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 2)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 3,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 3)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 4,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 5,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 5)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 6,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 6)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 7,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 7)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 8,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 9,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 9)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 10,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 10)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 11,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 11)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 12,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 12)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 13,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 13)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 14,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 14)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 15,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 15)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 16,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 17,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 17)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 18,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 18)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 19,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 19)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 20,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 20)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 21,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 21)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 22,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 22)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 23,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 23)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 24,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 24)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 25,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 25)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 26,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 26)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 27,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 27)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 28,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 28)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 29,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 29)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 30,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 30)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 31,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 31), broadcomp),
                       _mm256_set1_epi32(-1)));
}

/* we packed 256 2-bit values, touching 2 256-bit words, using 32 bytes */
static void filterneq2(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  2 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(3);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 1,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 2)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 2,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 3,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 6)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 4,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 5,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 10)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 6,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 12)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 7,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 14)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 8,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 9,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 18)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 10,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 20)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 11,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 22)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 12,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 24)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 13,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 26)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 14,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 28)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 15,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 30), broadcomp),
                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 16, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 17,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 2)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 18,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 19,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 6)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 20,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 21,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 10)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 22,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 12)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 23,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 14)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 24,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 25,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 18)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 26,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 20)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 27,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 22)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 28,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 24)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 29,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 26)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 30,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 28)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 31,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 30), broadcomp),
                       _mm256_set1_epi32(-1)));
}

/* we packed 256 3-bit values, touching 3 256-bit words, using 48 bytes */
static void filterneq3(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  3 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(7);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 1,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 3)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 2,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 6)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 3,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 9)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 4,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 12)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 5,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 15)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 6,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 18)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 7,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 21)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 8,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 24)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 9,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 27)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 10,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                     _mm256_slli_epi32(w1, 2))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 11,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 1)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 12,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 13,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 7)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 14,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 10)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 15,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 13)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 16,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 17,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 19)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 18,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 22)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 19,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 25)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 20,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 28)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 21,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 31),
                                                     _mm256_slli_epi32(w0, 1))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 22,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 2)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 23,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 5)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 24,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 25,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 11)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 26,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 14)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 27,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 17)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 28,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 20)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 29,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 23)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 30,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 26)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 31,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 29), broadcomp),
                       _mm256_set1_epi32(-1)));
}

/* we packed 256 4-bit values, touching 4 256-bit words, using 64 bytes */
static void filterneq4(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  4 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(15);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 1,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 2,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 3,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 12)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 4,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 5,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 20)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 6,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 24)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 7,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 28), broadcomp),
                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 8, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 9,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 10,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 11,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 12)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 12,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 13,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 20)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 14,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 24)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 15,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 28), broadcomp),
                       _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 16, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 17,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 18,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 19,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 12)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 20,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 21,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 20)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 22,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 24)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 23,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 28), broadcomp),
                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 24, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 25,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 26,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 27,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 12)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 28,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 29,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 20)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 30,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 24)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 31,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 28), broadcomp),
                       _mm256_set1_epi32(-1)));
}

/* we packed 256 5-bit values, touching 5 256-bit words, using 80 bytes */
static void filterneq5(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  5 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(31);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 1,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 5)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 2,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 10)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 3,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 15)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 4,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 20)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 5,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 25)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 6,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                     _mm256_slli_epi32(w1, 2))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 7,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 3)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 8,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 9,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 13)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 10,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 18)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 11,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 23)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 12,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                     _mm256_slli_epi32(w0, 4))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 13,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 1)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 14,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 6)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 15,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 11)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 16,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 17,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 21)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 18,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 26)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 19,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 31),
                                                     _mm256_slli_epi32(w1, 1))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 20,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 21,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 9)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 22,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 14)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 23,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 19)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 24,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 24)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 25,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 29),
                                                     _mm256_slli_epi32(w0, 3))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 26,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 2)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 27,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 7)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 28,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 12)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 29,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 17)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 30,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 22)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 31,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 27), broadcomp),
                       _mm256_set1_epi32(-1)));
}

/* we packed 256 6-bit values, touching 6 256-bit words, using 96 bytes */
static void filterneq6(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  6 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(63);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 1,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 6)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 2,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 12)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 3,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 18)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 4,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 24)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 5,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                     _mm256_slli_epi32(w1, 2))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 6,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 7,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 10)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 8,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 9,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 22)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 10,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                     _mm256_slli_epi32(w0, 4))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 11,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 2)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 12,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 13,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 14)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 14,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 20)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 15,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 26), broadcomp),
                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 16, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 17,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 6)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 18,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 12)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 19,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 18)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 20,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 24)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 21,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                     _mm256_slli_epi32(w0, 2))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 22,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 23,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 10)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 24,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 25,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 22)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 26,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                     _mm256_slli_epi32(w1, 4))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 27,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 2)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 28,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 29,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 14)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 30,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 20)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 31,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 26), broadcomp),
                       _mm256_set1_epi32(-1)));
}

/* we packed 256 7-bit values, touching 7 256-bit words, using 112 bytes */
static void filterneq7(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  7 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(127);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 1,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 7)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 2,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 14)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 3,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 21)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 4,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                     _mm256_slli_epi32(w1, 4))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 5,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 3)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 6,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 10)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 7,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 17)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 8,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 24)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 9,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 31),
                                                     _mm256_slli_epi32(w0, 1))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 10,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 6)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 11,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 13)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 12,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 20)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 13,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 27),
                                                     _mm256_slli_epi32(w1, 5))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 14,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 2)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 15,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 9)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 16,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 17,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 23)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 18,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                     _mm256_slli_epi32(w0, 2))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 19,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 5)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 20,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 12)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 21,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 19)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 22,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 26),
                                                     _mm256_slli_epi32(w1, 6))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 23,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 1)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 24,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 25,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 15)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 26,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 22)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 27,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 29),
                                                     _mm256_slli_epi32(w0, 3))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 28,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 29,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 11)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 30,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 18)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 31,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 25), broadcomp),
                       _mm256_set1_epi32(-1)));
}

/* we packed 256 8-bit values, touching 8 256-bit words, using 128 bytes */
static void filterneq8(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  8 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(255);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 1,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 2,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 3,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 24), broadcomp),
                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 4, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 5,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 6,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 7,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 24), broadcomp),
                       _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 8, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 9,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 10,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 11,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 24), broadcomp),
                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 12, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 13,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 14,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 15,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 24), broadcomp),
                       _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 16, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 17,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 18,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 19,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 24), broadcomp),
                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 20, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 21,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 22,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 23,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 24), broadcomp),
                       _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 24, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 25,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 26,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 27,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 24), broadcomp),
                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 28, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 29,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 30,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 31,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 24), broadcomp),
                       _mm256_set1_epi32(-1)));
}

/* we packed 256 9-bit values, touching 9 256-bit words, using 144 bytes */
static void filterneq9(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  9 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(511);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 1,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 9)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 2,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 18)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 3,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 27),
                                                     _mm256_slli_epi32(w1, 5))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 4,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 5,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 13)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 6,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 22)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 7,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 31),
                                                     _mm256_slli_epi32(w0, 1))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 8,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 9,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 17)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 10,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 26),
                                                     _mm256_slli_epi32(w1, 6))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 11,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 3)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 12,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 12)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 13,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 21)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 14,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                     _mm256_slli_epi32(w0, 2))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 15,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 7)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 16,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 17,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 25),
                                                     _mm256_slli_epi32(w1, 7))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 18,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 2)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 19,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 11)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 20,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 20)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 21,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 29),
                                                     _mm256_slli_epi32(w0, 3))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 22,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 6)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 23,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 15)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 24,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                     _mm256_slli_epi32(w1, 8))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 25,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 1)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 26,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 10)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 27,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 19)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 28,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                     _mm256_slli_epi32(w0, 4))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 29,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 5)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 30,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 14)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 31,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 23), broadcomp),
                       _mm256_set1_epi32(-1)));
}

/* we packed 256 10-bit values, touching 10 256-bit words, using 160 bytes */
static void filterneq10(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  10 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(1023);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 1,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 10)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 2,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 20)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 3,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                     _mm256_slli_epi32(w1, 2))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 4,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 5,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 18)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 6,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                     _mm256_slli_epi32(w0, 4))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 7,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 6)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 8,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 9,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 26),
                                                     _mm256_slli_epi32(w1, 6))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 10,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 11,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 14)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 12,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                     _mm256_slli_epi32(w0, 8))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 13,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 2)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 14,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 12)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 15,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 22), broadcomp),
                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 16, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 17,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 10)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 18,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 20)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 19,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                     _mm256_slli_epi32(w0, 2))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 20,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 21,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 18)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 22,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                     _mm256_slli_epi32(w1, 4))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 23,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 6)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 24,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 25,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                     _mm256_slli_epi32(w0, 6))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 26,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 27,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 14)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 28,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                     _mm256_slli_epi32(w1, 8))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 29,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 2)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 30,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 12)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 31,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 22), broadcomp),
                       _mm256_set1_epi32(-1)));
}

/* we packed 256 11-bit values, touching 11 256-bit words, using 176 bytes */
static void filterneq11(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  11 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(2047);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 1,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 11)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 2, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w0, 22),
                                                 _mm256_slli_epi32(w1, 10))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 3,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 1)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 4,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 12)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 5,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 23),
                                                     _mm256_slli_epi32(w0, 9))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 6,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 2)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 7,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 13)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 8,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                     _mm256_slli_epi32(w1, 8))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 9,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 3)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 10,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 14)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 11,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 25),
                                                     _mm256_slli_epi32(w0, 7))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 12,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 13,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 15)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 14,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 26),
                                                     _mm256_slli_epi32(w1, 6))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 15,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 5)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 16,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 17,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 27),
                                                     _mm256_slli_epi32(w0, 5))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 18,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 6)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 19,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 17)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 20,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                     _mm256_slli_epi32(w1, 4))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 21,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 7)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 22,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 18)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 23,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 29),
                                                     _mm256_slli_epi32(w0, 3))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 24,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 25,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 19)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 26,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                     _mm256_slli_epi32(w1, 2))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 27,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 9)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 28,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 20)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 29,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 31),
                                                     _mm256_slli_epi32(w0, 1))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 30,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 10)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 31,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 21), broadcomp),
                       _mm256_set1_epi32(-1)));
}

/* we packed 256 12-bit values, touching 12 256-bit words, using 192 bytes */
static void filterneq12(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  12 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(4095);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 1,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 12)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 2,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                     _mm256_slli_epi32(w1, 8))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 3,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 4,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 5,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                     _mm256_slli_epi32(w0, 4))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 6,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 7,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 20), broadcomp),
                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 8, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 9,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 12)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 10,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                     _mm256_slli_epi32(w0, 8))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 11,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 12,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 13,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                     _mm256_slli_epi32(w1, 4))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 14,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 15,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 20), broadcomp),
                       _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 16, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 17,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 12)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 18,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                     _mm256_slli_epi32(w1, 8))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 19,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 20,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 21,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                     _mm256_slli_epi32(w0, 4))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 22,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 23,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 20), broadcomp),
                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 24, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 25,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 12)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 26,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                     _mm256_slli_epi32(w0, 8))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 27,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 28,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 29,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                     _mm256_slli_epi32(w1, 4))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 30,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 31,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 20), broadcomp),
                       _mm256_set1_epi32(-1)));
}

/* we packed 256 13-bit values, touching 13 256-bit words, using 208 bytes */
static void filterneq13(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  13 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(8191);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 1,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 13)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 2,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 26),
                                                     _mm256_slli_epi32(w1, 6))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 3,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 7)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 4, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 5,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 1)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 6,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 14)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 7,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 27),
                                                     _mm256_slli_epi32(w1, 5))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 8,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 9, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w1, 21),
                                                 _mm256_slli_epi32(w0, 11))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 10,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 2)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 11,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 15)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 12,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                     _mm256_slli_epi32(w1, 4))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 13,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 9)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 14, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 22),
                                                  _mm256_slli_epi32(w0, 10))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 15,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 3)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 16,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 17,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 29),
                                                     _mm256_slli_epi32(w1, 3))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 18,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 10)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 19,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 23),
                                                     _mm256_slli_epi32(w0, 9))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 20,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 21,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 17)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 22,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                     _mm256_slli_epi32(w1, 2))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 23,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 11)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 24,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                     _mm256_slli_epi32(w0, 8))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 25,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 5)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 26,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 18)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 27,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 31),
                                                     _mm256_slli_epi32(w1, 1))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 28,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 12)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 29,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 25),
                                                     _mm256_slli_epi32(w0, 7))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 30,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 6)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 31,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 19), broadcomp),
                       _mm256_set1_epi32(-1)));
}

/* we packed 256 14-bit values, touching 14 256-bit words, using 224 bytes */
static void filterneq14(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  14 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(16383);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 1,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 14)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 2,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                     _mm256_slli_epi32(w1, 4))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 3,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 10)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 4,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                     _mm256_slli_epi32(w0, 8))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 5,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 6)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 6, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                 _mm256_slli_epi32(w1, 12))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 7,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 2)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 8,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 9,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                     _mm256_slli_epi32(w0, 2))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 10,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 12)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 11,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 26),
                                                     _mm256_slli_epi32(w1, 6))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 12,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 13, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 22),
                                                  _mm256_slli_epi32(w0, 10))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 14,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 15,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 18), broadcomp),
                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 16, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 17,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 14)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 18,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                     _mm256_slli_epi32(w0, 4))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 19,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 10)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 20,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                     _mm256_slli_epi32(w1, 8))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 21,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 6)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 22, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                  _mm256_slli_epi32(w0, 12))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 23,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 2)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 24,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 25,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                     _mm256_slli_epi32(w1, 2))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 26,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 12)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 27,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                     _mm256_slli_epi32(w0, 6))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 28,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 29, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 22),
                                                  _mm256_slli_epi32(w1, 10))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 30,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 31,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 18), broadcomp),
                       _mm256_set1_epi32(-1)));
}

/* we packed 256 15-bit values, touching 15 256-bit words, using 240 bytes */
static void filterneq15(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  15 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(32767);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 1,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 15)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 2,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                     _mm256_slli_epi32(w1, 2))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 3,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 13)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 4,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                     _mm256_slli_epi32(w0, 4))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 5,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 11)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 6,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 26),
                                                     _mm256_slli_epi32(w1, 6))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 7,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 9)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 8,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                     _mm256_slli_epi32(w0, 8))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 9,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 7)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 10, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 22),
                                                  _mm256_slli_epi32(w1, 10))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 11,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 5)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 12, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                  _mm256_slli_epi32(w0, 12))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 13,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 3)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 14, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 18),
                                                  _mm256_slli_epi32(w1, 14))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 15,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 1)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 16,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 17,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 31),
                                                     _mm256_slli_epi32(w0, 1))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 18,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 14)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 19,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 29),
                                                     _mm256_slli_epi32(w1, 3))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 20,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 12)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 21,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 27),
                                                     _mm256_slli_epi32(w0, 5))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 22,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 10)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 23,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 25),
                                                     _mm256_slli_epi32(w1, 7))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 24,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 25,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 23),
                                                     _mm256_slli_epi32(w0, 9))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 26,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 6)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 27, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 21),
                                                  _mm256_slli_epi32(w1, 11))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 28,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 29, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 19),
                                                  _mm256_slli_epi32(w0, 13))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 30,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 2)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 31,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 17), broadcomp),
                       _mm256_set1_epi32(-1)));
}

/* we packed 256 16-bit values, touching 16 256-bit words, using 256 bytes */
static void filterneq16(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  16 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(65535);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 1,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 16), broadcomp),
                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 2, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 3,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 16), broadcomp),
                       _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 4, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 5,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 16), broadcomp),
                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 6, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 7,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 16), broadcomp),
                       _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 8, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 9,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 16), broadcomp),
                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 10, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 11,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 16), broadcomp),
                       _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 12, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 13,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 16), broadcomp),
                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 14, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 15,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 16), broadcomp),
                       _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 16, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 17,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 16), broadcomp),
                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 18, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 19,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 16), broadcomp),
                       _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 20, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 21,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 16), broadcomp),
                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 22, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 23,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 16), broadcomp),
                       _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 24, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 25,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 16), broadcomp),
                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 26, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 27,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 16), broadcomp),
                       _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 28, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 29,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 16), broadcomp),
                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 30, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 31,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 16), broadcomp),
                       _mm256_set1_epi32(-1)));
}

/* we packed 256 17-bit values, touching 17 256-bit words, using 272 bytes */
static void filterneq17(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  17 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(131071);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                   _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w0, 17),
                                                 _mm256_slli_epi32(w1, 15))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 2,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 2)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 3, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w1, 19),
                                                 _mm256_slli_epi32(w0, 13))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 4,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 5, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w0, 21),
                                                 _mm256_slli_epi32(w1, 11))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 6,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 6)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 7,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 23),
                                                     _mm256_slli_epi32(w0, 9))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 8,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 9,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 25),
                                                     _mm256_slli_epi32(w1, 7))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 10,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 10)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 11,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 27),
                                                     _mm256_slli_epi32(w0, 5))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 12,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 12)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 13,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 29),
                                                     _mm256_slli_epi32(w1, 3))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 14,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 14)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 15,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 31),
                                                     _mm256_slli_epi32(w0, 1))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 16, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                  _mm256_slli_epi32(w1, 16))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 17,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 1)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 18, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 18),
                                                  _mm256_slli_epi32(w0, 14))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 19,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 3)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 20, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                  _mm256_slli_epi32(w1, 12))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 21,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 5)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 22, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 22),
                                                  _mm256_slli_epi32(w0, 10))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 23,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 7)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 24,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                     _mm256_slli_epi32(w1, 8))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 25,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 9)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 26,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                     _mm256_slli_epi32(w0, 6))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 27,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 11)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 28,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                     _mm256_slli_epi32(w1, 4))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 29,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 13)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 30,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                     _mm256_slli_epi32(w0, 2))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 31,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 15), broadcomp),
                       _mm256_set1_epi32(-1)));
}

/* we packed 256 18-bit values, touching 18 256-bit words, using 288 bytes */
static void filterneq18(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  18 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(262143);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                   _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w0, 18),
                                                 _mm256_slli_epi32(w1, 14))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 2,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 3, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w1, 22),
                                                 _mm256_slli_epi32(w0, 10))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 4,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 5,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 26),
                                                     _mm256_slli_epi32(w1, 6))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 6,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 12)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 7,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                     _mm256_slli_epi32(w0, 2))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 8, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 9,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 2)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 10, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                  _mm256_slli_epi32(w0, 12))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 11,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 6)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 12,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                     _mm256_slli_epi32(w1, 8))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 13,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 10)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 14,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                     _mm256_slli_epi32(w0, 4))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 15,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 14), broadcomp),
                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 16, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 17, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 18),
                                                  _mm256_slli_epi32(w0, 14))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 18,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 19, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 22),
                                                  _mm256_slli_epi32(w1, 10))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 20,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 21,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                     _mm256_slli_epi32(w0, 6))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 22,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 12)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 23,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                     _mm256_slli_epi32(w1, 2))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 24, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                  _mm256_slli_epi32(w0, 16))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 25,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 2)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 26, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                  _mm256_slli_epi32(w1, 12))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 27,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 6)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 28,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                     _mm256_slli_epi32(w0, 8))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 29,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 10)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 30,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                     _mm256_slli_epi32(w1, 4))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 31,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 14), broadcomp),
                       _mm256_set1_epi32(-1)));
}

/* we packed 256 19-bit values, touching 19 256-bit words, using 304 bytes */
static void filterneq19(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  19 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(524287);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                   _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w0, 19),
                                                 _mm256_slli_epi32(w1, 13))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 2,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 6)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 3,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 25),
                                                     _mm256_slli_epi32(w0, 7))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 4,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 12)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 5,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 31),
                                                     _mm256_slli_epi32(w1, 1))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 6, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w1, 18),
                                                 _mm256_slli_epi32(w0, 14))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 7,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 5)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 8,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                     _mm256_slli_epi32(w1, 8))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 9,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 11)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 10,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                     _mm256_slli_epi32(w0, 2))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 11, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 17),
                                                  _mm256_slli_epi32(w1, 15))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 12,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 13,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 23),
                                                     _mm256_slli_epi32(w0, 9))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 14,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 10)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 15,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 29),
                                                     _mm256_slli_epi32(w1, 3))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 16, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                  _mm256_slli_epi32(w0, 16))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 17,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 3)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 18, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 22),
                                                  _mm256_slli_epi32(w1, 10))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 19,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 9)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 20,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                     _mm256_slli_epi32(w0, 4))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 21, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 15),
                                                  _mm256_slli_epi32(w1, 17))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 22,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 2)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 23, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 21),
                                                  _mm256_slli_epi32(w0, 11))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 24,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 25,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 27),
                                                     _mm256_slli_epi32(w1, 5))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 26, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 14),
                                                  _mm256_slli_epi32(w0, 18))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 27,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 1)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 28, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                  _mm256_slli_epi32(w1, 12))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 29,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 7)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 30,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                     _mm256_slli_epi32(w0, 6))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 31,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 13), broadcomp),
                       _mm256_set1_epi32(-1)));
}

/* we packed 256 20-bit values, touching 20 256-bit words, using 320 bytes */
static void filterneq20(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  20 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(1048575);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                   _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                 _mm256_slli_epi32(w1, 12))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 2,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 3,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                     _mm256_slli_epi32(w0, 4))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 4, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 5,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 6,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                     _mm256_slli_epi32(w0, 8))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 7,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 12), broadcomp),
                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 8, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp),
                   _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 9, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 10,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 11,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                     _mm256_slli_epi32(w1, 4))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 12, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                  _mm256_slli_epi32(w0, 16))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 13,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 14,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                     _mm256_slli_epi32(w1, 8))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 15,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 12), broadcomp),
                       _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 16, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                    _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 17, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                  _mm256_slli_epi32(w1, 12))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 18,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 19,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                     _mm256_slli_epi32(w0, 4))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 20, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                  _mm256_slli_epi32(w1, 16))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 21,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 22,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                     _mm256_slli_epi32(w0, 8))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 23,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 12), broadcomp),
                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 24, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 25, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                  _mm256_slli_epi32(w0, 12))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 26,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 27,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                     _mm256_slli_epi32(w1, 4))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 28, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                  _mm256_slli_epi32(w0, 16))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 29,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 30,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                     _mm256_slli_epi32(w1, 8))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 31,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 12), broadcomp),
                       _mm256_set1_epi32(-1)));
}

/* we packed 256 21-bit values, touching 21 256-bit words, using 336 bytes */
static void filterneq21(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  21 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(2097151);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                   _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w0, 21),
                                                 _mm256_slli_epi32(w1, 11))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 2,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 10)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 3,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 31),
                                                     _mm256_slli_epi32(w0, 1))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 4, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                 _mm256_slli_epi32(w1, 12))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 5,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 9)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 6,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                     _mm256_slli_epi32(w0, 2))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 7, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w0, 19),
                                                 _mm256_slli_epi32(w1, 13))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 8,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 9,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 29),
                                                     _mm256_slli_epi32(w0, 3))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 10, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 18),
                                                  _mm256_slli_epi32(w1, 14))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 11,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 7)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 12,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                     _mm256_slli_epi32(w0, 4))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 13, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 17),
                                                  _mm256_slli_epi32(w1, 15))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 14,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 6)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 15,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 27),
                                                     _mm256_slli_epi32(w0, 5))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 16, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                  _mm256_slli_epi32(w1, 16))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 17,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 5)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 18,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                     _mm256_slli_epi32(w0, 6))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 19, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 15),
                                                  _mm256_slli_epi32(w1, 17))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 20,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 21,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 25),
                                                     _mm256_slli_epi32(w0, 7))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 22, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 14),
                                                  _mm256_slli_epi32(w1, 18))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 23,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 3)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 24,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                     _mm256_slli_epi32(w0, 8))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 25, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 13),
                                                  _mm256_slli_epi32(w1, 19))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 26,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 2)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 27,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 23),
                                                     _mm256_slli_epi32(w0, 9))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 28, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 12),
                                                  _mm256_slli_epi32(w1, 20))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 29,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 1)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(
      out + 30, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 22),
                                                  _mm256_slli_epi32(w0, 10))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 31,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 11), broadcomp),
                       _mm256_set1_epi32(-1)));
}

/* we packed 256 22-bit values, touching 22 256-bit words, using 352 bytes */
static void filterneq22(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  22 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(4194303);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                   _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w0, 22),
                                                 _mm256_slli_epi32(w1, 10))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 2, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w1, 12),
                                                 _mm256_slli_epi32(w0, 20))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 3,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 2)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 4,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                     _mm256_slli_epi32(w1, 8))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 5, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w1, 14),
                                                 _mm256_slli_epi32(w0, 18))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 6,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 7,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 26),
                                                     _mm256_slli_epi32(w1, 6))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 8, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 9,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 6)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 10,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                     _mm256_slli_epi32(w1, 4))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 11, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 18),
                                                  _mm256_slli_epi32(w0, 14))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 12,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 13,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                     _mm256_slli_epi32(w1, 2))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 14, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                  _mm256_slli_epi32(w0, 12))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 15,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 10), broadcomp),
                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 16, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 17, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 22),
                                                  _mm256_slli_epi32(w0, 10))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 18, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 12),
                                                  _mm256_slli_epi32(w1, 20))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 19,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 2)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 20,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                     _mm256_slli_epi32(w0, 8))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 21, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 14),
                                                  _mm256_slli_epi32(w1, 18))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 22,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 23,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                     _mm256_slli_epi32(w0, 6))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 24, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                  _mm256_slli_epi32(w1, 16))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 25,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 6)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 26,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                     _mm256_slli_epi32(w0, 4))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 27, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 18),
                                                  _mm256_slli_epi32(w1, 14))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 28,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(
      out + 29,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                     _mm256_slli_epi32(w0, 2))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 21);
  _mm256_storeu_si256(
      out + 30, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                  _mm256_slli_epi32(w1, 12))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 31,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 10), broadcomp),
                       _mm256_set1_epi32(-1)));
}

/* we packed 256 23-bit values, touching 23 256-bit words, using 368 bytes */
static void filterneq23(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  23 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(8388607);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                   _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 23),
                                                     _mm256_slli_epi32(w1, 9))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 2, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w1, 14),
                                                 _mm256_slli_epi32(w0, 18))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 3,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 5)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 4,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                     _mm256_slli_epi32(w1, 4))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 5, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w1, 19),
                                                 _mm256_slli_epi32(w0, 13))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 6, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w0, 10),
                                                 _mm256_slli_epi32(w1, 22))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 7,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 1)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 8,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                     _mm256_slli_epi32(w0, 8))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 9, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w0, 15),
                                                 _mm256_slli_epi32(w1, 17))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 10,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 6)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 11,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 29),
                                                     _mm256_slli_epi32(w0, 3))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 12, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                  _mm256_slli_epi32(w1, 12))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 13, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 11),
                                                  _mm256_slli_epi32(w0, 21))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 14,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 2)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 15,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 25),
                                                     _mm256_slli_epi32(w1, 7))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 16, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                  _mm256_slli_epi32(w0, 16))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 17,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 7)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 18,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                     _mm256_slli_epi32(w1, 2))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 19, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 21),
                                                  _mm256_slli_epi32(w0, 11))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 20, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 12),
                                                  _mm256_slli_epi32(w1, 20))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 21,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 3)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 22,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                     _mm256_slli_epi32(w0, 6))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 23, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 17),
                                                  _mm256_slli_epi32(w1, 15))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 24,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 25,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 31),
                                                     _mm256_slli_epi32(w0, 1))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 26, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 22),
                                                  _mm256_slli_epi32(w1, 10))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(
      out + 27, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 13),
                                                  _mm256_slli_epi32(w0, 19))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 28,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 21);
  _mm256_storeu_si256(
      out + 29,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 27),
                                                     _mm256_slli_epi32(w1, 5))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 22);
  _mm256_storeu_si256(
      out + 30, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 18),
                                                  _mm256_slli_epi32(w0, 14))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 31,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 9), broadcomp),
                       _mm256_set1_epi32(-1)));
}

/* we packed 256 24-bit values, touching 24 256-bit words, using 384 bytes */
static void filterneq24(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  24 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(16777215);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                   _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                     _mm256_slli_epi32(w1, 8))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 2, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 3,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 8), broadcomp),
                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 4, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp),
                   _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 5,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                     _mm256_slli_epi32(w0, 8))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 6, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 7,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 8), broadcomp),
                       _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 8, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                   _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 9,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                     _mm256_slli_epi32(w1, 8))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 10, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                  _mm256_slli_epi32(w0, 16))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 11,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 8), broadcomp),
                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 12, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 13,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                     _mm256_slli_epi32(w0, 8))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 14, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                  _mm256_slli_epi32(w1, 16))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 15,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 8), broadcomp),
                       _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 16, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                    _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 17,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                     _mm256_slli_epi32(w1, 8))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 18, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                  _mm256_slli_epi32(w0, 16))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 19,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 8), broadcomp),
                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 20, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 21,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                     _mm256_slli_epi32(w0, 8))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 22, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                  _mm256_slli_epi32(w1, 16))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 23,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 8), broadcomp),
                       _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 24, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                    _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 25,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                     _mm256_slli_epi32(w1, 8))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(
      out + 26, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                  _mm256_slli_epi32(w0, 16))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 27,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 8), broadcomp),
                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 21);
  _mm256_storeu_si256(
      out + 28, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 22);
  _mm256_storeu_si256(
      out + 29,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                     _mm256_slli_epi32(w0, 8))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 23);
  _mm256_storeu_si256(
      out + 30, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                  _mm256_slli_epi32(w1, 16))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 31,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 8), broadcomp),
                       _mm256_set1_epi32(-1)));
}

/* we packed 256 25-bit values, touching 25 256-bit words, using 400 bytes */
static void filterneq25(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  25 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(33554431);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                   _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 25),
                                                     _mm256_slli_epi32(w1, 7))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 2, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w1, 18),
                                                 _mm256_slli_epi32(w0, 14))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 3, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w0, 11),
                                                 _mm256_slli_epi32(w1, 21))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 4,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 5,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 29),
                                                     _mm256_slli_epi32(w0, 3))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 6, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w0, 22),
                                                 _mm256_slli_epi32(w1, 10))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 7, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w1, 15),
                                                 _mm256_slli_epi32(w0, 17))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 8, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w0, 8),
                                                 _mm256_slli_epi32(w1, 24))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 9,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 1)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 10,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                     _mm256_slli_epi32(w0, 6))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 11, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 19),
                                                  _mm256_slli_epi32(w1, 13))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 12, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 12),
                                                  _mm256_slli_epi32(w0, 20))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 13,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 5)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 14,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                     _mm256_slli_epi32(w1, 2))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 15,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 23),
                                                     _mm256_slli_epi32(w0, 9))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 16, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                  _mm256_slli_epi32(w1, 16))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 17, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 9),
                                                  _mm256_slli_epi32(w0, 23))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 18,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 2)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 19,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 27),
                                                     _mm256_slli_epi32(w1, 5))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 20, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                  _mm256_slli_epi32(w0, 12))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 21, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 13),
                                                  _mm256_slli_epi32(w1, 19))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 22,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 6)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 23,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 31),
                                                     _mm256_slli_epi32(w0, 1))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 24,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                     _mm256_slli_epi32(w1, 8))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(
      out + 25, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 17),
                                                  _mm256_slli_epi32(w0, 15))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 21);
  _mm256_storeu_si256(
      out + 26, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 10),
                                                  _mm256_slli_epi32(w1, 22))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 27,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 3)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 22);
  _mm256_storeu_si256(
      out + 28,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                     _mm256_slli_epi32(w0, 4))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 23);
  _mm256_storeu_si256(
      out + 29, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 21),
                                                  _mm256_slli_epi32(w1, 11))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 24);
  _mm256_storeu_si256(
      out + 30, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 14),
                                                  _mm256_slli_epi32(w0, 18))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 31,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 7), broadcomp),
                       _mm256_set1_epi32(-1)));
}

/* we packed 256 26-bit values, touching 26 256-bit words, using 416 bytes */
static void filterneq26(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  26 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(67108863);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                   _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 26),
                                                     _mm256_slli_epi32(w1, 6))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 2, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 3, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w0, 14),
                                                 _mm256_slli_epi32(w1, 18))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 4, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w1, 8),
                                                 _mm256_slli_epi32(w0, 24))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 5,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 2)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 6,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                     _mm256_slli_epi32(w1, 4))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 7, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w1, 22),
                                                 _mm256_slli_epi32(w0, 10))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 8, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 9, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w1, 10),
                                                 _mm256_slli_epi32(w0, 22))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 10,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 11,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                     _mm256_slli_epi32(w1, 2))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 12,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                     _mm256_slli_epi32(w0, 8))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 13, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 18),
                                                  _mm256_slli_epi32(w1, 14))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 14, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 12),
                                                  _mm256_slli_epi32(w0, 20))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 15,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 6), broadcomp),
                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 16, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 17,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                     _mm256_slli_epi32(w0, 6))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 18, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                  _mm256_slli_epi32(w1, 12))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 19, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 14),
                                                  _mm256_slli_epi32(w0, 18))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 20, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 8),
                                                  _mm256_slli_epi32(w1, 24))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 21,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 2)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 22,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                     _mm256_slli_epi32(w0, 4))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 23, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 22),
                                                  _mm256_slli_epi32(w1, 10))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(
      out + 24, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                  _mm256_slli_epi32(w0, 16))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 21);
  _mm256_storeu_si256(
      out + 25, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 10),
                                                  _mm256_slli_epi32(w1, 22))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 26,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 22);
  _mm256_storeu_si256(
      out + 27,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                     _mm256_slli_epi32(w0, 2))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 23);
  _mm256_storeu_si256(
      out + 28,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                     _mm256_slli_epi32(w1, 8))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 24);
  _mm256_storeu_si256(
      out + 29, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 18),
                                                  _mm256_slli_epi32(w0, 14))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 25);
  _mm256_storeu_si256(
      out + 30, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 12),
                                                  _mm256_slli_epi32(w1, 20))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 31,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 6), broadcomp),
                       _mm256_set1_epi32(-1)));
}

/* we packed 256 27-bit values, touching 27 256-bit words, using 432 bytes */
static void filterneq27(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  27 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(134217727);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                   _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 27),
                                                     _mm256_slli_epi32(w1, 5))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 2, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w1, 22),
                                                 _mm256_slli_epi32(w0, 10))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 3, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w0, 17),
                                                 _mm256_slli_epi32(w1, 15))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 4, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w1, 12),
                                                 _mm256_slli_epi32(w0, 20))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 5, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w0, 7),
                                                 _mm256_slli_epi32(w1, 25))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 6,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 2)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 7,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 29),
                                                     _mm256_slli_epi32(w0, 3))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 8,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                     _mm256_slli_epi32(w1, 8))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 9, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w1, 19),
                                                 _mm256_slli_epi32(w0, 13))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 10, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 14),
                                                  _mm256_slli_epi32(w1, 18))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 11, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 9),
                                                  _mm256_slli_epi32(w0, 23))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 12,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 13,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 31),
                                                     _mm256_slli_epi32(w1, 1))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 14,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                     _mm256_slli_epi32(w0, 6))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 15, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 21),
                                                  _mm256_slli_epi32(w1, 11))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 16, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                  _mm256_slli_epi32(w0, 16))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 17, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 11),
                                                  _mm256_slli_epi32(w1, 21))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 18, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 6),
                                                  _mm256_slli_epi32(w0, 26))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 19,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 1)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 20,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                     _mm256_slli_epi32(w1, 4))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 21,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 23),
                                                     _mm256_slli_epi32(w0, 9))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 22, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 18),
                                                  _mm256_slli_epi32(w1, 14))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(
      out + 23, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 13),
                                                  _mm256_slli_epi32(w0, 19))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 21);
  _mm256_storeu_si256(
      out + 24, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 8),
                                                  _mm256_slli_epi32(w1, 24))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 25,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 3)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 22);
  _mm256_storeu_si256(
      out + 26,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                     _mm256_slli_epi32(w0, 2))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 23);
  _mm256_storeu_si256(
      out + 27,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 25),
                                                     _mm256_slli_epi32(w1, 7))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 24);
  _mm256_storeu_si256(
      out + 28, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                  _mm256_slli_epi32(w0, 12))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 25);
  _mm256_storeu_si256(
      out + 29, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 15),
                                                  _mm256_slli_epi32(w1, 17))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 26);
  _mm256_storeu_si256(
      out + 30, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 10),
                                                  _mm256_slli_epi32(w0, 22))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 31,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 5), broadcomp),
                       _mm256_set1_epi32(-1)));
}

/* we packed 256 28-bit values, touching 28 256-bit words, using 448 bytes */
static void filterneq28(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  28 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(268435455);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                   _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                     _mm256_slli_epi32(w1, 4))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 2,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                     _mm256_slli_epi32(w0, 8))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 3, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                 _mm256_slli_epi32(w1, 12))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 4, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 5, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w0, 12),
                                                 _mm256_slli_epi32(w1, 20))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 6, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w1, 8),
                                                 _mm256_slli_epi32(w0, 24))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 7,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 4), broadcomp),
                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 8, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp),
                   _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 9,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                     _mm256_slli_epi32(w0, 4))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 10,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                     _mm256_slli_epi32(w1, 8))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 11, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                  _mm256_slli_epi32(w0, 12))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 12, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                  _mm256_slli_epi32(w1, 16))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 13, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 12),
                                                  _mm256_slli_epi32(w0, 20))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 14, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 8),
                                                  _mm256_slli_epi32(w1, 24))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 15,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 4), broadcomp),
                       _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 16, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                    _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 17,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                     _mm256_slli_epi32(w1, 4))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 18,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                     _mm256_slli_epi32(w0, 8))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 19, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                  _mm256_slli_epi32(w1, 12))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 20, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                  _mm256_slli_epi32(w0, 16))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 21, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 12),
                                                  _mm256_slli_epi32(w1, 20))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(
      out + 22, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 8),
                                                  _mm256_slli_epi32(w0, 24))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 23,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 4), broadcomp),
                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 21);
  _mm256_storeu_si256(
      out + 24, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 22);
  _mm256_storeu_si256(
      out + 25,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                     _mm256_slli_epi32(w0, 4))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 23);
  _mm256_storeu_si256(
      out + 26,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                     _mm256_slli_epi32(w1, 8))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 24);
  _mm256_storeu_si256(
      out + 27, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                  _mm256_slli_epi32(w0, 12))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 25);
  _mm256_storeu_si256(
      out + 28, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                  _mm256_slli_epi32(w1, 16))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 26);
  _mm256_storeu_si256(
      out + 29, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 12),
                                                  _mm256_slli_epi32(w0, 20))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 27);
  _mm256_storeu_si256(
      out + 30, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 8),
                                                  _mm256_slli_epi32(w1, 24))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 31,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 4), broadcomp),
                       _mm256_set1_epi32(-1)));
}

/* we packed 256 29-bit values, touching 29 256-bit words, using 464 bytes */
static void filterneq29(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  29 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(536870911);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                   _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 29),
                                                     _mm256_slli_epi32(w1, 3))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 2,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                     _mm256_slli_epi32(w0, 6))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 3,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 23),
                                                     _mm256_slli_epi32(w1, 9))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 4, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 5, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w0, 17),
                                                 _mm256_slli_epi32(w1, 15))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 6, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w1, 14),
                                                 _mm256_slli_epi32(w0, 18))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 7, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w0, 11),
                                                 _mm256_slli_epi32(w1, 21))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 8, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w1, 8),
                                                 _mm256_slli_epi32(w0, 24))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 9, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w0, 5),
                                                 _mm256_slli_epi32(w1, 27))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 10,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 2)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 11,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 31),
                                                     _mm256_slli_epi32(w0, 1))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 12,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                     _mm256_slli_epi32(w1, 4))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 13,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 25),
                                                     _mm256_slli_epi32(w0, 7))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 14, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 22),
                                                  _mm256_slli_epi32(w1, 10))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 15, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 19),
                                                  _mm256_slli_epi32(w0, 13))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 16, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                  _mm256_slli_epi32(w1, 16))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 17, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 13),
                                                  _mm256_slli_epi32(w0, 19))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 18, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 10),
                                                  _mm256_slli_epi32(w1, 22))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 19, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 7),
                                                  _mm256_slli_epi32(w0, 25))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 20, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 4),
                                                  _mm256_slli_epi32(w1, 28))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 21,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 1)),
                             broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(
      out + 22,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                     _mm256_slli_epi32(w0, 2))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 21);
  _mm256_storeu_si256(
      out + 23,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 27),
                                                     _mm256_slli_epi32(w1, 5))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 22);
  _mm256_storeu_si256(
      out + 24,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                     _mm256_slli_epi32(w0, 8))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 23);
  _mm256_storeu_si256(
      out + 25, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 21),
                                                  _mm256_slli_epi32(w1, 11))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 24);
  _mm256_storeu_si256(
      out + 26, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 18),
                                                  _mm256_slli_epi32(w0, 14))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 25);
  _mm256_storeu_si256(
      out + 27, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 15),
                                                  _mm256_slli_epi32(w1, 17))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 26);
  _mm256_storeu_si256(
      out + 28, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 12),
                                                  _mm256_slli_epi32(w0, 20))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 27);
  _mm256_storeu_si256(
      out + 29, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 9),
                                                  _mm256_slli_epi32(w1, 23))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 28);
  _mm256_storeu_si256(
      out + 30, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 6),
                                                  _mm256_slli_epi32(w0, 26))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 31,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 3), broadcomp),
                       _mm256_set1_epi32(-1)));
}

/* we packed 256 30-bit values, touching 30 256-bit words, using 480 bytes */
static void filterneq30(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  30 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(1073741823);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                   _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                     _mm256_slli_epi32(w1, 2))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 2,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                     _mm256_slli_epi32(w0, 4))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 3,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 26),
                                                     _mm256_slli_epi32(w1, 6))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 4,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                     _mm256_slli_epi32(w0, 8))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 5, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w0, 22),
                                                 _mm256_slli_epi32(w1, 10))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 6, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 7, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w0, 18),
                                                 _mm256_slli_epi32(w1, 14))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 8, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 9, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(
                       _mm256_and_si256(
                           mask, _mm256_or_si256(_mm256_srli_epi32(w0, 14),
                                                 _mm256_slli_epi32(w1, 18))),
                       broadcomp),
                   _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 10, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 12),
                                                  _mm256_slli_epi32(w0, 20))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 11, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 10),
                                                  _mm256_slli_epi32(w1, 22))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 12, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 8),
                                                  _mm256_slli_epi32(w0, 24))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 13, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 6),
                                                  _mm256_slli_epi32(w1, 26))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 14, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 4),
                                                  _mm256_slli_epi32(w0, 28))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 15,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 2), broadcomp),
                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 16, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(_mm256_and_si256(mask, w1), broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 17,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                     _mm256_slli_epi32(w0, 2))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 18,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                     _mm256_slli_epi32(w1, 4))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 19,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                     _mm256_slli_epi32(w0, 6))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 20,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                     _mm256_slli_epi32(w1, 8))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(
      out + 21, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 22),
                                                  _mm256_slli_epi32(w0, 10))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 21);
  _mm256_storeu_si256(
      out + 22, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                  _mm256_slli_epi32(w1, 12))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 22);
  _mm256_storeu_si256(
      out + 23, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 18),
                                                  _mm256_slli_epi32(w0, 14))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 23);
  _mm256_storeu_si256(
      out + 24, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                  _mm256_slli_epi32(w1, 16))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 24);
  _mm256_storeu_si256(
      out + 25, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 14),
                                                  _mm256_slli_epi32(w0, 18))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 25);
  _mm256_storeu_si256(
      out + 26, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 12),
                                                  _mm256_slli_epi32(w1, 20))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 26);
  _mm256_storeu_si256(
      out + 27, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 10),
                                                  _mm256_slli_epi32(w0, 22))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 27);
  _mm256_storeu_si256(
      out + 28, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 8),
                                                  _mm256_slli_epi32(w1, 24))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 28);
  _mm256_storeu_si256(
      out + 29, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 6),
                                                  _mm256_slli_epi32(w0, 26))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 29);
  _mm256_storeu_si256(
      out + 30, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 4),
                                                  _mm256_slli_epi32(w1, 28))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 31,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w1, 2), broadcomp),
                       _mm256_set1_epi32(-1)));
}

/* we packed 256 31-bit values, touching 31 256-bit words, using 496 bytes */
static void filterneq31(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  31 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(2147483647);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_xor_si256(
                   _mm256_cmpeq_epi32(_mm256_and_si256(mask, w0), broadcomp),
                   _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 31),
                                                     _mm256_slli_epi32(w1, 1))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 2,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                     _mm256_slli_epi32(w0, 2))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 3,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 29),
                                                     _mm256_slli_epi32(w1, 3))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 4,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                     _mm256_slli_epi32(w0, 4))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 5,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 27),
                                                     _mm256_slli_epi32(w1, 5))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 6,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                     _mm256_slli_epi32(w0, 6))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 7,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 25),
                                                     _mm256_slli_epi32(w1, 7))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 8,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                     _mm256_slli_epi32(w0, 8))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 9,
      _mm256_xor_si256(
          _mm256_cmpeq_epi32(
              _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 23),
                                                     _mm256_slli_epi32(w1, 9))),
              broadcomp),
          _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 10, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 22),
                                                  _mm256_slli_epi32(w0, 10))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 11, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 21),
                                                  _mm256_slli_epi32(w1, 11))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 12, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                  _mm256_slli_epi32(w0, 12))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 13, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 19),
                                                  _mm256_slli_epi32(w1, 13))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 14, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 18),
                                                  _mm256_slli_epi32(w0, 14))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 15, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 17),
                                                  _mm256_slli_epi32(w1, 15))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 16, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                  _mm256_slli_epi32(w0, 16))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 17, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 15),
                                                  _mm256_slli_epi32(w1, 17))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 18, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 14),
                                                  _mm256_slli_epi32(w0, 18))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 19, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 13),
                                                  _mm256_slli_epi32(w1, 19))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(
      out + 20, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 12),
                                                  _mm256_slli_epi32(w0, 20))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 21);
  _mm256_storeu_si256(
      out + 21, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 11),
                                                  _mm256_slli_epi32(w1, 21))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 22);
  _mm256_storeu_si256(
      out + 22, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 10),
                                                  _mm256_slli_epi32(w0, 22))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 23);
  _mm256_storeu_si256(
      out + 23, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 9),
                                                  _mm256_slli_epi32(w1, 23))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 24);
  _mm256_storeu_si256(
      out + 24, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 8),
                                                  _mm256_slli_epi32(w0, 24))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 25);
  _mm256_storeu_si256(
      out + 25, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 7),
                                                  _mm256_slli_epi32(w1, 25))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 26);
  _mm256_storeu_si256(
      out + 26, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 6),
                                                  _mm256_slli_epi32(w0, 26))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 27);
  _mm256_storeu_si256(
      out + 27, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 5),
                                                  _mm256_slli_epi32(w1, 27))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 28);
  _mm256_storeu_si256(
      out + 28, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 4),
                                                  _mm256_slli_epi32(w0, 28))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 29);
  _mm256_storeu_si256(
      out + 29, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w0, 3),
                                                  _mm256_slli_epi32(w1, 29))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 30);
  _mm256_storeu_si256(
      out + 30, _mm256_xor_si256(
                    _mm256_cmpeq_epi32(
                        _mm256_and_si256(
                            mask, _mm256_or_si256(_mm256_srli_epi32(w1, 2),
                                                  _mm256_slli_epi32(w0, 30))),
                        broadcomp),
                    _mm256_set1_epi32(-1)));
  _mm256_storeu_si256(
      out + 31,
      _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_srli_epi32(w0, 1), broadcomp),
                       _mm256_set1_epi32(-1)));
}

/* we packed 256 32-bit values, touching 32 256-bit words, using 512 bytes */
static void filterneq32(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  32 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(out + 0,
                      _mm256_xor_si256(_mm256_cmpeq_epi32(w0, broadcomp),
                                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(out + 1,
                      _mm256_xor_si256(_mm256_cmpeq_epi32(w1, broadcomp),
                                       _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(out + 2,
                      _mm256_xor_si256(_mm256_cmpeq_epi32(w0, broadcomp),
                                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(out + 3,
                      _mm256_xor_si256(_mm256_cmpeq_epi32(w1, broadcomp),
                                       _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(out + 4,
                      _mm256_xor_si256(_mm256_cmpeq_epi32(w0, broadcomp),
                                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(out + 5,
                      _mm256_xor_si256(_mm256_cmpeq_epi32(w1, broadcomp),
                                       _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(out + 6,
                      _mm256_xor_si256(_mm256_cmpeq_epi32(w0, broadcomp),
                                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(out + 7,
                      _mm256_xor_si256(_mm256_cmpeq_epi32(w1, broadcomp),
                                       _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(out + 8,
                      _mm256_xor_si256(_mm256_cmpeq_epi32(w0, broadcomp),
                                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(out + 9,
                      _mm256_xor_si256(_mm256_cmpeq_epi32(w1, broadcomp),
                                       _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(out + 10,
                      _mm256_xor_si256(_mm256_cmpeq_epi32(w0, broadcomp),
                                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(out + 11,
                      _mm256_xor_si256(_mm256_cmpeq_epi32(w1, broadcomp),
                                       _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(out + 12,
                      _mm256_xor_si256(_mm256_cmpeq_epi32(w0, broadcomp),
                                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(out + 13,
                      _mm256_xor_si256(_mm256_cmpeq_epi32(w1, broadcomp),
                                       _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(out + 14,
                      _mm256_xor_si256(_mm256_cmpeq_epi32(w0, broadcomp),
                                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(out + 15,
                      _mm256_xor_si256(_mm256_cmpeq_epi32(w1, broadcomp),
                                       _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(out + 16,
                      _mm256_xor_si256(_mm256_cmpeq_epi32(w0, broadcomp),
                                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(out + 17,
                      _mm256_xor_si256(_mm256_cmpeq_epi32(w1, broadcomp),
                                       _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(out + 18,
                      _mm256_xor_si256(_mm256_cmpeq_epi32(w0, broadcomp),
                                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(out + 19,
                      _mm256_xor_si256(_mm256_cmpeq_epi32(w1, broadcomp),
                                       _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(out + 20,
                      _mm256_xor_si256(_mm256_cmpeq_epi32(w0, broadcomp),
                                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 21);
  _mm256_storeu_si256(out + 21,
                      _mm256_xor_si256(_mm256_cmpeq_epi32(w1, broadcomp),
                                       _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 22);
  _mm256_storeu_si256(out + 22,
                      _mm256_xor_si256(_mm256_cmpeq_epi32(w0, broadcomp),
                                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 23);
  _mm256_storeu_si256(out + 23,
                      _mm256_xor_si256(_mm256_cmpeq_epi32(w1, broadcomp),
                                       _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 24);
  _mm256_storeu_si256(out + 24,
                      _mm256_xor_si256(_mm256_cmpeq_epi32(w0, broadcomp),
                                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 25);
  _mm256_storeu_si256(out + 25,
                      _mm256_xor_si256(_mm256_cmpeq_epi32(w1, broadcomp),
                                       _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 26);
  _mm256_storeu_si256(out + 26,
                      _mm256_xor_si256(_mm256_cmpeq_epi32(w0, broadcomp),
                                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 27);
  _mm256_storeu_si256(out + 27,
                      _mm256_xor_si256(_mm256_cmpeq_epi32(w1, broadcomp),
                                       _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 28);
  _mm256_storeu_si256(out + 28,
                      _mm256_xor_si256(_mm256_cmpeq_epi32(w0, broadcomp),
                                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 29);
  _mm256_storeu_si256(out + 29,
                      _mm256_xor_si256(_mm256_cmpeq_epi32(w1, broadcomp),
                                       _mm256_set1_epi32(-1)));
  w0 = _mm256_lddqu_si256(in + 30);
  _mm256_storeu_si256(out + 30,
                      _mm256_xor_si256(_mm256_cmpeq_epi32(w0, broadcomp),
                                       _mm256_set1_epi32(-1)));
  w1 = _mm256_lddqu_si256(in + 31);
  _mm256_storeu_si256(out + 31,
                      _mm256_xor_si256(_mm256_cmpeq_epi32(w1, broadcomp),
                                       _mm256_set1_epi32(-1)));
}

static void filtergt0(const __m256i *in, u32 *matches, const INTEGER comp) {
  if (comp < 0)
    memset(matches, 1, 256 * sizeof(*matches));
  else
    memset(matches, 0, 256 * sizeof(*matches));
}

/* we packed 256 1-bit values, touching 1 256-bit words, using 16 bytes */
static void filtergt1(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  1 256-bit word */
  __m256i w0;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(1);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 1)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 2)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 3)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 5)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 6)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 7)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 9)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 10)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 11)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 12)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 13)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 14)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 15)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 17)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 18)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 19)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 20)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 21)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 22)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 23)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 24)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 25)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 26)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 27)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 28)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 29)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 30)),
                         broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 31), broadcomp));
}

/* we packed 256 2-bit values, touching 2 256-bit words, using 32 bytes */
static void filtergt2(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  2 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(3);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 2)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 6)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 10)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 12)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 14)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 18)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 20)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 22)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 24)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 26)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 28)),
                         broadcomp));
  _mm256_storeu_si256(out + 15,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 30), broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 16, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w1), broadcomp));
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 2)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 6)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 10)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 12)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 14)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 18)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 20)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 22)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 24)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 26)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 28)),
                         broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w1, 30), broadcomp));
}

/* we packed 256 3-bit values, touching 3 256-bit words, using 48 bytes */
static void filtergt3(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  3 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(7);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 3)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 6)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 9)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 12)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 15)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 18)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 21)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 24)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 27)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                 _mm256_slli_epi32(w1, 2))),
          broadcomp));
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 1)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 7)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 10)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 13)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 19)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 22)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 25)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 28)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 31),
                                                 _mm256_slli_epi32(w0, 1))),
          broadcomp));
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 2)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 5)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 11)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 14)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 17)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 20)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 23)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 26)),
                         broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 29), broadcomp));
}

/* we packed 256 4-bit values, touching 4 256-bit words, using 64 bytes */
static void filtergt4(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  4 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(15);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 12)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 20)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 24)),
                         broadcomp));
  _mm256_storeu_si256(out + 7,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 28), broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 8, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w1), broadcomp));
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 12)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 20)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 24)),
                         broadcomp));
  _mm256_storeu_si256(out + 15,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w1, 28), broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 16, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 12)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 20)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 24)),
                         broadcomp));
  _mm256_storeu_si256(out + 23,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 28), broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 24, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w1), broadcomp));
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 12)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 20)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 24)),
                         broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w1, 28), broadcomp));
}

/* we packed 256 5-bit values, touching 5 256-bit words, using 80 bytes */
static void filtergt5(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  5 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(31);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 5)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 10)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 15)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 20)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 25)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                 _mm256_slli_epi32(w1, 2))),
          broadcomp));
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 3)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 13)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 18)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 23)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4))),
          broadcomp));
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 1)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 6)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 11)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 21)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 26)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 31),
                                                 _mm256_slli_epi32(w1, 1))),
          broadcomp));
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 9)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 14)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 19)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 24)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 29),
                                                 _mm256_slli_epi32(w0, 3))),
          broadcomp));
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 2)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 7)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 12)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 17)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 22)),
                         broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 27), broadcomp));
}

/* we packed 256 6-bit values, touching 6 256-bit words, using 96 bytes */
static void filtergt6(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  6 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(63);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 6)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 12)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 18)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 24)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                 _mm256_slli_epi32(w1, 2))),
          broadcomp));
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 10)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 22)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4))),
          broadcomp));
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 2)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 14)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 20)),
                         broadcomp));
  _mm256_storeu_si256(out + 15,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 26), broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 16, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w1), broadcomp));
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 6)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 12)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 18)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 24)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                 _mm256_slli_epi32(w0, 2))),
          broadcomp));
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 10)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 22)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4))),
          broadcomp));
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 2)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 14)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 20)),
                         broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w1, 26), broadcomp));
}

/* we packed 256 7-bit values, touching 7 256-bit words, using 112 bytes */
static void filtergt7(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  7 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(127);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 7)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 14)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 21)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4))),
          broadcomp));
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 3)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 10)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 17)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 24)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 31),
                                                 _mm256_slli_epi32(w0, 1))),
          broadcomp));
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 6)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 13)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 20)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 27),
                                                 _mm256_slli_epi32(w1, 5))),
          broadcomp));
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 2)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 9)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 23)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                 _mm256_slli_epi32(w0, 2))),
          broadcomp));
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 5)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 12)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 19)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 26),
                                                 _mm256_slli_epi32(w1, 6))),
          broadcomp));
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 1)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 15)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 22)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 29),
                                                 _mm256_slli_epi32(w0, 3))),
          broadcomp));
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 11)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 18)),
                         broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 25), broadcomp));
}

/* we packed 256 8-bit values, touching 8 256-bit words, using 128 bytes */
static void filtergt8(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  8 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(255);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                         broadcomp));
  _mm256_storeu_si256(out + 3,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 24), broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 4, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w1), broadcomp));
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                         broadcomp));
  _mm256_storeu_si256(out + 7,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w1, 24), broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 8, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                         broadcomp));
  _mm256_storeu_si256(out + 11,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 24), broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 12, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w1), broadcomp));
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                         broadcomp));
  _mm256_storeu_si256(out + 15,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w1, 24), broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 16, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                         broadcomp));
  _mm256_storeu_si256(out + 19,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 24), broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 20, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w1), broadcomp));
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                         broadcomp));
  _mm256_storeu_si256(out + 23,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w1, 24), broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 24, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                         broadcomp));
  _mm256_storeu_si256(out + 27,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 24), broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 28, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w1), broadcomp));
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                         broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w1, 24), broadcomp));
}

/* we packed 256 9-bit values, touching 9 256-bit words, using 144 bytes */
static void filtergt9(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  9 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(511);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 9)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 18)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 27),
                                                 _mm256_slli_epi32(w1, 5))),
          broadcomp));
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 13)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 22)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 31),
                                                 _mm256_slli_epi32(w0, 1))),
          broadcomp));
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 17)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 26),
                                                 _mm256_slli_epi32(w1, 6))),
          broadcomp));
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 3)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 12)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 21)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                 _mm256_slli_epi32(w0, 2))),
          broadcomp));
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 7)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 25),
                                                 _mm256_slli_epi32(w1, 7))),
          broadcomp));
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 2)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 11)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 20)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 29),
                                                 _mm256_slli_epi32(w0, 3))),
          broadcomp));
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 6)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 15)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8))),
          broadcomp));
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 1)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 10)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 19)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4))),
          broadcomp));
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 5)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 14)),
                         broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 23), broadcomp));
}

/* we packed 256 10-bit values, touching 10 256-bit words, using 160 bytes */
static void filtergt10(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  10 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(1023);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 10)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 20)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                 _mm256_slli_epi32(w1, 2))),
          broadcomp));
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 18)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4))),
          broadcomp));
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 6)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 26),
                                                 _mm256_slli_epi32(w1, 6))),
          broadcomp));
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 14)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8))),
          broadcomp));
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 2)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 12)),
                         broadcomp));
  _mm256_storeu_si256(out + 15,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 22), broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 16, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w1), broadcomp));
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 10)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 20)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                 _mm256_slli_epi32(w0, 2))),
          broadcomp));
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 18)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4))),
          broadcomp));
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 6)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                 _mm256_slli_epi32(w0, 6))),
          broadcomp));
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 14)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8))),
          broadcomp));
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 2)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 12)),
                         broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w1, 22), broadcomp));
}

/* we packed 256 11-bit values, touching 11 256-bit words, using 176 bytes */
static void filtergt11(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  11 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(2047);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 11)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 22),
                                                 _mm256_slli_epi32(w1, 10))),
          broadcomp));
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 1)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 12)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 23),
                                                 _mm256_slli_epi32(w0, 9))),
          broadcomp));
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 2)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 13)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8))),
          broadcomp));
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 3)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 14)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 25),
                                                 _mm256_slli_epi32(w0, 7))),
          broadcomp));
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 15)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 26),
                                                 _mm256_slli_epi32(w1, 6))),
          broadcomp));
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 5)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 27),
                                                 _mm256_slli_epi32(w0, 5))),
          broadcomp));
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 6)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 17)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4))),
          broadcomp));
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 7)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 18)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 29),
                                                 _mm256_slli_epi32(w0, 3))),
          broadcomp));
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 19)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                 _mm256_slli_epi32(w1, 2))),
          broadcomp));
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 9)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 20)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 31),
                                                 _mm256_slli_epi32(w0, 1))),
          broadcomp));
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 10)),
                         broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 21), broadcomp));
}

/* we packed 256 12-bit values, touching 12 256-bit words, using 192 bytes */
static void filtergt12(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  12 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(4095);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 12)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8))),
          broadcomp));
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4))),
          broadcomp));
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                         broadcomp));
  _mm256_storeu_si256(out + 7,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 20), broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 8, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w1), broadcomp));
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 12)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8))),
          broadcomp));
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4))),
          broadcomp));
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                         broadcomp));
  _mm256_storeu_si256(out + 15,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w1, 20), broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 16, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 12)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8))),
          broadcomp));
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4))),
          broadcomp));
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                         broadcomp));
  _mm256_storeu_si256(out + 23,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 20), broadcomp));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 24, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w1), broadcomp));
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 12)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8))),
          broadcomp));
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4))),
          broadcomp));
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                         broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w1, 20), broadcomp));
}

/* we packed 256 13-bit values, touching 13 256-bit words, using 208 bytes */
static void filtergt13(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  13 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(8191);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 13)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 26),
                                                 _mm256_slli_epi32(w1, 6))),
          broadcomp));
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 7)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12))),
          broadcomp));
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 1)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 14)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 27),
                                                 _mm256_slli_epi32(w1, 5))),
          broadcomp));
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 21),
                                                 _mm256_slli_epi32(w0, 11))),
          broadcomp));
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 2)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 15)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4))),
          broadcomp));
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 9)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 22),
                                                 _mm256_slli_epi32(w0, 10))),
          broadcomp));
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 3)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 29),
                                                 _mm256_slli_epi32(w1, 3))),
          broadcomp));
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 10)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 23),
                                                 _mm256_slli_epi32(w0, 9))),
          broadcomp));
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 17)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                 _mm256_slli_epi32(w1, 2))),
          broadcomp));
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 11)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8))),
          broadcomp));
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 5)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 18)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 31),
                                                 _mm256_slli_epi32(w1, 1))),
          broadcomp));
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 12)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 25),
                                                 _mm256_slli_epi32(w0, 7))),
          broadcomp));
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 6)),
                         broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 19), broadcomp));
}

/* we packed 256 14-bit values, touching 14 256-bit words, using 224 bytes */
static void filtergt14(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  14 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(16383);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 14)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4))),
          broadcomp));
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 10)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8))),
          broadcomp));
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 6)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                 _mm256_slli_epi32(w1, 12))),
          broadcomp));
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 2)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                 _mm256_slli_epi32(w0, 2))),
          broadcomp));
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 12)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 26),
                                                 _mm256_slli_epi32(w1, 6))),
          broadcomp));
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 22),
                                                 _mm256_slli_epi32(w0, 10))),
          broadcomp));
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                         broadcomp));
  _mm256_storeu_si256(out + 15,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 18), broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 16, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w1), broadcomp));
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 14)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4))),
          broadcomp));
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 10)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8))),
          broadcomp));
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 6)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12))),
          broadcomp));
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 2)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 16)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                 _mm256_slli_epi32(w1, 2))),
          broadcomp));
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 12)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                 _mm256_slli_epi32(w0, 6))),
          broadcomp));
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 22),
                                                 _mm256_slli_epi32(w1, 10))),
          broadcomp));
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                         broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w1, 18), broadcomp));
}

/* we packed 256 15-bit values, touching 15 256-bit words, using 240 bytes */
static void filtergt15(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  15 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(32767);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 15)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                 _mm256_slli_epi32(w1, 2))),
          broadcomp));
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 13)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4))),
          broadcomp));
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 11)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 26),
                                                 _mm256_slli_epi32(w1, 6))),
          broadcomp));
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 9)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8))),
          broadcomp));
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 7)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 22),
                                                 _mm256_slli_epi32(w1, 10))),
          broadcomp));
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 5)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12))),
          broadcomp));
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 3)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 18),
                                                 _mm256_slli_epi32(w1, 14))),
          broadcomp));
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 1)),
                         broadcomp));
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 16)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 31),
                                                 _mm256_slli_epi32(w0, 1))),
          broadcomp));
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 14)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 29),
                                                 _mm256_slli_epi32(w1, 3))),
          broadcomp));
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 12)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 27),
                                                 _mm256_slli_epi32(w0, 5))),
          broadcomp));
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 10)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 25),
                                                 _mm256_slli_epi32(w1, 7))),
          broadcomp));
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 23),
                                                 _mm256_slli_epi32(w0, 9))),
          broadcomp));
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 6)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 21),
                                                 _mm256_slli_epi32(w1, 11))),
          broadcomp));
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 19),
                                                 _mm256_slli_epi32(w0, 13))),
          broadcomp));
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 2)),
                         broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 17), broadcomp));
}

/* we packed 256 16-bit values, touching 16 256-bit words, using 256 bytes */
static void filtergt16(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  16 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(65535);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(out + 1,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 16), broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 2, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w1), broadcomp));
  _mm256_storeu_si256(out + 3,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w1, 16), broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 4, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(out + 5,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 16), broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 6, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w1), broadcomp));
  _mm256_storeu_si256(out + 7,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w1, 16), broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 8, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(out + 9,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 16), broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 10, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w1), broadcomp));
  _mm256_storeu_si256(out + 11,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w1, 16), broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 12, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(out + 13,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 16), broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 14, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w1), broadcomp));
  _mm256_storeu_si256(out + 15,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w1, 16), broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 16, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(out + 17,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 16), broadcomp));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 18, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w1), broadcomp));
  _mm256_storeu_si256(out + 19,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w1, 16), broadcomp));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 20, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(out + 21,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 16), broadcomp));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 22, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w1), broadcomp));
  _mm256_storeu_si256(out + 23,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w1, 16), broadcomp));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 24, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(out + 25,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 16), broadcomp));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 26, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w1), broadcomp));
  _mm256_storeu_si256(out + 27,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w1, 16), broadcomp));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 28, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  _mm256_storeu_si256(out + 29,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 16), broadcomp));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 30, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w1), broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w1, 16), broadcomp));
}

/* we packed 256 17-bit values, touching 17 256-bit words, using 272 bytes */
static void filtergt17(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  17 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(131071);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 17),
                                                 _mm256_slli_epi32(w1, 15))),
          broadcomp));
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 2)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 19),
                                                 _mm256_slli_epi32(w0, 13))),
          broadcomp));
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 21),
                                                 _mm256_slli_epi32(w1, 11))),
          broadcomp));
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 6)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 23),
                                                 _mm256_slli_epi32(w0, 9))),
          broadcomp));
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 25),
                                                 _mm256_slli_epi32(w1, 7))),
          broadcomp));
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 10)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 27),
                                                 _mm256_slli_epi32(w0, 5))),
          broadcomp));
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 12)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 29),
                                                 _mm256_slli_epi32(w1, 3))),
          broadcomp));
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 14)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 31),
                                                 _mm256_slli_epi32(w0, 1))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16))),
          broadcomp));
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 1)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 18),
                                                 _mm256_slli_epi32(w0, 14))),
          broadcomp));
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 3)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                 _mm256_slli_epi32(w1, 12))),
          broadcomp));
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 5)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 22),
                                                 _mm256_slli_epi32(w0, 10))),
          broadcomp));
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 7)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8))),
          broadcomp));
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 9)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                 _mm256_slli_epi32(w0, 6))),
          broadcomp));
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 11)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4))),
          broadcomp));
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 13)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                 _mm256_slli_epi32(w0, 2))),
          broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 15), broadcomp));
}

/* we packed 256 18-bit values, touching 18 256-bit words, using 288 bytes */
static void filtergt18(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  18 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(262143);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 18),
                                                 _mm256_slli_epi32(w1, 14))),
          broadcomp));
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 22),
                                                 _mm256_slli_epi32(w0, 10))),
          broadcomp));
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 26),
                                                 _mm256_slli_epi32(w1, 6))),
          broadcomp));
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 12)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                 _mm256_slli_epi32(w0, 2))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16))),
          broadcomp));
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 2)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12))),
          broadcomp));
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 6)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8))),
          broadcomp));
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 10)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4))),
          broadcomp));
  _mm256_storeu_si256(out + 15,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 14), broadcomp));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 16, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w1), broadcomp));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 18),
                                                 _mm256_slli_epi32(w0, 14))),
          broadcomp));
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 22),
                                                 _mm256_slli_epi32(w1, 10))),
          broadcomp));
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                 _mm256_slli_epi32(w0, 6))),
          broadcomp));
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 12)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                 _mm256_slli_epi32(w1, 2))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16))),
          broadcomp));
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 2)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                 _mm256_slli_epi32(w1, 12))),
          broadcomp));
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 6)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8))),
          broadcomp));
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 10)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4))),
          broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w1, 14), broadcomp));
}

/* we packed 256 19-bit values, touching 19 256-bit words, using 304 bytes */
static void filtergt19(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  19 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(524287);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 19),
                                                 _mm256_slli_epi32(w1, 13))),
          broadcomp));
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 6)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 25),
                                                 _mm256_slli_epi32(w0, 7))),
          broadcomp));
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 12)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 31),
                                                 _mm256_slli_epi32(w1, 1))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 18),
                                                 _mm256_slli_epi32(w0, 14))),
          broadcomp));
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 5)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8))),
          broadcomp));
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 11)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                 _mm256_slli_epi32(w0, 2))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 17),
                                                 _mm256_slli_epi32(w1, 15))),
          broadcomp));
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 23),
                                                 _mm256_slli_epi32(w0, 9))),
          broadcomp));
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 10)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 29),
                                                 _mm256_slli_epi32(w1, 3))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16))),
          broadcomp));
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 3)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 22),
                                                 _mm256_slli_epi32(w1, 10))),
          broadcomp));
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 9)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 15),
                                                 _mm256_slli_epi32(w1, 17))),
          broadcomp));
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 2)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 21),
                                                 _mm256_slli_epi32(w0, 11))),
          broadcomp));
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 27),
                                                 _mm256_slli_epi32(w1, 5))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 14),
                                                 _mm256_slli_epi32(w0, 18))),
          broadcomp));
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 1)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                 _mm256_slli_epi32(w1, 12))),
          broadcomp));
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 7)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                 _mm256_slli_epi32(w0, 6))),
          broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 13), broadcomp));
}

/* we packed 256 20-bit values, touching 20 256-bit words, using 320 bytes */
static void filtergt20(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  20 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(1048575);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                 _mm256_slli_epi32(w1, 12))),
          broadcomp));
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16))),
          broadcomp));
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8))),
          broadcomp));
  _mm256_storeu_si256(out + 7,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 12), broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 8, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w1), broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12))),
          broadcomp));
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16))),
          broadcomp));
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8))),
          broadcomp));
  _mm256_storeu_si256(out + 15,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w1, 12), broadcomp));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 16, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                 _mm256_slli_epi32(w1, 12))),
          broadcomp));
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16))),
          broadcomp));
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8))),
          broadcomp));
  _mm256_storeu_si256(out + 23,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 12), broadcomp));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 24, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w1), broadcomp));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12))),
          broadcomp));
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16))),
          broadcomp));
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8))),
          broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w1, 12), broadcomp));
}

/* we packed 256 21-bit values, touching 21 256-bit words, using 336 bytes */
static void filtergt21(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  21 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(2097151);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 21),
                                                 _mm256_slli_epi32(w1, 11))),
          broadcomp));
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 10)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 31),
                                                 _mm256_slli_epi32(w0, 1))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                 _mm256_slli_epi32(w1, 12))),
          broadcomp));
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 9)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                 _mm256_slli_epi32(w0, 2))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 19),
                                                 _mm256_slli_epi32(w1, 13))),
          broadcomp));
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 29),
                                                 _mm256_slli_epi32(w0, 3))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 18),
                                                 _mm256_slli_epi32(w1, 14))),
          broadcomp));
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 7)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 17),
                                                 _mm256_slli_epi32(w1, 15))),
          broadcomp));
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 6)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 27),
                                                 _mm256_slli_epi32(w0, 5))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16))),
          broadcomp));
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 5)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                 _mm256_slli_epi32(w0, 6))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 15),
                                                 _mm256_slli_epi32(w1, 17))),
          broadcomp));
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 25),
                                                 _mm256_slli_epi32(w0, 7))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 14),
                                                 _mm256_slli_epi32(w1, 18))),
          broadcomp));
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 3)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 13),
                                                 _mm256_slli_epi32(w1, 19))),
          broadcomp));
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 2)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 23),
                                                 _mm256_slli_epi32(w0, 9))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 12),
                                                 _mm256_slli_epi32(w1, 20))),
          broadcomp));
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 1)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 22),
                                                 _mm256_slli_epi32(w0, 10))),
          broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 11), broadcomp));
}

/* we packed 256 22-bit values, touching 22 256-bit words, using 352 bytes */
static void filtergt22(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  22 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(4194303);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 22),
                                                 _mm256_slli_epi32(w1, 10))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 12),
                                                 _mm256_slli_epi32(w0, 20))),
          broadcomp));
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 2)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 14),
                                                 _mm256_slli_epi32(w0, 18))),
          broadcomp));
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 26),
                                                 _mm256_slli_epi32(w1, 6))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16))),
          broadcomp));
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 6)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 18),
                                                 _mm256_slli_epi32(w0, 14))),
          broadcomp));
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 8)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                 _mm256_slli_epi32(w1, 2))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12))),
          broadcomp));
  _mm256_storeu_si256(out + 15,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 10), broadcomp));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 16, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w1), broadcomp));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 22),
                                                 _mm256_slli_epi32(w0, 10))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 12),
                                                 _mm256_slli_epi32(w1, 20))),
          broadcomp));
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 2)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 14),
                                                 _mm256_slli_epi32(w1, 18))),
          broadcomp));
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                 _mm256_slli_epi32(w0, 6))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16))),
          broadcomp));
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 6)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 18),
                                                 _mm256_slli_epi32(w1, 14))),
          broadcomp));
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                 _mm256_slli_epi32(w0, 2))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 21);
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                 _mm256_slli_epi32(w1, 12))),
          broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w1, 10), broadcomp));
}

/* we packed 256 23-bit values, touching 23 256-bit words, using 368 bytes */
static void filtergt23(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  23 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(8388607);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 23),
                                                 _mm256_slli_epi32(w1, 9))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 14),
                                                 _mm256_slli_epi32(w0, 18))),
          broadcomp));
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 5)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 19),
                                                 _mm256_slli_epi32(w0, 13))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 10),
                                                 _mm256_slli_epi32(w1, 22))),
          broadcomp));
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 1)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 15),
                                                 _mm256_slli_epi32(w1, 17))),
          broadcomp));
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 6)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 29),
                                                 _mm256_slli_epi32(w0, 3))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                 _mm256_slli_epi32(w1, 12))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 11),
                                                 _mm256_slli_epi32(w0, 21))),
          broadcomp));
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 2)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 25),
                                                 _mm256_slli_epi32(w1, 7))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16))),
          broadcomp));
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 7)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                 _mm256_slli_epi32(w1, 2))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 21),
                                                 _mm256_slli_epi32(w0, 11))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 12),
                                                 _mm256_slli_epi32(w1, 20))),
          broadcomp));
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 3)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                 _mm256_slli_epi32(w0, 6))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 17),
                                                 _mm256_slli_epi32(w1, 15))),
          broadcomp));
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 8)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 31),
                                                 _mm256_slli_epi32(w0, 1))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 22),
                                                 _mm256_slli_epi32(w1, 10))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 13),
                                                 _mm256_slli_epi32(w0, 19))),
          broadcomp));
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 21);
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 27),
                                                 _mm256_slli_epi32(w1, 5))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 22);
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 18),
                                                 _mm256_slli_epi32(w0, 14))),
          broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 9), broadcomp));
}

/* we packed 256 24-bit values, touching 24 256-bit words, using 384 bytes */
static void filtergt24(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  24 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(16777215);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16))),
          broadcomp));
  _mm256_storeu_si256(out + 3,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 8), broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 4, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w1), broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16))),
          broadcomp));
  _mm256_storeu_si256(out + 7,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w1, 8), broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 8, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16))),
          broadcomp));
  _mm256_storeu_si256(out + 11,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 8), broadcomp));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 12, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w1), broadcomp));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16))),
          broadcomp));
  _mm256_storeu_si256(out + 15,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w1, 8), broadcomp));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 16, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16))),
          broadcomp));
  _mm256_storeu_si256(out + 19,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 8), broadcomp));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 20, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w1), broadcomp));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16))),
          broadcomp));
  _mm256_storeu_si256(out + 23,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w1, 8), broadcomp));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 24, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16))),
          broadcomp));
  _mm256_storeu_si256(out + 27,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 8), broadcomp));
  w1 = _mm256_lddqu_si256(in + 21);
  _mm256_storeu_si256(
      out + 28, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w1), broadcomp));
  w0 = _mm256_lddqu_si256(in + 22);
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 23);
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16))),
          broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w1, 8), broadcomp));
}

/* we packed 256 25-bit values, touching 25 256-bit words, using 400 bytes */
static void filtergt25(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  25 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(33554431);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 25),
                                                 _mm256_slli_epi32(w1, 7))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 18),
                                                 _mm256_slli_epi32(w0, 14))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 11),
                                                 _mm256_slli_epi32(w1, 21))),
          broadcomp));
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 29),
                                                 _mm256_slli_epi32(w0, 3))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 22),
                                                 _mm256_slli_epi32(w1, 10))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 15),
                                                 _mm256_slli_epi32(w0, 17))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 8),
                                                 _mm256_slli_epi32(w1, 24))),
          broadcomp));
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 1)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                 _mm256_slli_epi32(w0, 6))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 19),
                                                 _mm256_slli_epi32(w1, 13))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 12),
                                                 _mm256_slli_epi32(w0, 20))),
          broadcomp));
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 5)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                 _mm256_slli_epi32(w1, 2))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 23),
                                                 _mm256_slli_epi32(w0, 9))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 9),
                                                 _mm256_slli_epi32(w0, 23))),
          broadcomp));
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 2)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 27),
                                                 _mm256_slli_epi32(w1, 5))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 13),
                                                 _mm256_slli_epi32(w1, 19))),
          broadcomp));
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 6)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 31),
                                                 _mm256_slli_epi32(w0, 1))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 17),
                                                 _mm256_slli_epi32(w0, 15))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 21);
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 10),
                                                 _mm256_slli_epi32(w1, 22))),
          broadcomp));
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 3)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 22);
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 23);
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 21),
                                                 _mm256_slli_epi32(w1, 11))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 24);
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 14),
                                                 _mm256_slli_epi32(w0, 18))),
          broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 7), broadcomp));
}

/* we packed 256 26-bit values, touching 26 256-bit words, using 416 bytes */
static void filtergt26(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  26 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(67108863);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 26),
                                                 _mm256_slli_epi32(w1, 6))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 14),
                                                 _mm256_slli_epi32(w1, 18))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 8),
                                                 _mm256_slli_epi32(w0, 24))),
          broadcomp));
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 2)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 22),
                                                 _mm256_slli_epi32(w0, 10))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 10),
                                                 _mm256_slli_epi32(w0, 22))),
          broadcomp));
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                 _mm256_slli_epi32(w1, 2))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 18),
                                                 _mm256_slli_epi32(w1, 14))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 12),
                                                 _mm256_slli_epi32(w0, 20))),
          broadcomp));
  _mm256_storeu_si256(out + 15,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 6), broadcomp));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 16, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w1), broadcomp));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                 _mm256_slli_epi32(w0, 6))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                 _mm256_slli_epi32(w1, 12))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 14),
                                                 _mm256_slli_epi32(w0, 18))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 8),
                                                 _mm256_slli_epi32(w1, 24))),
          broadcomp));
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 2)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 22),
                                                 _mm256_slli_epi32(w1, 10))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 21);
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 10),
                                                 _mm256_slli_epi32(w1, 22))),
          broadcomp));
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 4)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 22);
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                 _mm256_slli_epi32(w0, 2))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 23);
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 24);
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 18),
                                                 _mm256_slli_epi32(w0, 14))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 25);
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 12),
                                                 _mm256_slli_epi32(w1, 20))),
          broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w1, 6), broadcomp));
}

/* we packed 256 27-bit values, touching 27 256-bit words, using 432 bytes */
static void filtergt27(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  27 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(134217727);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 27),
                                                 _mm256_slli_epi32(w1, 5))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 22),
                                                 _mm256_slli_epi32(w0, 10))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 17),
                                                 _mm256_slli_epi32(w1, 15))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 12),
                                                 _mm256_slli_epi32(w0, 20))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 7),
                                                 _mm256_slli_epi32(w1, 25))),
          broadcomp));
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 2)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 29),
                                                 _mm256_slli_epi32(w0, 3))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 19),
                                                 _mm256_slli_epi32(w0, 13))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 14),
                                                 _mm256_slli_epi32(w1, 18))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 9),
                                                 _mm256_slli_epi32(w0, 23))),
          broadcomp));
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 4)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 31),
                                                 _mm256_slli_epi32(w1, 1))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                 _mm256_slli_epi32(w0, 6))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 21),
                                                 _mm256_slli_epi32(w1, 11))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 11),
                                                 _mm256_slli_epi32(w1, 21))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 6),
                                                 _mm256_slli_epi32(w0, 26))),
          broadcomp));
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w0, 1)),
                         broadcomp));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 23),
                                                 _mm256_slli_epi32(w0, 9))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 18),
                                                 _mm256_slli_epi32(w1, 14))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 13),
                                                 _mm256_slli_epi32(w0, 19))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 21);
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 8),
                                                 _mm256_slli_epi32(w1, 24))),
          broadcomp));
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 3)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 22);
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                 _mm256_slli_epi32(w0, 2))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 23);
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 25),
                                                 _mm256_slli_epi32(w1, 7))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 24);
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 25);
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 15),
                                                 _mm256_slli_epi32(w1, 17))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 26);
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 10),
                                                 _mm256_slli_epi32(w0, 22))),
          broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 5), broadcomp));
}

/* we packed 256 28-bit values, touching 28 256-bit words, using 448 bytes */
static void filtergt28(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  28 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(268435455);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                 _mm256_slli_epi32(w1, 12))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 12),
                                                 _mm256_slli_epi32(w1, 20))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 8),
                                                 _mm256_slli_epi32(w0, 24))),
          broadcomp));
  _mm256_storeu_si256(out + 7,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 4), broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 8, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w1), broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 12),
                                                 _mm256_slli_epi32(w0, 20))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 8),
                                                 _mm256_slli_epi32(w1, 24))),
          broadcomp));
  _mm256_storeu_si256(out + 15,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w1, 4), broadcomp));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 16, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                 _mm256_slli_epi32(w1, 12))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 12),
                                                 _mm256_slli_epi32(w1, 20))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 8),
                                                 _mm256_slli_epi32(w0, 24))),
          broadcomp));
  _mm256_storeu_si256(out + 23,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 4), broadcomp));
  w1 = _mm256_lddqu_si256(in + 21);
  _mm256_storeu_si256(
      out + 24, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w1), broadcomp));
  w0 = _mm256_lddqu_si256(in + 22);
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 23);
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 24);
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 25);
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 26);
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 12),
                                                 _mm256_slli_epi32(w0, 20))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 27);
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 8),
                                                 _mm256_slli_epi32(w1, 24))),
          broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w1, 4), broadcomp));
}

/* we packed 256 29-bit values, touching 29 256-bit words, using 464 bytes */
static void filtergt29(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  29 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(536870911);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 29),
                                                 _mm256_slli_epi32(w1, 3))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                 _mm256_slli_epi32(w0, 6))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 23),
                                                 _mm256_slli_epi32(w1, 9))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 17),
                                                 _mm256_slli_epi32(w1, 15))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 14),
                                                 _mm256_slli_epi32(w0, 18))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 11),
                                                 _mm256_slli_epi32(w1, 21))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 8),
                                                 _mm256_slli_epi32(w0, 24))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 5),
                                                 _mm256_slli_epi32(w1, 27))),
          broadcomp));
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 2)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 31),
                                                 _mm256_slli_epi32(w0, 1))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 25),
                                                 _mm256_slli_epi32(w0, 7))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 22),
                                                 _mm256_slli_epi32(w1, 10))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 19),
                                                 _mm256_slli_epi32(w0, 13))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 13),
                                                 _mm256_slli_epi32(w0, 19))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 10),
                                                 _mm256_slli_epi32(w1, 22))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 7),
                                                 _mm256_slli_epi32(w0, 25))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 4),
                                                 _mm256_slli_epi32(w1, 28))),
          broadcomp));
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(_mm256_and_si256(mask, _mm256_srli_epi32(w1, 1)),
                         broadcomp));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                 _mm256_slli_epi32(w0, 2))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 21);
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 27),
                                                 _mm256_slli_epi32(w1, 5))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 22);
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 23);
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 21),
                                                 _mm256_slli_epi32(w1, 11))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 24);
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 18),
                                                 _mm256_slli_epi32(w0, 14))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 25);
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 15),
                                                 _mm256_slli_epi32(w1, 17))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 26);
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 12),
                                                 _mm256_slli_epi32(w0, 20))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 27);
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 9),
                                                 _mm256_slli_epi32(w1, 23))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 28);
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 6),
                                                 _mm256_slli_epi32(w0, 26))),
          broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 3), broadcomp));
}

/* we packed 256 30-bit values, touching 30 256-bit words, using 480 bytes */
static void filtergt30(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  30 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(1073741823);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                 _mm256_slli_epi32(w1, 2))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 26),
                                                 _mm256_slli_epi32(w1, 6))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 22),
                                                 _mm256_slli_epi32(w1, 10))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 18),
                                                 _mm256_slli_epi32(w1, 14))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 14),
                                                 _mm256_slli_epi32(w1, 18))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 12),
                                                 _mm256_slli_epi32(w0, 20))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 10),
                                                 _mm256_slli_epi32(w1, 22))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 8),
                                                 _mm256_slli_epi32(w0, 24))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 6),
                                                 _mm256_slli_epi32(w1, 26))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 4),
                                                 _mm256_slli_epi32(w0, 28))),
          broadcomp));
  _mm256_storeu_si256(out + 15,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 2), broadcomp));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 16, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w1), broadcomp));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                 _mm256_slli_epi32(w0, 2))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                 _mm256_slli_epi32(w0, 6))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 22),
                                                 _mm256_slli_epi32(w0, 10))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 21);
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                 _mm256_slli_epi32(w1, 12))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 22);
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 18),
                                                 _mm256_slli_epi32(w0, 14))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 23);
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 24);
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 14),
                                                 _mm256_slli_epi32(w0, 18))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 25);
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 12),
                                                 _mm256_slli_epi32(w1, 20))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 26);
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 10),
                                                 _mm256_slli_epi32(w0, 22))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 27);
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 8),
                                                 _mm256_slli_epi32(w1, 24))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 28);
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 6),
                                                 _mm256_slli_epi32(w0, 26))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 29);
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 4),
                                                 _mm256_slli_epi32(w1, 28))),
          broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w1, 2), broadcomp));
}

/* we packed 256 31-bit values, touching 31 256-bit words, using 496 bytes */
static void filtergt31(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  31 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(2147483647);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(_mm256_and_si256(mask, w0), broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 31),
                                                 _mm256_slli_epi32(w1, 1))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                 _mm256_slli_epi32(w0, 2))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 29),
                                                 _mm256_slli_epi32(w1, 3))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 27),
                                                 _mm256_slli_epi32(w1, 5))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                 _mm256_slli_epi32(w0, 6))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 25),
                                                 _mm256_slli_epi32(w1, 7))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 23),
                                                 _mm256_slli_epi32(w1, 9))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 22),
                                                 _mm256_slli_epi32(w0, 10))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 21),
                                                 _mm256_slli_epi32(w1, 11))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 19),
                                                 _mm256_slli_epi32(w1, 13))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 18),
                                                 _mm256_slli_epi32(w0, 14))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 17),
                                                 _mm256_slli_epi32(w1, 15))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 15),
                                                 _mm256_slli_epi32(w1, 17))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 14),
                                                 _mm256_slli_epi32(w0, 18))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 13),
                                                 _mm256_slli_epi32(w1, 19))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 12),
                                                 _mm256_slli_epi32(w0, 20))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 21);
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 11),
                                                 _mm256_slli_epi32(w1, 21))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 22);
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 10),
                                                 _mm256_slli_epi32(w0, 22))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 23);
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 9),
                                                 _mm256_slli_epi32(w1, 23))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 24);
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 8),
                                                 _mm256_slli_epi32(w0, 24))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 25);
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 7),
                                                 _mm256_slli_epi32(w1, 25))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 26);
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 6),
                                                 _mm256_slli_epi32(w0, 26))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 27);
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 5),
                                                 _mm256_slli_epi32(w1, 27))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 28);
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 4),
                                                 _mm256_slli_epi32(w0, 28))),
          broadcomp));
  w1 = _mm256_lddqu_si256(in + 29);
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 3),
                                                 _mm256_slli_epi32(w1, 29))),
          broadcomp));
  w0 = _mm256_lddqu_si256(in + 30);
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 2),
                                                 _mm256_slli_epi32(w0, 30))),
          broadcomp));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(_mm256_srli_epi32(w0, 1), broadcomp));
}

/* we packed 256 32-bit values, touching 32 256-bit words, using 512 bytes */
static void filtergt32(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  32 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(out + 0, _mm256_cmpgt_epi32(w0, broadcomp));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(out + 1, _mm256_cmpgt_epi32(w1, broadcomp));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(out + 2, _mm256_cmpgt_epi32(w0, broadcomp));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(out + 3, _mm256_cmpgt_epi32(w1, broadcomp));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(out + 4, _mm256_cmpgt_epi32(w0, broadcomp));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(out + 5, _mm256_cmpgt_epi32(w1, broadcomp));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(out + 6, _mm256_cmpgt_epi32(w0, broadcomp));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(out + 7, _mm256_cmpgt_epi32(w1, broadcomp));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(out + 8, _mm256_cmpgt_epi32(w0, broadcomp));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(out + 9, _mm256_cmpgt_epi32(w1, broadcomp));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(out + 10, _mm256_cmpgt_epi32(w0, broadcomp));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(out + 11, _mm256_cmpgt_epi32(w1, broadcomp));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(out + 12, _mm256_cmpgt_epi32(w0, broadcomp));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(out + 13, _mm256_cmpgt_epi32(w1, broadcomp));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(out + 14, _mm256_cmpgt_epi32(w0, broadcomp));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(out + 15, _mm256_cmpgt_epi32(w1, broadcomp));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(out + 16, _mm256_cmpgt_epi32(w0, broadcomp));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(out + 17, _mm256_cmpgt_epi32(w1, broadcomp));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(out + 18, _mm256_cmpgt_epi32(w0, broadcomp));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(out + 19, _mm256_cmpgt_epi32(w1, broadcomp));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(out + 20, _mm256_cmpgt_epi32(w0, broadcomp));
  w1 = _mm256_lddqu_si256(in + 21);
  _mm256_storeu_si256(out + 21, _mm256_cmpgt_epi32(w1, broadcomp));
  w0 = _mm256_lddqu_si256(in + 22);
  _mm256_storeu_si256(out + 22, _mm256_cmpgt_epi32(w0, broadcomp));
  w1 = _mm256_lddqu_si256(in + 23);
  _mm256_storeu_si256(out + 23, _mm256_cmpgt_epi32(w1, broadcomp));
  w0 = _mm256_lddqu_si256(in + 24);
  _mm256_storeu_si256(out + 24, _mm256_cmpgt_epi32(w0, broadcomp));
  w1 = _mm256_lddqu_si256(in + 25);
  _mm256_storeu_si256(out + 25, _mm256_cmpgt_epi32(w1, broadcomp));
  w0 = _mm256_lddqu_si256(in + 26);
  _mm256_storeu_si256(out + 26, _mm256_cmpgt_epi32(w0, broadcomp));
  w1 = _mm256_lddqu_si256(in + 27);
  _mm256_storeu_si256(out + 27, _mm256_cmpgt_epi32(w1, broadcomp));
  w0 = _mm256_lddqu_si256(in + 28);
  _mm256_storeu_si256(out + 28, _mm256_cmpgt_epi32(w0, broadcomp));
  w1 = _mm256_lddqu_si256(in + 29);
  _mm256_storeu_si256(out + 29, _mm256_cmpgt_epi32(w1, broadcomp));
  w0 = _mm256_lddqu_si256(in + 30);
  _mm256_storeu_si256(out + 30, _mm256_cmpgt_epi32(w0, broadcomp));
  w1 = _mm256_lddqu_si256(in + 31);
  _mm256_storeu_si256(out + 31, _mm256_cmpgt_epi32(w1, broadcomp));
}

static void filterlt0(const __m256i *in, u32 *matches, const INTEGER comp) {
  if (comp > 0)
    memset(matches, 1, 256 * sizeof(*matches));
  else
    memset(matches, 0, 256 * sizeof(*matches));
}

/* we packed 256 1-bit values, touching 1 256-bit words, using 16 bytes */
static void filterlt1(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  1 256-bit word */
  __m256i w0;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(1);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 1))));
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 2))));
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 3))));
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 4))));
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 5))));
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 6))));
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 7))));
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 8))));
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 9))));
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 10))));
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 11))));
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 12))));
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 13))));
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 14))));
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 15))));
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 16))));
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 17))));
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 18))));
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 19))));
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 20))));
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 21))));
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 22))));
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 23))));
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 24))));
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 25))));
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 26))));
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 27))));
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 28))));
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 29))));
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 30))));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 31)));
}

/* we packed 256 2-bit values, touching 2 256-bit words, using 32 bytes */
static void filterlt2(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  2 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(3);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 2))));
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 4))));
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 6))));
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 8))));
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 10))));
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 12))));
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 14))));
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 16))));
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 18))));
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 20))));
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 22))));
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 24))));
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 26))));
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 28))));
  _mm256_storeu_si256(out + 15,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 30)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 16, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w1)));
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 2))));
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 4))));
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 6))));
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 8))));
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 10))));
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 12))));
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 14))));
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 16))));
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 18))));
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 20))));
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 22))));
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 24))));
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 26))));
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 28))));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w1, 30)));
}

/* we packed 256 3-bit values, touching 3 256-bit words, using 48 bytes */
static void filterlt3(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  3 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(7);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 3))));
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 6))));
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 9))));
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 12))));
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 15))));
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 18))));
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 21))));
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 24))));
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 27))));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                 _mm256_slli_epi32(w1, 2)))));
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 1))));
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 4))));
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 7))));
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 10))));
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 13))));
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 16))));
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 19))));
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 22))));
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 25))));
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 28))));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 31),
                                                 _mm256_slli_epi32(w0, 1)))));
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 2))));
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 5))));
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 8))));
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 11))));
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 14))));
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 17))));
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 20))));
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 23))));
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 26))));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 29)));
}

/* we packed 256 4-bit values, touching 4 256-bit words, using 64 bytes */
static void filterlt4(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  4 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(15);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 4))));
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 8))));
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 12))));
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 16))));
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 20))));
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 24))));
  _mm256_storeu_si256(out + 7,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 28)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 8, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w1)));
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 4))));
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 8))));
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 12))));
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 16))));
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 20))));
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 24))));
  _mm256_storeu_si256(out + 15,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w1, 28)));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 16, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 4))));
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 8))));
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 12))));
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 16))));
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 20))));
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 24))));
  _mm256_storeu_si256(out + 23,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 28)));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 24, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w1)));
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 4))));
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 8))));
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 12))));
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 16))));
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 20))));
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 24))));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w1, 28)));
}

/* we packed 256 5-bit values, touching 5 256-bit words, using 80 bytes */
static void filterlt5(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  5 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(31);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 5))));
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 10))));
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 15))));
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 20))));
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 25))));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                 _mm256_slli_epi32(w1, 2)))));
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 3))));
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 8))));
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 13))));
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 18))));
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 23))));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4)))));
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 1))));
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 6))));
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 11))));
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 16))));
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 21))));
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 26))));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 31),
                                                 _mm256_slli_epi32(w1, 1)))));
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 4))));
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 9))));
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 14))));
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 19))));
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 24))));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 29),
                                                 _mm256_slli_epi32(w0, 3)))));
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 2))));
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 7))));
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 12))));
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 17))));
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 22))));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 27)));
}

/* we packed 256 6-bit values, touching 6 256-bit words, using 96 bytes */
static void filterlt6(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  6 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(63);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 6))));
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 12))));
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 18))));
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 24))));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                 _mm256_slli_epi32(w1, 2)))));
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 4))));
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 10))));
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 16))));
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 22))));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4)))));
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 2))));
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 8))));
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 14))));
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 20))));
  _mm256_storeu_si256(out + 15,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 26)));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 16, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w1)));
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 6))));
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 12))));
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 18))));
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 24))));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                 _mm256_slli_epi32(w0, 2)))));
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 4))));
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 10))));
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 16))));
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 22))));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4)))));
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 2))));
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 8))));
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 14))));
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 20))));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w1, 26)));
}

/* we packed 256 7-bit values, touching 7 256-bit words, using 112 bytes */
static void filterlt7(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  7 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(127);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 7))));
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 14))));
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 21))));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4)))));
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 3))));
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 10))));
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 17))));
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 24))));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 31),
                                                 _mm256_slli_epi32(w0, 1)))));
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 6))));
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 13))));
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 20))));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 27),
                                                 _mm256_slli_epi32(w1, 5)))));
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 2))));
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 9))));
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 16))));
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 23))));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                 _mm256_slli_epi32(w0, 2)))));
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 5))));
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 12))));
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 19))));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 26),
                                                 _mm256_slli_epi32(w1, 6)))));
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 1))));
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 8))));
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 15))));
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 22))));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 29),
                                                 _mm256_slli_epi32(w0, 3)))));
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 4))));
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 11))));
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 18))));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 25)));
}

/* we packed 256 8-bit values, touching 8 256-bit words, using 128 bytes */
static void filterlt8(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  8 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(255);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 8))));
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 16))));
  _mm256_storeu_si256(out + 3,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 24)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 4, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w1)));
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 8))));
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 16))));
  _mm256_storeu_si256(out + 7,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w1, 24)));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 8, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 8))));
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 16))));
  _mm256_storeu_si256(out + 11,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 24)));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 12, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w1)));
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 8))));
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 16))));
  _mm256_storeu_si256(out + 15,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w1, 24)));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 16, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 8))));
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 16))));
  _mm256_storeu_si256(out + 19,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 24)));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 20, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w1)));
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 8))));
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 16))));
  _mm256_storeu_si256(out + 23,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w1, 24)));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 24, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 8))));
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 16))));
  _mm256_storeu_si256(out + 27,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 24)));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 28, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w1)));
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 8))));
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 16))));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w1, 24)));
}

/* we packed 256 9-bit values, touching 9 256-bit words, using 144 bytes */
static void filterlt9(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  9 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(511);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 9))));
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 18))));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 27),
                                                 _mm256_slli_epi32(w1, 5)))));
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 4))));
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 13))));
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 22))));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 31),
                                                 _mm256_slli_epi32(w0, 1)))));
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 8))));
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 17))));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 26),
                                                 _mm256_slli_epi32(w1, 6)))));
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 3))));
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 12))));
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 21))));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                 _mm256_slli_epi32(w0, 2)))));
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 7))));
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 16))));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 25),
                                                 _mm256_slli_epi32(w1, 7)))));
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 2))));
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 11))));
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 20))));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 29),
                                                 _mm256_slli_epi32(w0, 3)))));
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 6))));
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 15))));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8)))));
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 1))));
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 10))));
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 19))));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4)))));
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 5))));
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 14))));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 23)));
}

/* we packed 256 10-bit values, touching 10 256-bit words, using 160 bytes */
static void filterlt10(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  10 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(1023);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 10))));
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 20))));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                 _mm256_slli_epi32(w1, 2)))));
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 8))));
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 18))));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4)))));
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 6))));
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 16))));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 26),
                                                 _mm256_slli_epi32(w1, 6)))));
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 4))));
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 14))));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8)))));
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 2))));
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 12))));
  _mm256_storeu_si256(out + 15,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 22)));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 16, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w1)));
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 10))));
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 20))));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                 _mm256_slli_epi32(w0, 2)))));
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 8))));
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 18))));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4)))));
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 6))));
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 16))));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                 _mm256_slli_epi32(w0, 6)))));
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 4))));
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 14))));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8)))));
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 2))));
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 12))));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w1, 22)));
}

/* we packed 256 11-bit values, touching 11 256-bit words, using 176 bytes */
static void filterlt11(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  11 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(2047);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 11))));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 22),
                                                 _mm256_slli_epi32(w1, 10)))));
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 1))));
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 12))));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 23),
                                                 _mm256_slli_epi32(w0, 9)))));
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 2))));
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 13))));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8)))));
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 3))));
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 14))));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 25),
                                                 _mm256_slli_epi32(w0, 7)))));
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 4))));
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 15))));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 26),
                                                 _mm256_slli_epi32(w1, 6)))));
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 5))));
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 16))));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 27),
                                                 _mm256_slli_epi32(w0, 5)))));
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 6))));
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 17))));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4)))));
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 7))));
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 18))));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 29),
                                                 _mm256_slli_epi32(w0, 3)))));
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 8))));
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 19))));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                 _mm256_slli_epi32(w1, 2)))));
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 9))));
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 20))));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 31),
                                                 _mm256_slli_epi32(w0, 1)))));
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 10))));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 21)));
}

/* we packed 256 12-bit values, touching 12 256-bit words, using 192 bytes */
static void filterlt12(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  12 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(4095);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 12))));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8)))));
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 4))));
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 16))));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4)))));
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 8))));
  _mm256_storeu_si256(out + 7,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 20)));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 8, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w1)));
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 12))));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8)))));
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 4))));
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 16))));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4)))));
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 8))));
  _mm256_storeu_si256(out + 15,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w1, 20)));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 16, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 12))));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8)))));
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 4))));
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 16))));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4)))));
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 8))));
  _mm256_storeu_si256(out + 23,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 20)));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 24, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w1)));
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 12))));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8)))));
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 4))));
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 16))));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4)))));
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 8))));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w1, 20)));
}

/* we packed 256 13-bit values, touching 13 256-bit words, using 208 bytes */
static void filterlt13(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  13 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(8191);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 13))));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 26),
                                                 _mm256_slli_epi32(w1, 6)))));
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 7))));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12)))));
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 1))));
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 14))));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 27),
                                                 _mm256_slli_epi32(w1, 5)))));
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 8))));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 21),
                                                 _mm256_slli_epi32(w0, 11)))));
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 2))));
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 15))));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4)))));
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 9))));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 22),
                                                 _mm256_slli_epi32(w0, 10)))));
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 3))));
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 16))));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 29),
                                                 _mm256_slli_epi32(w1, 3)))));
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 10))));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 23),
                                                 _mm256_slli_epi32(w0, 9)))));
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 4))));
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 17))));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                 _mm256_slli_epi32(w1, 2)))));
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 11))));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8)))));
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 5))));
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 18))));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 31),
                                                 _mm256_slli_epi32(w1, 1)))));
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 12))));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 25),
                                                 _mm256_slli_epi32(w0, 7)))));
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 6))));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 19)));
}

/* we packed 256 14-bit values, touching 14 256-bit words, using 224 bytes */
static void filterlt14(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  14 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(16383);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 14))));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4)))));
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 10))));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8)))));
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 6))));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                 _mm256_slli_epi32(w1, 12)))));
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 2))));
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 16))));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                 _mm256_slli_epi32(w0, 2)))));
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 12))));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 26),
                                                 _mm256_slli_epi32(w1, 6)))));
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 8))));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 22),
                                                 _mm256_slli_epi32(w0, 10)))));
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 4))));
  _mm256_storeu_si256(out + 15,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 18)));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 16, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w1)));
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 14))));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4)))));
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 10))));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8)))));
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 6))));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12)))));
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 2))));
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 16))));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                 _mm256_slli_epi32(w1, 2)))));
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 12))));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                 _mm256_slli_epi32(w0, 6)))));
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 8))));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 22),
                                                 _mm256_slli_epi32(w1, 10)))));
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 4))));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w1, 18)));
}

/* we packed 256 15-bit values, touching 15 256-bit words, using 240 bytes */
static void filterlt15(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  15 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(32767);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 15))));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                 _mm256_slli_epi32(w1, 2)))));
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 13))));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4)))));
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 11))));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 26),
                                                 _mm256_slli_epi32(w1, 6)))));
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 9))));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8)))));
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 7))));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 22),
                                                 _mm256_slli_epi32(w1, 10)))));
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 5))));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12)))));
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 3))));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 18),
                                                 _mm256_slli_epi32(w1, 14)))));
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 1))));
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 16))));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 31),
                                                 _mm256_slli_epi32(w0, 1)))));
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 14))));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 29),
                                                 _mm256_slli_epi32(w1, 3)))));
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 12))));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 27),
                                                 _mm256_slli_epi32(w0, 5)))));
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 10))));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 25),
                                                 _mm256_slli_epi32(w1, 7)))));
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 8))));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 23),
                                                 _mm256_slli_epi32(w0, 9)))));
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 6))));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 21),
                                                 _mm256_slli_epi32(w1, 11)))));
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 4))));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 19),
                                                 _mm256_slli_epi32(w0, 13)))));
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 2))));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 17)));
}

/* we packed 256 16-bit values, touching 16 256-bit words, using 256 bytes */
static void filterlt16(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  16 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(65535);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  _mm256_storeu_si256(out + 1,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 16)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 2, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w1)));
  _mm256_storeu_si256(out + 3,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w1, 16)));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 4, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  _mm256_storeu_si256(out + 5,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 16)));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 6, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w1)));
  _mm256_storeu_si256(out + 7,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w1, 16)));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 8, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  _mm256_storeu_si256(out + 9,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 16)));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 10, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w1)));
  _mm256_storeu_si256(out + 11,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w1, 16)));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 12, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  _mm256_storeu_si256(out + 13,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 16)));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 14, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w1)));
  _mm256_storeu_si256(out + 15,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w1, 16)));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 16, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  _mm256_storeu_si256(out + 17,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 16)));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 18, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w1)));
  _mm256_storeu_si256(out + 19,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w1, 16)));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 20, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  _mm256_storeu_si256(out + 21,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 16)));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 22, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w1)));
  _mm256_storeu_si256(out + 23,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w1, 16)));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 24, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  _mm256_storeu_si256(out + 25,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 16)));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 26, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w1)));
  _mm256_storeu_si256(out + 27,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w1, 16)));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 28, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  _mm256_storeu_si256(out + 29,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 16)));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 30, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w1)));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w1, 16)));
}

/* we packed 256 17-bit values, touching 17 256-bit words, using 272 bytes */
static void filterlt17(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  17 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(131071);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 17),
                                                 _mm256_slli_epi32(w1, 15)))));
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 2))));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 19),
                                                 _mm256_slli_epi32(w0, 13)))));
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 4))));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 21),
                                                 _mm256_slli_epi32(w1, 11)))));
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 6))));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 23),
                                                 _mm256_slli_epi32(w0, 9)))));
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 8))));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 25),
                                                 _mm256_slli_epi32(w1, 7)))));
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 10))));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 27),
                                                 _mm256_slli_epi32(w0, 5)))));
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 12))));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 29),
                                                 _mm256_slli_epi32(w1, 3)))));
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 14))));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 31),
                                                 _mm256_slli_epi32(w0, 1)))));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16)))));
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 1))));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 18),
                                                 _mm256_slli_epi32(w0, 14)))));
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 3))));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                 _mm256_slli_epi32(w1, 12)))));
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 5))));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 22),
                                                 _mm256_slli_epi32(w0, 10)))));
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 7))));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8)))));
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 9))));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                 _mm256_slli_epi32(w0, 6)))));
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 11))));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4)))));
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 13))));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                 _mm256_slli_epi32(w0, 2)))));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 15)));
}

/* we packed 256 18-bit values, touching 18 256-bit words, using 288 bytes */
static void filterlt18(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  18 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(262143);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 18),
                                                 _mm256_slli_epi32(w1, 14)))));
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 4))));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 22),
                                                 _mm256_slli_epi32(w0, 10)))));
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 8))));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 26),
                                                 _mm256_slli_epi32(w1, 6)))));
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 12))));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                 _mm256_slli_epi32(w0, 2)))));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16)))));
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 2))));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12)))));
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 6))));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8)))));
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 10))));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4)))));
  _mm256_storeu_si256(out + 15,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 14)));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 16, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w1)));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 18),
                                                 _mm256_slli_epi32(w0, 14)))));
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 4))));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 22),
                                                 _mm256_slli_epi32(w1, 10)))));
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 8))));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                 _mm256_slli_epi32(w0, 6)))));
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 12))));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                 _mm256_slli_epi32(w1, 2)))));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16)))));
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 2))));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                 _mm256_slli_epi32(w1, 12)))));
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 6))));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8)))));
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 10))));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4)))));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w1, 14)));
}

/* we packed 256 19-bit values, touching 19 256-bit words, using 304 bytes */
static void filterlt19(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  19 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(524287);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 19),
                                                 _mm256_slli_epi32(w1, 13)))));
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 6))));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 25),
                                                 _mm256_slli_epi32(w0, 7)))));
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 12))));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 31),
                                                 _mm256_slli_epi32(w1, 1)))));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 18),
                                                 _mm256_slli_epi32(w0, 14)))));
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 5))));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8)))));
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 11))));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                 _mm256_slli_epi32(w0, 2)))));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 17),
                                                 _mm256_slli_epi32(w1, 15)))));
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 4))));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 23),
                                                 _mm256_slli_epi32(w0, 9)))));
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 10))));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 29),
                                                 _mm256_slli_epi32(w1, 3)))));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16)))));
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 3))));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 22),
                                                 _mm256_slli_epi32(w1, 10)))));
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 9))));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4)))));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 15),
                                                 _mm256_slli_epi32(w1, 17)))));
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 2))));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 21),
                                                 _mm256_slli_epi32(w0, 11)))));
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 8))));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 27),
                                                 _mm256_slli_epi32(w1, 5)))));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 14),
                                                 _mm256_slli_epi32(w0, 18)))));
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 1))));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                 _mm256_slli_epi32(w1, 12)))));
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 7))));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                 _mm256_slli_epi32(w0, 6)))));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 13)));
}

/* we packed 256 20-bit values, touching 20 256-bit words, using 320 bytes */
static void filterlt20(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  20 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(1048575);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                 _mm256_slli_epi32(w1, 12)))));
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 8))));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4)))));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16)))));
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 4))));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8)))));
  _mm256_storeu_si256(out + 7,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 12)));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 8, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w1)));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12)))));
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 8))));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4)))));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16)))));
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 4))));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8)))));
  _mm256_storeu_si256(out + 15,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w1, 12)));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 16, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                 _mm256_slli_epi32(w1, 12)))));
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 8))));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4)))));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16)))));
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 4))));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8)))));
  _mm256_storeu_si256(out + 23,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 12)));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 24, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w1)));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12)))));
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 8))));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4)))));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16)))));
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 4))));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8)))));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w1, 12)));
}

/* we packed 256 21-bit values, touching 21 256-bit words, using 336 bytes */
static void filterlt21(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  21 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(2097151);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 21),
                                                 _mm256_slli_epi32(w1, 11)))));
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 10))));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 31),
                                                 _mm256_slli_epi32(w0, 1)))));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                 _mm256_slli_epi32(w1, 12)))));
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 9))));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                 _mm256_slli_epi32(w0, 2)))));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 19),
                                                 _mm256_slli_epi32(w1, 13)))));
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 8))));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 29),
                                                 _mm256_slli_epi32(w0, 3)))));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 18),
                                                 _mm256_slli_epi32(w1, 14)))));
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 7))));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4)))));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 17),
                                                 _mm256_slli_epi32(w1, 15)))));
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 6))));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 27),
                                                 _mm256_slli_epi32(w0, 5)))));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16)))));
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 5))));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                 _mm256_slli_epi32(w0, 6)))));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 15),
                                                 _mm256_slli_epi32(w1, 17)))));
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 4))));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 25),
                                                 _mm256_slli_epi32(w0, 7)))));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 14),
                                                 _mm256_slli_epi32(w1, 18)))));
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 3))));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8)))));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 13),
                                                 _mm256_slli_epi32(w1, 19)))));
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 2))));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 23),
                                                 _mm256_slli_epi32(w0, 9)))));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 12),
                                                 _mm256_slli_epi32(w1, 20)))));
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 1))));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 22),
                                                 _mm256_slli_epi32(w0, 10)))));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 11)));
}

/* we packed 256 22-bit values, touching 22 256-bit words, using 352 bytes */
static void filterlt22(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  22 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(4194303);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 22),
                                                 _mm256_slli_epi32(w1, 10)))));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 12),
                                                 _mm256_slli_epi32(w0, 20)))));
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 2))));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8)))));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 14),
                                                 _mm256_slli_epi32(w0, 18)))));
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 4))));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 26),
                                                 _mm256_slli_epi32(w1, 6)))));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16)))));
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 6))));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4)))));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 18),
                                                 _mm256_slli_epi32(w0, 14)))));
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 8))));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                 _mm256_slli_epi32(w1, 2)))));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12)))));
  _mm256_storeu_si256(out + 15,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 10)));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 16, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w1)));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 22),
                                                 _mm256_slli_epi32(w0, 10)))));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 12),
                                                 _mm256_slli_epi32(w1, 20)))));
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 2))));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8)))));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 14),
                                                 _mm256_slli_epi32(w1, 18)))));
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 4))));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                 _mm256_slli_epi32(w0, 6)))));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16)))));
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 6))));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4)))));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 18),
                                                 _mm256_slli_epi32(w1, 14)))));
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 8))));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                 _mm256_slli_epi32(w0, 2)))));
  w1 = _mm256_lddqu_si256(in + 21);
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                 _mm256_slli_epi32(w1, 12)))));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w1, 10)));
}

/* we packed 256 23-bit values, touching 23 256-bit words, using 368 bytes */
static void filterlt23(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  23 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(8388607);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 23),
                                                 _mm256_slli_epi32(w1, 9)))));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 14),
                                                 _mm256_slli_epi32(w0, 18)))));
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 5))));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4)))));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 19),
                                                 _mm256_slli_epi32(w0, 13)))));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 10),
                                                 _mm256_slli_epi32(w1, 22)))));
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 1))));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8)))));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 15),
                                                 _mm256_slli_epi32(w1, 17)))));
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 6))));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 29),
                                                 _mm256_slli_epi32(w0, 3)))));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                 _mm256_slli_epi32(w1, 12)))));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 11),
                                                 _mm256_slli_epi32(w0, 21)))));
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 2))));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 25),
                                                 _mm256_slli_epi32(w1, 7)))));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16)))));
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 7))));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                 _mm256_slli_epi32(w1, 2)))));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 21),
                                                 _mm256_slli_epi32(w0, 11)))));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 12),
                                                 _mm256_slli_epi32(w1, 20)))));
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 3))));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                 _mm256_slli_epi32(w0, 6)))));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 17),
                                                 _mm256_slli_epi32(w1, 15)))));
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 8))));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 31),
                                                 _mm256_slli_epi32(w0, 1)))));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 22),
                                                 _mm256_slli_epi32(w1, 10)))));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 13),
                                                 _mm256_slli_epi32(w0, 19)))));
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 4))));
  w1 = _mm256_lddqu_si256(in + 21);
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 27),
                                                 _mm256_slli_epi32(w1, 5)))));
  w0 = _mm256_lddqu_si256(in + 22);
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 18),
                                                 _mm256_slli_epi32(w0, 14)))));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 9)));
}

/* we packed 256 24-bit values, touching 24 256-bit words, using 384 bytes */
static void filterlt24(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  24 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(16777215);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8)))));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16)))));
  _mm256_storeu_si256(out + 3,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 8)));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 4, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w1)));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8)))));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16)))));
  _mm256_storeu_si256(out + 7,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w1, 8)));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 8, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8)))));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16)))));
  _mm256_storeu_si256(out + 11,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 8)));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 12, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w1)));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8)))));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16)))));
  _mm256_storeu_si256(out + 15,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w1, 8)));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 16, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8)))));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16)))));
  _mm256_storeu_si256(out + 19,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 8)));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 20, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w1)));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8)))));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16)))));
  _mm256_storeu_si256(out + 23,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w1, 8)));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 24, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8)))));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16)))));
  _mm256_storeu_si256(out + 27,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 8)));
  w1 = _mm256_lddqu_si256(in + 21);
  _mm256_storeu_si256(
      out + 28, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w1)));
  w0 = _mm256_lddqu_si256(in + 22);
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8)))));
  w1 = _mm256_lddqu_si256(in + 23);
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16)))));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w1, 8)));
}

/* we packed 256 25-bit values, touching 25 256-bit words, using 400 bytes */
static void filterlt25(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  25 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(33554431);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 25),
                                                 _mm256_slli_epi32(w1, 7)))));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 18),
                                                 _mm256_slli_epi32(w0, 14)))));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 11),
                                                 _mm256_slli_epi32(w1, 21)))));
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 4))));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 29),
                                                 _mm256_slli_epi32(w0, 3)))));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 22),
                                                 _mm256_slli_epi32(w1, 10)))));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 15),
                                                 _mm256_slli_epi32(w0, 17)))));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 8),
                                                 _mm256_slli_epi32(w1, 24)))));
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 1))));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                 _mm256_slli_epi32(w0, 6)))));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 19),
                                                 _mm256_slli_epi32(w1, 13)))));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 12),
                                                 _mm256_slli_epi32(w0, 20)))));
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 5))));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                 _mm256_slli_epi32(w1, 2)))));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 23),
                                                 _mm256_slli_epi32(w0, 9)))));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16)))));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 9),
                                                 _mm256_slli_epi32(w0, 23)))));
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 2))));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 27),
                                                 _mm256_slli_epi32(w1, 5)))));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12)))));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 13),
                                                 _mm256_slli_epi32(w1, 19)))));
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 6))));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 31),
                                                 _mm256_slli_epi32(w0, 1)))));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8)))));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 17),
                                                 _mm256_slli_epi32(w0, 15)))));
  w1 = _mm256_lddqu_si256(in + 21);
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 10),
                                                 _mm256_slli_epi32(w1, 22)))));
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 3))));
  w0 = _mm256_lddqu_si256(in + 22);
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4)))));
  w1 = _mm256_lddqu_si256(in + 23);
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 21),
                                                 _mm256_slli_epi32(w1, 11)))));
  w0 = _mm256_lddqu_si256(in + 24);
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 14),
                                                 _mm256_slli_epi32(w0, 18)))));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 7)));
}

/* we packed 256 26-bit values, touching 26 256-bit words, using 416 bytes */
static void filterlt26(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  26 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(67108863);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 26),
                                                 _mm256_slli_epi32(w1, 6)))));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12)))));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 14),
                                                 _mm256_slli_epi32(w1, 18)))));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 8),
                                                 _mm256_slli_epi32(w0, 24)))));
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 2))));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4)))));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 22),
                                                 _mm256_slli_epi32(w0, 10)))));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16)))));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 10),
                                                 _mm256_slli_epi32(w0, 22)))));
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 4))));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                 _mm256_slli_epi32(w1, 2)))));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8)))));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 18),
                                                 _mm256_slli_epi32(w1, 14)))));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 12),
                                                 _mm256_slli_epi32(w0, 20)))));
  _mm256_storeu_si256(out + 15,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 6)));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 16, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w1)));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                 _mm256_slli_epi32(w0, 6)))));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                 _mm256_slli_epi32(w1, 12)))));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 14),
                                                 _mm256_slli_epi32(w0, 18)))));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 8),
                                                 _mm256_slli_epi32(w1, 24)))));
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 2))));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4)))));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 22),
                                                 _mm256_slli_epi32(w1, 10)))));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16)))));
  w1 = _mm256_lddqu_si256(in + 21);
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 10),
                                                 _mm256_slli_epi32(w1, 22)))));
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 4))));
  w0 = _mm256_lddqu_si256(in + 22);
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                 _mm256_slli_epi32(w0, 2)))));
  w1 = _mm256_lddqu_si256(in + 23);
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8)))));
  w0 = _mm256_lddqu_si256(in + 24);
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 18),
                                                 _mm256_slli_epi32(w0, 14)))));
  w1 = _mm256_lddqu_si256(in + 25);
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 12),
                                                 _mm256_slli_epi32(w1, 20)))));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w1, 6)));
}

/* we packed 256 27-bit values, touching 27 256-bit words, using 432 bytes */
static void filterlt27(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  27 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(134217727);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 27),
                                                 _mm256_slli_epi32(w1, 5)))));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 22),
                                                 _mm256_slli_epi32(w0, 10)))));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 17),
                                                 _mm256_slli_epi32(w1, 15)))));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 12),
                                                 _mm256_slli_epi32(w0, 20)))));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 7),
                                                 _mm256_slli_epi32(w1, 25)))));
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 2))));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 29),
                                                 _mm256_slli_epi32(w0, 3)))));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8)))));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 19),
                                                 _mm256_slli_epi32(w0, 13)))));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 14),
                                                 _mm256_slli_epi32(w1, 18)))));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 9),
                                                 _mm256_slli_epi32(w0, 23)))));
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 4))));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 31),
                                                 _mm256_slli_epi32(w1, 1)))));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                 _mm256_slli_epi32(w0, 6)))));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 21),
                                                 _mm256_slli_epi32(w1, 11)))));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16)))));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 11),
                                                 _mm256_slli_epi32(w1, 21)))));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 6),
                                                 _mm256_slli_epi32(w0, 26)))));
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w0, 1))));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4)))));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 23),
                                                 _mm256_slli_epi32(w0, 9)))));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 18),
                                                 _mm256_slli_epi32(w1, 14)))));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 13),
                                                 _mm256_slli_epi32(w0, 19)))));
  w1 = _mm256_lddqu_si256(in + 21);
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 8),
                                                 _mm256_slli_epi32(w1, 24)))));
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 3))));
  w0 = _mm256_lddqu_si256(in + 22);
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                 _mm256_slli_epi32(w0, 2)))));
  w1 = _mm256_lddqu_si256(in + 23);
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 25),
                                                 _mm256_slli_epi32(w1, 7)))));
  w0 = _mm256_lddqu_si256(in + 24);
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12)))));
  w1 = _mm256_lddqu_si256(in + 25);
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 15),
                                                 _mm256_slli_epi32(w1, 17)))));
  w0 = _mm256_lddqu_si256(in + 26);
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 10),
                                                 _mm256_slli_epi32(w0, 22)))));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 5)));
}

/* we packed 256 28-bit values, touching 28 256-bit words, using 448 bytes */
static void filterlt28(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  28 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(268435455);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4)))));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8)))));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                 _mm256_slli_epi32(w1, 12)))));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16)))));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 12),
                                                 _mm256_slli_epi32(w1, 20)))));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 8),
                                                 _mm256_slli_epi32(w0, 24)))));
  _mm256_storeu_si256(out + 7,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 4)));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 8, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w1)));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4)))));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8)))));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12)))));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16)))));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 12),
                                                 _mm256_slli_epi32(w0, 20)))));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 8),
                                                 _mm256_slli_epi32(w1, 24)))));
  _mm256_storeu_si256(out + 15,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w1, 4)));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 16, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4)))));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8)))));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                 _mm256_slli_epi32(w1, 12)))));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16)))));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 12),
                                                 _mm256_slli_epi32(w1, 20)))));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 8),
                                                 _mm256_slli_epi32(w0, 24)))));
  _mm256_storeu_si256(out + 23,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 4)));
  w1 = _mm256_lddqu_si256(in + 21);
  _mm256_storeu_si256(
      out + 24, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w1)));
  w0 = _mm256_lddqu_si256(in + 22);
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4)))));
  w1 = _mm256_lddqu_si256(in + 23);
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8)))));
  w0 = _mm256_lddqu_si256(in + 24);
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12)))));
  w1 = _mm256_lddqu_si256(in + 25);
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16)))));
  w0 = _mm256_lddqu_si256(in + 26);
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 12),
                                                 _mm256_slli_epi32(w0, 20)))));
  w1 = _mm256_lddqu_si256(in + 27);
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 8),
                                                 _mm256_slli_epi32(w1, 24)))));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w1, 4)));
}

/* we packed 256 29-bit values, touching 29 256-bit words, using 464 bytes */
static void filterlt29(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  29 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(536870911);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 29),
                                                 _mm256_slli_epi32(w1, 3)))));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                 _mm256_slli_epi32(w0, 6)))));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 23),
                                                 _mm256_slli_epi32(w1, 9)))));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12)))));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 17),
                                                 _mm256_slli_epi32(w1, 15)))));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 14),
                                                 _mm256_slli_epi32(w0, 18)))));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 11),
                                                 _mm256_slli_epi32(w1, 21)))));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 8),
                                                 _mm256_slli_epi32(w0, 24)))));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 5),
                                                 _mm256_slli_epi32(w1, 27)))));
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 2))));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 31),
                                                 _mm256_slli_epi32(w0, 1)))));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4)))));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 25),
                                                 _mm256_slli_epi32(w0, 7)))));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 22),
                                                 _mm256_slli_epi32(w1, 10)))));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 19),
                                                 _mm256_slli_epi32(w0, 13)))));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16)))));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 13),
                                                 _mm256_slli_epi32(w0, 19)))));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 10),
                                                 _mm256_slli_epi32(w1, 22)))));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 7),
                                                 _mm256_slli_epi32(w0, 25)))));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 4),
                                                 _mm256_slli_epi32(w1, 28)))));
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(broadcomp,
                         _mm256_and_si256(mask, _mm256_srli_epi32(w1, 1))));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                 _mm256_slli_epi32(w0, 2)))));
  w1 = _mm256_lddqu_si256(in + 21);
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 27),
                                                 _mm256_slli_epi32(w1, 5)))));
  w0 = _mm256_lddqu_si256(in + 22);
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8)))));
  w1 = _mm256_lddqu_si256(in + 23);
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 21),
                                                 _mm256_slli_epi32(w1, 11)))));
  w0 = _mm256_lddqu_si256(in + 24);
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 18),
                                                 _mm256_slli_epi32(w0, 14)))));
  w1 = _mm256_lddqu_si256(in + 25);
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 15),
                                                 _mm256_slli_epi32(w1, 17)))));
  w0 = _mm256_lddqu_si256(in + 26);
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 12),
                                                 _mm256_slli_epi32(w0, 20)))));
  w1 = _mm256_lddqu_si256(in + 27);
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 9),
                                                 _mm256_slli_epi32(w1, 23)))));
  w0 = _mm256_lddqu_si256(in + 28);
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 6),
                                                 _mm256_slli_epi32(w0, 26)))));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 3)));
}

/* we packed 256 30-bit values, touching 30 256-bit words, using 480 bytes */
static void filterlt30(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  30 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(1073741823);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                 _mm256_slli_epi32(w1, 2)))));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4)))));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 26),
                                                 _mm256_slli_epi32(w1, 6)))));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8)))));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 22),
                                                 _mm256_slli_epi32(w1, 10)))));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12)))));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 18),
                                                 _mm256_slli_epi32(w1, 14)))));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16)))));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 14),
                                                 _mm256_slli_epi32(w1, 18)))));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 12),
                                                 _mm256_slli_epi32(w0, 20)))));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 10),
                                                 _mm256_slli_epi32(w1, 22)))));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 8),
                                                 _mm256_slli_epi32(w0, 24)))));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 6),
                                                 _mm256_slli_epi32(w1, 26)))));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 4),
                                                 _mm256_slli_epi32(w0, 28)))));
  _mm256_storeu_si256(out + 15,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 2)));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 16, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w1)));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                 _mm256_slli_epi32(w0, 2)))));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 28),
                                                 _mm256_slli_epi32(w1, 4)))));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                 _mm256_slli_epi32(w0, 6)))));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 24),
                                                 _mm256_slli_epi32(w1, 8)))));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 22),
                                                 _mm256_slli_epi32(w0, 10)))));
  w1 = _mm256_lddqu_si256(in + 21);
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 20),
                                                 _mm256_slli_epi32(w1, 12)))));
  w0 = _mm256_lddqu_si256(in + 22);
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 18),
                                                 _mm256_slli_epi32(w0, 14)))));
  w1 = _mm256_lddqu_si256(in + 23);
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 16),
                                                 _mm256_slli_epi32(w1, 16)))));
  w0 = _mm256_lddqu_si256(in + 24);
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 14),
                                                 _mm256_slli_epi32(w0, 18)))));
  w1 = _mm256_lddqu_si256(in + 25);
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 12),
                                                 _mm256_slli_epi32(w1, 20)))));
  w0 = _mm256_lddqu_si256(in + 26);
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 10),
                                                 _mm256_slli_epi32(w0, 22)))));
  w1 = _mm256_lddqu_si256(in + 27);
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 8),
                                                 _mm256_slli_epi32(w1, 24)))));
  w0 = _mm256_lddqu_si256(in + 28);
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 6),
                                                 _mm256_slli_epi32(w0, 26)))));
  w1 = _mm256_lddqu_si256(in + 29);
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 4),
                                                 _mm256_slli_epi32(w1, 28)))));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w1, 2)));
}

/* we packed 256 31-bit values, touching 31 256-bit words, using 496 bytes */
static void filterlt31(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  31 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i mask = _mm256_set1_epi32(2147483647);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(
      out + 0, _mm256_cmpgt_epi32(broadcomp, _mm256_and_si256(mask, w0)));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(
      out + 1,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 31),
                                                 _mm256_slli_epi32(w1, 1)))));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(
      out + 2,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 30),
                                                 _mm256_slli_epi32(w0, 2)))));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(
      out + 3,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 29),
                                                 _mm256_slli_epi32(w1, 3)))));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(
      out + 4,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                 _mm256_slli_epi32(w0, 4)))));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(
      out + 5,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 27),
                                                 _mm256_slli_epi32(w1, 5)))));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(
      out + 6,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 26),
                                                 _mm256_slli_epi32(w0, 6)))));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(
      out + 7,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 25),
                                                 _mm256_slli_epi32(w1, 7)))));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(
      out + 8,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 24),
                                                 _mm256_slli_epi32(w0, 8)))));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(
      out + 9,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 23),
                                                 _mm256_slli_epi32(w1, 9)))));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(
      out + 10,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 22),
                                                 _mm256_slli_epi32(w0, 10)))));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(
      out + 11,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 21),
                                                 _mm256_slli_epi32(w1, 11)))));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(
      out + 12,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 20),
                                                 _mm256_slli_epi32(w0, 12)))));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(
      out + 13,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 19),
                                                 _mm256_slli_epi32(w1, 13)))));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(
      out + 14,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 18),
                                                 _mm256_slli_epi32(w0, 14)))));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(
      out + 15,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 17),
                                                 _mm256_slli_epi32(w1, 15)))));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(
      out + 16,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 16),
                                                 _mm256_slli_epi32(w0, 16)))));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(
      out + 17,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 15),
                                                 _mm256_slli_epi32(w1, 17)))));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(
      out + 18,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 14),
                                                 _mm256_slli_epi32(w0, 18)))));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(
      out + 19,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 13),
                                                 _mm256_slli_epi32(w1, 19)))));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(
      out + 20,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 12),
                                                 _mm256_slli_epi32(w0, 20)))));
  w1 = _mm256_lddqu_si256(in + 21);
  _mm256_storeu_si256(
      out + 21,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 11),
                                                 _mm256_slli_epi32(w1, 21)))));
  w0 = _mm256_lddqu_si256(in + 22);
  _mm256_storeu_si256(
      out + 22,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 10),
                                                 _mm256_slli_epi32(w0, 22)))));
  w1 = _mm256_lddqu_si256(in + 23);
  _mm256_storeu_si256(
      out + 23,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 9),
                                                 _mm256_slli_epi32(w1, 23)))));
  w0 = _mm256_lddqu_si256(in + 24);
  _mm256_storeu_si256(
      out + 24,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 8),
                                                 _mm256_slli_epi32(w0, 24)))));
  w1 = _mm256_lddqu_si256(in + 25);
  _mm256_storeu_si256(
      out + 25,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 7),
                                                 _mm256_slli_epi32(w1, 25)))));
  w0 = _mm256_lddqu_si256(in + 26);
  _mm256_storeu_si256(
      out + 26,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 6),
                                                 _mm256_slli_epi32(w0, 26)))));
  w1 = _mm256_lddqu_si256(in + 27);
  _mm256_storeu_si256(
      out + 27,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 5),
                                                 _mm256_slli_epi32(w1, 27)))));
  w0 = _mm256_lddqu_si256(in + 28);
  _mm256_storeu_si256(
      out + 28,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 4),
                                                 _mm256_slli_epi32(w0, 28)))));
  w1 = _mm256_lddqu_si256(in + 29);
  _mm256_storeu_si256(
      out + 29,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w0, 3),
                                                 _mm256_slli_epi32(w1, 29)))));
  w0 = _mm256_lddqu_si256(in + 30);
  _mm256_storeu_si256(
      out + 30,
      _mm256_cmpgt_epi32(
          broadcomp,
          _mm256_and_si256(mask, _mm256_or_si256(_mm256_srli_epi32(w1, 2),
                                                 _mm256_slli_epi32(w0, 30)))));
  _mm256_storeu_si256(out + 31,
                      _mm256_cmpgt_epi32(broadcomp, _mm256_srli_epi32(w0, 1)));
}

/* we packed 256 32-bit values, touching 32 256-bit words, using 512 bytes */
static void filterlt32(const __m256i *in, u32 *matches, const INTEGER comp) {
  /* we are going to access  32 256-bit words */
  __m256i w0, w1;
  auto out = reinterpret_cast<__m256i *>(matches);
  const __m256i broadcomp = _mm256_set1_epi32(comp);
  w0 = _mm256_lddqu_si256(in);
  _mm256_storeu_si256(out + 0, _mm256_cmpgt_epi32(broadcomp, w0));
  w1 = _mm256_lddqu_si256(in + 1);
  _mm256_storeu_si256(out + 1, _mm256_cmpgt_epi32(broadcomp, w1));
  w0 = _mm256_lddqu_si256(in + 2);
  _mm256_storeu_si256(out + 2, _mm256_cmpgt_epi32(broadcomp, w0));
  w1 = _mm256_lddqu_si256(in + 3);
  _mm256_storeu_si256(out + 3, _mm256_cmpgt_epi32(broadcomp, w1));
  w0 = _mm256_lddqu_si256(in + 4);
  _mm256_storeu_si256(out + 4, _mm256_cmpgt_epi32(broadcomp, w0));
  w1 = _mm256_lddqu_si256(in + 5);
  _mm256_storeu_si256(out + 5, _mm256_cmpgt_epi32(broadcomp, w1));
  w0 = _mm256_lddqu_si256(in + 6);
  _mm256_storeu_si256(out + 6, _mm256_cmpgt_epi32(broadcomp, w0));
  w1 = _mm256_lddqu_si256(in + 7);
  _mm256_storeu_si256(out + 7, _mm256_cmpgt_epi32(broadcomp, w1));
  w0 = _mm256_lddqu_si256(in + 8);
  _mm256_storeu_si256(out + 8, _mm256_cmpgt_epi32(broadcomp, w0));
  w1 = _mm256_lddqu_si256(in + 9);
  _mm256_storeu_si256(out + 9, _mm256_cmpgt_epi32(broadcomp, w1));
  w0 = _mm256_lddqu_si256(in + 10);
  _mm256_storeu_si256(out + 10, _mm256_cmpgt_epi32(broadcomp, w0));
  w1 = _mm256_lddqu_si256(in + 11);
  _mm256_storeu_si256(out + 11, _mm256_cmpgt_epi32(broadcomp, w1));
  w0 = _mm256_lddqu_si256(in + 12);
  _mm256_storeu_si256(out + 12, _mm256_cmpgt_epi32(broadcomp, w0));
  w1 = _mm256_lddqu_si256(in + 13);
  _mm256_storeu_si256(out + 13, _mm256_cmpgt_epi32(broadcomp, w1));
  w0 = _mm256_lddqu_si256(in + 14);
  _mm256_storeu_si256(out + 14, _mm256_cmpgt_epi32(broadcomp, w0));
  w1 = _mm256_lddqu_si256(in + 15);
  _mm256_storeu_si256(out + 15, _mm256_cmpgt_epi32(broadcomp, w1));
  w0 = _mm256_lddqu_si256(in + 16);
  _mm256_storeu_si256(out + 16, _mm256_cmpgt_epi32(broadcomp, w0));
  w1 = _mm256_lddqu_si256(in + 17);
  _mm256_storeu_si256(out + 17, _mm256_cmpgt_epi32(broadcomp, w1));
  w0 = _mm256_lddqu_si256(in + 18);
  _mm256_storeu_si256(out + 18, _mm256_cmpgt_epi32(broadcomp, w0));
  w1 = _mm256_lddqu_si256(in + 19);
  _mm256_storeu_si256(out + 19, _mm256_cmpgt_epi32(broadcomp, w1));
  w0 = _mm256_lddqu_si256(in + 20);
  _mm256_storeu_si256(out + 20, _mm256_cmpgt_epi32(broadcomp, w0));
  w1 = _mm256_lddqu_si256(in + 21);
  _mm256_storeu_si256(out + 21, _mm256_cmpgt_epi32(broadcomp, w1));
  w0 = _mm256_lddqu_si256(in + 22);
  _mm256_storeu_si256(out + 22, _mm256_cmpgt_epi32(broadcomp, w0));
  w1 = _mm256_lddqu_si256(in + 23);
  _mm256_storeu_si256(out + 23, _mm256_cmpgt_epi32(broadcomp, w1));
  w0 = _mm256_lddqu_si256(in + 24);
  _mm256_storeu_si256(out + 24, _mm256_cmpgt_epi32(broadcomp, w0));
  w1 = _mm256_lddqu_si256(in + 25);
  _mm256_storeu_si256(out + 25, _mm256_cmpgt_epi32(broadcomp, w1));
  w0 = _mm256_lddqu_si256(in + 26);
  _mm256_storeu_si256(out + 26, _mm256_cmpgt_epi32(broadcomp, w0));
  w1 = _mm256_lddqu_si256(in + 27);
  _mm256_storeu_si256(out + 27, _mm256_cmpgt_epi32(broadcomp, w1));
  w0 = _mm256_lddqu_si256(in + 28);
  _mm256_storeu_si256(out + 28, _mm256_cmpgt_epi32(broadcomp, w0));
  w1 = _mm256_lddqu_si256(in + 29);
  _mm256_storeu_si256(out + 29, _mm256_cmpgt_epi32(broadcomp, w1));
  w0 = _mm256_lddqu_si256(in + 30);
  _mm256_storeu_si256(out + 30, _mm256_cmpgt_epi32(broadcomp, w0));
  w1 = _mm256_lddqu_si256(in + 31);
  _mm256_storeu_si256(out + 31, _mm256_cmpgt_epi32(broadcomp, w1));
}

void filtereq(const __m256i *in, u32 *matches, const INTEGER comp,
              const u8 bit) {
  switch (bit) {
  case 0:
    filtereq0(in, matches, comp);
    break;
  case 1:
    filtereq1(in, matches, comp);
    break;
  case 2:
    filtereq2(in, matches, comp);
    break;
  case 3:
    filtereq3(in, matches, comp);
    break;
  case 4:
    filtereq4(in, matches, comp);
    break;
  case 5:
    filtereq5(in, matches, comp);
    break;
  case 6:
    filtereq6(in, matches, comp);
    break;
  case 7:
    filtereq7(in, matches, comp);
    break;
  case 8:
    filtereq8(in, matches, comp);
    break;
  case 9:
    filtereq9(in, matches, comp);
    break;
  case 10:
    filtereq10(in, matches, comp);
    break;
  case 11:
    filtereq11(in, matches, comp);
    break;
  case 12:
    filtereq12(in, matches, comp);
    break;
  case 13:
    filtereq13(in, matches, comp);
    break;
  case 14:
    filtereq14(in, matches, comp);
    break;
  case 15:
    filtereq15(in, matches, comp);
    break;
  case 16:
    filtereq16(in, matches, comp);
    break;
  case 17:
    filtereq17(in, matches, comp);
    break;
  case 18:
    filtereq18(in, matches, comp);
    break;
  case 19:
    filtereq19(in, matches, comp);
    break;
  case 20:
    filtereq20(in, matches, comp);
    break;
  case 21:
    filtereq21(in, matches, comp);
    break;
  case 22:
    filtereq22(in, matches, comp);
    break;
  case 23:
    filtereq23(in, matches, comp);
    break;
  case 24:
    filtereq24(in, matches, comp);
    break;
  case 25:
    filtereq25(in, matches, comp);
    break;
  case 26:
    filtereq26(in, matches, comp);
    break;
  case 27:
    filtereq27(in, matches, comp);
    break;
  case 28:
    filtereq28(in, matches, comp);
    break;
  case 29:
    filtereq29(in, matches, comp);
    break;
  case 30:
    filtereq30(in, matches, comp);
    break;
  case 31:
    filtereq31(in, matches, comp);
    break;
  case 32:
    filtereq32(in, matches, comp);
    break;
  }
}

void filterneq(const __m256i *in, u32 *matches, const INTEGER comp,
               const u8 bit) {
  switch (bit) {
  case 0:
    filterneq0(in, matches, comp);
    break;
  case 1:
    filterneq1(in, matches, comp);
    break;
  case 2:
    filterneq2(in, matches, comp);
    break;
  case 3:
    filterneq3(in, matches, comp);
    break;
  case 4:
    filterneq4(in, matches, comp);
    break;
  case 5:
    filterneq5(in, matches, comp);
    break;
  case 6:
    filterneq6(in, matches, comp);
    break;
  case 7:
    filterneq7(in, matches, comp);
    break;
  case 8:
    filterneq8(in, matches, comp);
    break;
  case 9:
    filterneq9(in, matches, comp);
    break;
  case 10:
    filterneq10(in, matches, comp);
    break;
  case 11:
    filterneq11(in, matches, comp);
    break;
  case 12:
    filterneq12(in, matches, comp);
    break;
  case 13:
    filterneq13(in, matches, comp);
    break;
  case 14:
    filterneq14(in, matches, comp);
    break;
  case 15:
    filterneq15(in, matches, comp);
    break;
  case 16:
    filterneq16(in, matches, comp);
    break;
  case 17:
    filterneq17(in, matches, comp);
    break;
  case 18:
    filterneq18(in, matches, comp);
    break;
  case 19:
    filterneq19(in, matches, comp);
    break;
  case 20:
    filterneq20(in, matches, comp);
    break;
  case 21:
    filterneq21(in, matches, comp);
    break;
  case 22:
    filterneq22(in, matches, comp);
    break;
  case 23:
    filterneq23(in, matches, comp);
    break;
  case 24:
    filterneq24(in, matches, comp);
    break;
  case 25:
    filterneq25(in, matches, comp);
    break;
  case 26:
    filterneq26(in, matches, comp);
    break;
  case 27:
    filterneq27(in, matches, comp);
    break;
  case 28:
    filterneq28(in, matches, comp);
    break;
  case 29:
    filterneq29(in, matches, comp);
    break;
  case 30:
    filterneq30(in, matches, comp);
    break;
  case 31:
    filterneq31(in, matches, comp);
    break;
  case 32:
    filterneq32(in, matches, comp);
    break;
  }
}

void filtergt(const __m256i *in, u32 *matches, const INTEGER comp,
              const u8 bit) {
  switch (bit) {
  case 0:
    filtergt0(in, matches, comp);
    break;
  case 1:
    filtergt1(in, matches, comp);
    break;
  case 2:
    filtergt2(in, matches, comp);
    break;
  case 3:
    filtergt3(in, matches, comp);
    break;
  case 4:
    filtergt4(in, matches, comp);
    break;
  case 5:
    filtergt5(in, matches, comp);
    break;
  case 6:
    filtergt6(in, matches, comp);
    break;
  case 7:
    filtergt7(in, matches, comp);
    break;
  case 8:
    filtergt8(in, matches, comp);
    break;
  case 9:
    filtergt9(in, matches, comp);
    break;
  case 10:
    filtergt10(in, matches, comp);
    break;
  case 11:
    filtergt11(in, matches, comp);
    break;
  case 12:
    filtergt12(in, matches, comp);
    break;
  case 13:
    filtergt13(in, matches, comp);
    break;
  case 14:
    filtergt14(in, matches, comp);
    break;
  case 15:
    filtergt15(in, matches, comp);
    break;
  case 16:
    filtergt16(in, matches, comp);
    break;
  case 17:
    filtergt17(in, matches, comp);
    break;
  case 18:
    filtergt18(in, matches, comp);
    break;
  case 19:
    filtergt19(in, matches, comp);
    break;
  case 20:
    filtergt20(in, matches, comp);
    break;
  case 21:
    filtergt21(in, matches, comp);
    break;
  case 22:
    filtergt22(in, matches, comp);
    break;
  case 23:
    filtergt23(in, matches, comp);
    break;
  case 24:
    filtergt24(in, matches, comp);
    break;
  case 25:
    filtergt25(in, matches, comp);
    break;
  case 26:
    filtergt26(in, matches, comp);
    break;
  case 27:
    filtergt27(in, matches, comp);
    break;
  case 28:
    filtergt28(in, matches, comp);
    break;
  case 29:
    filtergt29(in, matches, comp);
    break;
  case 30:
    filtergt30(in, matches, comp);
    break;
  case 31:
    filtergt31(in, matches, comp);
    break;
  case 32:
    filtergt32(in, matches, comp);
    break;
  }
}

void filterlt(const __m256i *in, u32 *matches, const INTEGER comp,
              const u8 bit) {
  switch (bit) {
  case 0:
    filterlt0(in, matches, comp);
    break;
  case 1:
    filterlt1(in, matches, comp);
    break;
  case 2:
    filterlt2(in, matches, comp);
    break;
  case 3:
    filterlt3(in, matches, comp);
    break;
  case 4:
    filterlt4(in, matches, comp);
    break;
  case 5:
    filterlt5(in, matches, comp);
    break;
  case 6:
    filterlt6(in, matches, comp);
    break;
  case 7:
    filterlt7(in, matches, comp);
    break;
  case 8:
    filterlt8(in, matches, comp);
    break;
  case 9:
    filterlt9(in, matches, comp);
    break;
  case 10:
    filterlt10(in, matches, comp);
    break;
  case 11:
    filterlt11(in, matches, comp);
    break;
  case 12:
    filterlt12(in, matches, comp);
    break;
  case 13:
    filterlt13(in, matches, comp);
    break;
  case 14:
    filterlt14(in, matches, comp);
    break;
  case 15:
    filterlt15(in, matches, comp);
    break;
  case 16:
    filterlt16(in, matches, comp);
    break;
  case 17:
    filterlt17(in, matches, comp);
    break;
  case 18:
    filterlt18(in, matches, comp);
    break;
  case 19:
    filterlt19(in, matches, comp);
    break;
  case 20:
    filterlt20(in, matches, comp);
    break;
  case 21:
    filterlt21(in, matches, comp);
    break;
  case 22:
    filterlt22(in, matches, comp);
    break;
  case 23:
    filterlt23(in, matches, comp);
    break;
  case 24:
    filterlt24(in, matches, comp);
    break;
  case 25:
    filterlt25(in, matches, comp);
    break;
  case 26:
    filterlt26(in, matches, comp);
    break;
  case 27:
    filterlt27(in, matches, comp);
    break;
  case 28:
    filterlt28(in, matches, comp);
    break;
  case 29:
    filterlt29(in, matches, comp);
    break;
  case 30:
    filterlt30(in, matches, comp);
    break;
  case 31:
    filterlt31(in, matches, comp);
    break;
  case 32:
    filterlt32(in, matches, comp);
    break;
  }
}

void filter(const __m256i *in, u32 *matches, const u8 bit,
            const algebra::Predicate<INTEGER> &predicate) {
  const INTEGER comp = predicate.getValue();
  switch (predicate.getType()) {
  case algebra::PredicateType::EQ:
    filtereq(in, matches, comp, bit);
    break;
  case algebra::PredicateType::INEQ:
    filterneq(in, matches, comp, bit);
    break;
  case algebra::PredicateType::GT:
    filtergt(in, matches, comp, bit);
    break;
  case algebra::PredicateType::LT:
    filterlt(in, matches, comp, bit);
    break;
  default:
    break;
  }
}
//---------------------------------------------------------------------------
} // namespace avx
//---------------------------------------------------------------------------
} // namespace simd32
//---------------------------------------------------------------------------
} // namespace bitpacking
//---------------------------------------------------------------------------
} // namespace compression
