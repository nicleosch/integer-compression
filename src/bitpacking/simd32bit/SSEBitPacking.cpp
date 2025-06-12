#include <simdcomp.h>
//---------------------------------------------------------------------------
#include "bitpacking/simd32bit/SSEBitPacking.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
namespace bitpacking {
//---------------------------------------------------------------------------
namespace simd32 {
//---------------------------------------------------------------------------
namespace sse {
//---------------------------------------------------------------------------
void pack(const u32 *in, __m128i *out, const u8 bit) {
  return simdpackwithoutmask(in, out, bit);
}
//---------------------------------------------------------------------------
void unpack(const __m128i *in, u32 *out, const u8 bit) {
  return simdunpack(in, out, bit);
}
//---------------------------------------------------------------------------
__m128i *packLength(const u32 *in, u16 length, __m128i *out, const u8 bit) {
  return simdpack_length(in, length, out, bit);
}
//---------------------------------------------------------------------------
const __m128i *unpackLength(const __m128i *in, u16 length, u32 *out,
                            const u8 bit) {
  return simdunpack_length(in, length, out, bit);
}
//---------------------------------------------------------------------------
__m128i *packShortLength(const u32 *in, u16 length, __m128i *out,
                         const u8 bit) {
  return simdpack_shortlength(in, length, out, bit);
}
//---------------------------------------------------------------------------
const __m128i *unpackShortLength(const __m128i *in, u16 length, u32 *out,
                                 const u8 bit) {
  return simdunpack_shortlength(in, length, out, bit);
}
//---------------------------------------------------------------------------
} // namespace sse
//---------------------------------------------------------------------------
} // namespace simd32
//---------------------------------------------------------------------------
} // namespace bitpacking
//---------------------------------------------------------------------------
} // namespace compression