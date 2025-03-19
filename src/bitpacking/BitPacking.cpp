#include <simdcomp.h>
//---------------------------------------------------------------------------
#include "bitpacking/BitPacking.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
namespace bitpacking {
//---------------------------------------------------------------------------
static void pack64(const INTEGER *src, u8 *dest, const u8 pack_size) {
  simdpack_shortlength(reinterpret_cast<const u32*>(src), 64, reinterpret_cast<__m128i*>(dest), pack_size);
}
//---------------------------------------------------------------------------
static void pack128(const INTEGER *src, u8 *dest, const u8 pack_size) {
  simdpack(reinterpret_cast<const u32*>(src), reinterpret_cast<__m128i*>(dest), pack_size);
}
//---------------------------------------------------------------------------
static void pack256(const INTEGER *src, u8 *dest, const u8 pack_size) {
  avxpack(reinterpret_cast<const u32*>(src), reinterpret_cast<__m256i*>(dest), pack_size);
}
//---------------------------------------------------------------------------
static void pack512(const INTEGER *src, u8 *dest, const u8 pack_size) {
  avx512pack(reinterpret_cast<const u32*>(src), reinterpret_cast<__m512i*>(dest), pack_size);
}
//---------------------------------------------------------------------------
static void packArbitrary(const INTEGER *src, u8 *dest, const u32 length, const u8 pack_size) {
  simdpack_length(reinterpret_cast<const u32*>(src), length, reinterpret_cast<__m128i*>(dest), pack_size);
}
//---------------------------------------------------------------------------
static void unpack64(INTEGER *dest, const u8 *src, const u8 pack_size) {
  simdunpack_shortlength(reinterpret_cast<const __m128i*>(src), 64, reinterpret_cast<u32*>(dest), pack_size);
}
//---------------------------------------------------------------------------
static void unpack128(INTEGER *dest, const u8 *src, const u8 pack_size) {
  simdunpack(reinterpret_cast<const __m128i*>(src), reinterpret_cast<u32*>(dest), pack_size);
}
//---------------------------------------------------------------------------
static void unpack256(INTEGER *dest, const u8 *src, const u8 pack_size) {
  avxunpack(reinterpret_cast<const __m256i*>(src), reinterpret_cast<u32*>(dest), pack_size);
}
//---------------------------------------------------------------------------
static void unpack512(INTEGER *dest, const u8 *src, const u8 pack_size) {
  avx512unpack(reinterpret_cast<const __m512i*>(src), reinterpret_cast<u32*>(dest), pack_size);
}
//---------------------------------------------------------------------------
static void unpackArbitrary(INTEGER *dest, const u8 *src, const u32 length, const u8 pack_size) {
  simdunpack_length(reinterpret_cast<const __m128i*>(src), length, reinterpret_cast<u32*>(dest), pack_size);
}
//---------------------------------------------------------------------------
void pack(const INTEGER *src, u8 *dest, const u32 length, const u8 pack_size) {
  switch(length) {
  case 64:  
    pack64(src, dest, pack_size);
    return;
  case 128:
    pack128(src, dest, pack_size);
    return;
  case 256:
    pack256(src, dest, pack_size);
    return;
  case 512:
    pack512(src, dest, pack_size);
    return;
  default:
    packArbitrary(src, dest, length, pack_size);
    break;
  }
}
//---------------------------------------------------------------------------
void unpack(INTEGER *dest, const u8 *src, const u32 length, const u8 pack_size) {
  switch(length) {
  case 64:  
    unpack64(dest, src, pack_size);
    return;
  case 128:
    unpack128(dest, src, pack_size);
    return;
  case 256:
    unpack256(dest, src, pack_size);
    return;
  case 512:
    unpack512(dest, src, pack_size);
    return;
  default:
    unpackArbitrary(dest, src, length, pack_size);
    break;
  }
}
//---------------------------------------------------------------------------
}  // namespace bitpacking
//---------------------------------------------------------------------------
}  // namespace compression