#include "bitpacking/BitPacking.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
u8 BitPacking::packFixed(const INTEGER *src, u8 *dest, const u32 size,
                         INTEGER diff) {
  auto pack_size = packSizeFor(diff);

  if (pack_size <= 8) {
    compress<u8>(src, dest, size, 8);
    return 8;
  } else if (pack_size <= 16) {
    compress<u16>(src, dest, size, 16);
    return 16;
  } else {
    compress<u32>(src, dest, size, 32);
    return 32;
  }
}
//---------------------------------------------------------------------------
u8 BitPacking::packArbitrary(const INTEGER *src, u8 *dest, const u32 size,
                             INTEGER diff) {
  auto pack_size = packSizeFor(diff);

  if (pack_size <= 8) {
    compress<u8>(src, dest, size, pack_size);
  } else if (pack_size <= 16) {
    compress<u16>(src, dest, size, pack_size);
  } else {
    compress<u32>(src, dest, size, pack_size);
  }

  return pack_size;
}
//---------------------------------------------------------------------------
void BitPacking::unpack(INTEGER *dest, const u8 *src, const u32 size,
                        const u8 pack_size) {
  if (pack_size <= 8) {
    decompress<u8>(dest, src, size, pack_size);
  } else if (pack_size <= 16) {
    decompress<u16>(dest, src, size, pack_size);
  } else {
    decompress<u32>(dest, src, size, pack_size);
  }
}
//---------------------------------------------------------------------------
u8 BitPacking::packSizeFor(INTEGER value) {
  if (value < 0)
    return 32; // TODO: Improve handling negative integers
  if (value == 0)
    value += 1;
  return static_cast<u8>(sizeof(INTEGER) * 8) -
         __builtin_clz(static_cast<u32>(value));
}
//---------------------------------------------------------------------------
} // namespace compression