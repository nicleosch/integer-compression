#include "schemes/FOR.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
u32 FOR::compress(const INTEGER *src, const u32 total_size, u8 *dest,
                  const Statistics *stats) {
  auto &layout = *reinterpret_cast<FORLayout *>(dest);
  layout.reference = stats->min;
  layout.pack_size = compressDispatch(src, total_size, layout.data, stats);

  return offsetof(FORLayout, data) + layout.pack_size / 8 * total_size;
}
//---------------------------------------------------------------------------
void FOR::decompress(INTEGER *dest, const u32 total_size, const u8 *src) {
  const auto &layout = *reinterpret_cast<const FORLayout *>(src);
  decompressDispatch(dest, total_size, layout.data, layout.reference,
                     layout.pack_size);
}
//---------------------------------------------------------------------------
u8 FOR::compressDispatch(const INTEGER *src, const u32 size, u8 *dest,
                         const Statistics *stats) {
  if (stats->required_bits <= 8) {
    compressImpl<u8>(src, size, dest, stats);
    return 8;
  } else if (stats->required_bits <= 16) {
    compressImpl<u16>(src, size, dest, stats);
    return 16;
  } else {
    compressImpl<u32>(src, size, dest, stats);
    return 32;
  }
}
//---------------------------------------------------------------------------
void FOR::decompressDispatch(INTEGER *dest, const u32 size, const u8 *src,
                             const u32 reference, const u8 pack_size) {
  switch (pack_size) {
  case 8:
    decompressImpl<u8>(dest, size, src, reference);
    return;
  case 16:
    decompressImpl<u16>(dest, size, src, reference);
    return;
  case 32:
    decompressImpl<u32>(dest, size, src, reference);
    return;
  default:
    break;
  }
}
//---------------------------------------------------------------------------
} // namespace compression