#include "schemes/Delta.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
u32 Delta::compress(const INTEGER *src, const u32 total_size, u8 *dest,
                    const Statistics *stats) {
  auto &layout = *reinterpret_cast<DeltaLayout *>(dest);
  layout.pack_size = compressDispatch(src, total_size, layout.data, stats);
  return layout.pack_size / 8 * total_size;
}
//---------------------------------------------------------------------------
void Delta::decompress(INTEGER *dest, const u32 total_size, const u8 *src) {
  const auto &layout = *reinterpret_cast<const DeltaLayout *>(src);
  decompressDispatch(dest, total_size, layout.data, layout.pack_size);
}
//---------------------------------------------------------------------------
u8 Delta::compressDispatch(const INTEGER *src, const u32 size, u8 *dest,
                           const Statistics *stats) {
  if (stats->delta_bits <= 8) {
    compressImpl<u8>(src, size, dest, stats);
    return 8;
  } else if (stats->delta_bits <= 16) {
    compressImpl<u16>(src, size, dest, stats);
    return 16;
  } else {
    compressImpl<u32>(src, size, dest, stats);
    return 32;
  }
}
//---------------------------------------------------------------------------
void Delta::decompressDispatch(INTEGER *dest, const u32 size, const u8 *src,
                               const u8 pack_size) {
  switch (pack_size) {
  case 8:
    decompressImpl<u8>(dest, size, src);
    return;
  case 16:
    decompressImpl<u16>(dest, size, src);
    return;
  case 32:
    decompressImpl<u32>(dest, size, src);
    return;
  default:
    return;
  }
}
//---------------------------------------------------------------------------
} // namespace compression