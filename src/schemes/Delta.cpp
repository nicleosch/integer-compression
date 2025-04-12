#include "schemes/Delta.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
u32 Delta::compress(const INTEGER *src, const u32 size, u8 *dest,
                    const Statistics *stats) {
  auto &layout = *reinterpret_cast<DeltaLayout *>(dest);
  layout.pack_size = compressDispatch(src, size, layout.data, stats);
  layout.base = stats->min;
  return offsetof(DeltaLayout, data) + layout.pack_size / 8 * size;
}
//---------------------------------------------------------------------------
void Delta::decompress(INTEGER *dest, const u32 size, const u8 *src) {
  const auto &layout = *reinterpret_cast<const DeltaLayout *>(src);
  decompressDispatch(dest, size, layout);
}
//---------------------------------------------------------------------------
bool Delta::isPartitioningScheme() { return false; }
//---------------------------------------------------------------------------
u8 Delta::compressDispatch(const INTEGER *src, const u32 size, u8 *dest,
                           const Statistics *stats) {
  // Note: The following code assumes that the first element in the array is the
  // minimum and that the data increases monotonically.
  if (stats->delta_bits <= 8) {
    compressImpl<u8>(src, size, dest, stats->min);
    return 8;
  } else if (stats->delta_bits <= 16) {
    compressImpl<u16>(src, size, dest, stats->min);
    return 16;
  } else {
    compressImpl<u32>(src, size, dest, stats->min);
    return 32;
  }
}
//---------------------------------------------------------------------------
void Delta::decompressDispatch(INTEGER *dest, const u32 size,
                               const DeltaLayout &layout) {
  switch (layout.pack_size) {
  case 8:
    decompressImpl<u8>(dest, size, layout.data, layout.base);
    return;
  case 16:
    decompressImpl<u16>(dest, size, layout.data, layout.base);
    return;
  case 32:
    decompressImpl<u32>(dest, size, layout.data, layout.base);
    return;
  default:
    return;
  }
}
//---------------------------------------------------------------------------
} // namespace compression