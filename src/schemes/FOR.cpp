#include "schemes/FOR.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
u32 FOR::compress(const INTEGER *src, u8 *dest, const Statistics *stats,
                  const u32 total_size, const u16 block_size) {
  auto &layout = *reinterpret_cast<FORLayout *>(dest);
  layout.reference = stats->min;
  layout.pack_size = compressDispatch(src, layout.data, stats, total_size);

  return offsetof(FORLayout, data) + layout.pack_size / 8 * total_size;
}
//---------------------------------------------------------------------------
void FOR::decompress(INTEGER *dest, const u8 *src, const u32 total_size,
                     const u16 block_size) {
  const auto &layout = *reinterpret_cast<const FORLayout *>(src);
  decompressDispatch(dest, layout.data, layout.reference, total_size,
                     layout.pack_size);
}
//---------------------------------------------------------------------------
CompressionSchemeType FOR::getType() { return CompressionSchemeType::kFOR; }
//---------------------------------------------------------------------------
} // namespace compression