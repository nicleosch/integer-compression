#include <cassert>
//---------------------------------------------------------------------------
#include "schemes/FOR.hpp"
#include "common/BitPacking.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
u32 FOR::compress(
    const INTEGER* src,
    u8* dest,
    const Statistics* stats,
    const u32 total_size,
    const u16 block_size
) {
  assert(block_size == 0);

  auto& layout = *reinterpret_cast<FORLayout*>(dest);
  layout.reference = stats->min;
  layout.pack_size = compressImpl(src, layout.data, stats, total_size, BitPacking::packFixed);

  return offsetof(FORLayout, data) + layout.pack_size / 8 * total_size;
}
//---------------------------------------------------------------------------
void FOR::decompress(
    INTEGER* dest,
    const u8* src,
    const u32 total_size,
    const u16 block_size
) {
  const auto& layout = *reinterpret_cast<const FORLayout*>(src);
  decompressImpl(dest, layout.data, total_size, layout.reference, layout.pack_size);
}
//---------------------------------------------------------------------------
CompressionSchemeType FOR::getType() {
  return CompressionSchemeType::kFOR;
}
//---------------------------------------------------------------------------
u8 FOR::compressImpl(
  const INTEGER* src,
  u8* dest,
  const Statistics* stats,
  const u32 total_size,
  u8(*bitpack_func)(const INTEGER*, u8*, const u32, INTEGER)
) {
  // Normalize
  vector<INTEGER> normalized;
  for(u32 i = 0; i < total_size; ++i) {
    normalized.push_back(src[i] - stats->min);
  }

  // Compress
  u8 pack_size = bitpack_func(normalized.data(), dest, total_size, stats->max - stats->min);

  return pack_size;
}
//---------------------------------------------------------------------------
void FOR::decompressImpl(
  INTEGER* dest,
  const u8* src,
  const u32 total_size,
  const u32 reference,
  const u8 pack_size
) {
  // Decompress
  BitPacking::unpack(dest, src, total_size, pack_size);

  // Denormalize
  for(u32 i = 0; i < total_size; ++i) {
    dest[i] += reference;
  }
}
//---------------------------------------------------------------------------
}  // namespace compression