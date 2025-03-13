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

  // Normalize
  vector<INTEGER> normalized;
  for(u32 i = 0; i < total_size; ++i) {
    normalized.push_back(src[i] - layout.reference);
  }

  // Compress
  layout.pack_size = BitPacking::packFixed(normalized.data(), layout.data, total_size, stats->max - stats->min);

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

  // Decompress
  BitPacking::unpack(dest, layout.data, total_size, layout.pack_size);

  // Denormalize
  for(u32 i = 0; i < total_size; ++i) {
    dest[i] += layout.reference;
  }
}
//---------------------------------------------------------------------------
CompressionSchemeType FOR::getType() {
  return CompressionSchemeType::kFOR;
}
//---------------------------------------------------------------------------
}  // namespace compression