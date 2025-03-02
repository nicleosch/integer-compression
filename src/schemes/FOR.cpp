#include <cassert>
//---------------------------------------------------------------------------
#include "schemes/FOR.hpp"
#include "common/BitPacking.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
void FOR::compress(
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

  // TODO: Extract this into a global compressor functionality

  // Compress
  INTEGER diff = stats->max - stats->min;
  if(diff <= std::numeric_limits<u8>::max()) {
    layout.next_scheme = static_cast<u8>(CompressionSchemeType::kBitPacking8);
    pack<u8>(normalized.data(), layout.data, total_size, diff);
  } else if(diff <= std::numeric_limits<u16>::max()) {
    layout.next_scheme = static_cast<u8>(CompressionSchemeType::kBitPacking16);
    pack<u16>(normalized.data(), layout.data, total_size, diff);
  }
}
//---------------------------------------------------------------------------
void FOR::decompress(
    INTEGER* dest,
    const u8* src,
    const u32 total_size,
    const u16 block_size
) {
  const auto& layout = *reinterpret_cast<const FORLayout*>(src);

  // TODO: Extract this into a global decompressor functionality

  // Decompress
  auto scheme = static_cast<CompressionSchemeType>(layout.next_scheme);
  if(scheme == CompressionSchemeType::kBitPacking8) {
    unpack<u8>(dest, layout.data, total_size);
  } else if(scheme == CompressionSchemeType::kBitPacking16) {
    unpack<u16>(dest, layout.data, total_size);
  }

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