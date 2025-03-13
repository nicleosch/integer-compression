

#include "schemes/FORn.hpp"
#include "common/BitPacking.hpp"
#include <iostream>
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
u32 FORn::compress(
  const INTEGER* src,
  u8* dest,
  const Statistics* stats,
  const u32 total_size,
  const u16 block_size
) {
  // Layout: SLOTS | COMPRESSED DATA
  auto& layout = *reinterpret_cast<FORnLayout*>(dest);
  auto& header = *reinterpret_cast<FORnSlots*>(layout.data);

  const u32 block_count = std::ceil(static_cast<double>(total_size) / block_size);

  layout.data_offset = block_count * sizeof(FORnSlot);

  // Compress data
  u8* data_ptr = layout.data + layout.data_offset;
  u32 offset = 0;
  for(u32 block_i = 0; block_i < block_count; ++block_i) {
    auto& slot = header.slots[block_i];

    slot.reference = stats[block_i].min;

    u32 size =  std::min(static_cast<u32>(block_size), total_size - block_i * block_size);
    u8 pack_size = compressBlock(src, data_ptr + offset, slot, stats[block_i], size);

    slot.pack_size = pack_size;
    slot.offset = offset;
    offset += pack_size / 8 * size;

    src += size;
  }

  return sizeof(FORnLayout) + block_count * sizeof(FORnSlot) + offset;
}
//---------------------------------------------------------------------------
void FORn::decompress(
    INTEGER* dest,
    const u8* src,
    const u32 total_size,
    const u16 block_size
) {
  const auto& layout = *reinterpret_cast<const FORnLayout*>(src);
  const auto& header = *reinterpret_cast<const FORnSlots*>(layout.data);

  const u32 block_count = std::ceil(static_cast<double>(total_size) / block_size);

  const u8* read_ptr;
  for(u32 block_i = 0; block_i < block_count; ++block_i) {
    auto& slot = header.slots[block_i];

    read_ptr = layout.data + layout.data_offset + slot.offset;
    u32 size =  std::min(static_cast<u32>(block_size), total_size - block_i * block_size);

    // Decompress
    BitPacking::unpack(dest, read_ptr, size, slot.pack_size);

    // Denormalize
    for(u32 i = 0; i < size; ++i) {
        dest[i] += slot.reference;
    }

    dest += size;
  }
}
//---------------------------------------------------------------------------
CompressionSchemeType FORn::getType() {
  return CompressionSchemeType::kFORn;
}
//---------------------------------------------------------------------------
u8 FORn::compressBlock(
  const INTEGER* src,
  u8* dest,
  FORnSlot& slot,
  const Statistics& stats,
  const u32 size
) {
  // Normalize
  vector<INTEGER> normalized;
  for(u32 i = 0; i < size; ++i) {
    normalized.push_back(src[i] - slot.reference);
  }

  // Compress
  u8 pack_size = BitPacking::packFixed(normalized.data(), dest, size, stats.max - stats.min);

  return pack_size;
}
//---------------------------------------------------------------------------
}  // namespace compression