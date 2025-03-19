#include <cmath>
//---------------------------------------------------------------------------
#include "schemes/AdaptiveFORn.hpp"
#include "bitpacking/BitPacking.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
u32 AdaptiveFORn::compress(const INTEGER *src, u8 *dest,
                           const Statistics *stats, const u32 total_size,
                           const u16 block_size) {
  auto &layout = *reinterpret_cast<FORnLayout *>(dest);
  auto header = reinterpret_cast<FORnSlot *>(layout.data);

  const u32 block_count =
      std::ceil(static_cast<double>(total_size) / block_size);

  layout.data_offset = block_count * sizeof(FORnSlot);

  // Compress data
  u8 *data_ptr = layout.data + layout.data_offset;
  u32 offset = 0;
  for (u32 block_i = 0; block_i < block_count; ++block_i) {
    auto &slot = header[block_i];
    u32 size = std::min(static_cast<u32>(block_size),
                        total_size - block_i * block_size);

    slot.reference = stats[block_i].min;
    slot.pack_size = stats[block_i].required_bits;
    slot.offset = offset;

    // Normalize
    vector<INTEGER> normalized;
    for (u32 i = 0; i < size; ++i) {
      normalized.push_back(src[i] -  slot.reference);
    }

    // Compress
    bitpacking::pack(normalized.data(), data_ptr + offset, size, slot.pack_size);

    offset += std::ceil(static_cast<double>(slot.pack_size * size) / 8);
    src += size;
  }

  return sizeof(FORnLayout) + block_count * sizeof(FORnSlot) + offset;
}
//---------------------------------------------------------------------------
void AdaptiveFORn::decompress(INTEGER *dest, const u8 *src,
                              const u32 total_size, const u16 block_size) {
  const auto &layout = *reinterpret_cast<const FORnLayout *>(src);
  const auto &header = reinterpret_cast<const FORnSlot *>(layout.data);

  const u32 block_count =
      std::ceil(static_cast<double>(total_size) / block_size);

  const u8 *read_ptr;
  for (u32 block_i = 0; block_i < block_count; ++block_i) {
    auto &slot = header[block_i];
    u32 size = std::min(static_cast<u32>(block_size),
                        total_size - block_i * block_size);
    read_ptr = layout.data + layout.data_offset + slot.offset;

    // Decompress
    bitpacking::unpack(dest, read_ptr, size, slot.pack_size);

    // Denormalize
    for (u32 i = 0; i < size; ++i) {
      dest[i] += slot.reference;
    }

    dest += size;
  }
}
//---------------------------------------------------------------------------
CompressionSchemeType AdaptiveFORn::getType() {
  return CompressionSchemeType::kAdaptiveFORn;
}
//---------------------------------------------------------------------------
} // namespace compression