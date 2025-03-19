#include <cmath>
//---------------------------------------------------------------------------
#include "schemes/FORn.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
u32 FORn::compress(const INTEGER *src, u8 *dest, const Statistics *stats,
                   const u32 total_size, const u16 block_size) {
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

    slot.pack_size =
        FOR::compressDispatch(src, data_ptr + offset, &stats[block_i], size);
    slot.reference = stats[block_i].min;
    slot.offset = offset;

    offset += std::ceil(static_cast<double>(slot.pack_size * size) / 8);
    src += size;
  }

  return sizeof(FORnLayout) + block_count * sizeof(FORnSlot) + offset;
}
//---------------------------------------------------------------------------
void FORn::decompress(INTEGER *dest, const u8 *src, const u32 total_size,
                      const u16 block_size) {
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

    FOR::decompressDispatch(dest, read_ptr, slot.reference, size,
                            slot.pack_size);

    dest += size;
  }
}
//---------------------------------------------------------------------------
CompressionSchemeType FORn::getType() { return CompressionSchemeType::kFORn; }
//---------------------------------------------------------------------------
} // namespace compression