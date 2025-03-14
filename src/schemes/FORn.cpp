

#include "schemes/FORn.hpp"
#include "common/BitPacking.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
u32 FORn::compress(const INTEGER *src, u8 *dest, const Statistics *stats,
                   const u32 total_size, const u16 block_size) {
  return compressImpl(src, dest, stats, total_size, block_size,
                      BitPacking::packFixed);
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

    FOR::decompressImpl(dest, read_ptr, size, slot.reference, slot.pack_size);

    dest += size;
  }
}
//---------------------------------------------------------------------------
CompressionSchemeType FORn::getType() { return CompressionSchemeType::kFORn; }
//---------------------------------------------------------------------------
u32 FORn::compressImpl(const INTEGER *src, u8 *dest, const Statistics *stats,
                       const u32 total_size, const u16 block_size,
                       u8 (*bitpack_func)(const INTEGER *, u8 *, const u32,
                                          INTEGER)) {
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

    u8 pack_size = FOR::compressImpl(src, data_ptr + offset, &stats[block_i],
                                     size, bitpack_func);

    slot.reference = stats[block_i].min;
    slot.pack_size = pack_size;
    slot.offset = offset;

    offset += std::ceil(static_cast<double>(pack_size * size) / 8);
    src += size;
  }

  return sizeof(FORnLayout) + block_count * sizeof(FORnSlot) + offset;
}
//---------------------------------------------------------------------------
} // namespace compression