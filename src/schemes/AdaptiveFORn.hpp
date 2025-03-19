#pragma once
//---------------------------------------------------------------------------
#include <cmath>
//---------------------------------------------------------------------------
#include "bitpacking/BitPacking.hpp"
#include "common/Units.hpp"
#include "statistics/Statistics.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
struct AdaptiveFORnSlot {
  INTEGER reference;
  u32 offset;
  u8 pack_size;
};
//---------------------------------------------------------------------------
struct AdaptiveFORnLayout {
  u32 data_offset;
  // SLOTS [] | COMPRESSED DATA []
  u8 data[];
};
//---------------------------------------------------------------------------
class AdaptiveFORn {
public:
  //---------------------------------------------------------------------------
  template <const u16 kBlockSize>
  u32 compress(const INTEGER *src, const u32 total_size, u8 *dest,
               const Statistics *stats) {
    auto &layout = *reinterpret_cast<AdaptiveFORnLayout *>(dest);
    auto header = reinterpret_cast<AdaptiveFORnSlot *>(layout.data);

    const u32 block_count = total_size / kBlockSize;

    layout.data_offset = block_count * sizeof(AdaptiveFORnSlot);

    // Compress data
    u8 *data_ptr = layout.data + layout.data_offset;
    u32 offset = 0;
    for (u32 block_i = 0; block_i < block_count; ++block_i) {
      auto &slot = header[block_i];

      slot.reference = stats[block_i].min;
      slot.pack_size = stats[block_i].required_bits;
      slot.offset = offset;

      compressImpl<kBlockSize>(src, data_ptr + offset, slot);

      offset += std::ceil(static_cast<double>(slot.pack_size * kBlockSize) / 8);
      src += kBlockSize;
    }

    return sizeof(AdaptiveFORnLayout) + block_count * sizeof(AdaptiveFORnSlot) +
           offset;
  }
  //---------------------------------------------------------------------------
  template <const u16 kBlockSize>
  void decompress(INTEGER *dest, const u32 total_size, const u8 *src) {
    const auto &layout = *reinterpret_cast<const AdaptiveFORnLayout *>(src);
    const auto &header =
        reinterpret_cast<const AdaptiveFORnSlot *>(layout.data);

    const u32 block_count = total_size / kBlockSize;

    const u8 *read_ptr;
    for (u32 block_i = 0; block_i < block_count; ++block_i) {
      auto &slot = header[block_i];

      read_ptr = layout.data + layout.data_offset + slot.offset;
      decompressImpl<kBlockSize>(dest, read_ptr, slot);

      dest += kBlockSize;
    }
  }

private:
  //---------------------------------------------------------------------------
  template <const u16 kLength>
  void compressImpl(const INTEGER *src, u8 *dest,
                    const AdaptiveFORnSlot &slot) {
    // Normalize
    vector<INTEGER> normalized;
    for (u32 i = 0; i < kLength; ++i) {
      normalized.push_back(src[i] - slot.reference);
    }

    // Compress
    bitpacking::pack(normalized.data(), dest, kLength, slot.pack_size);
  }
  //---------------------------------------------------------------------------
  template <const u16 kLength>
  void decompressImpl(INTEGER *dest, const u8 *src,
                      const AdaptiveFORnSlot &slot) {
    // Decompress
    bitpacking::unpack(dest, src, kLength, slot.pack_size);

    // Denormalize
    for (u32 i = 0; i < kLength; ++i) {
      dest[i] += slot.reference;
    }
  }
};
//---------------------------------------------------------------------------
} // namespace compression