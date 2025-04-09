#pragma once
//---------------------------------------------------------------------------
#include <cmath>
//---------------------------------------------------------------------------
#include "bitpacking/BitPacking.hpp"
#include "common/Types.hpp"
#include "statistics/Statistics.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
struct BitPackingSlot {
  /// The offset into the data array.
  u32 offset;
  /// The number of bits used to store an integer in corresponding frame.
  u8 pack_size;
  /// The unpadded size of a slot.
  static u8 size() { return sizeof(offset) + sizeof(pack_size); }
};
//---------------------------------------------------------------------------
class BitPacking {
public:
  //---------------------------------------------------------------------------
  template <const u16 kBlockSize>
  u32 compress(const INTEGER *src, const u32 total_size, u8 *dest,
               const Statistics *stats) {
    const u32 block_count = total_size / kBlockSize;

    // Layout: HEADER | COMPRESSED DATA
    u8 *header_ptr = dest;
    u8 *data_ptr = dest;

    u32 data_offset = block_count * BitPackingSlot::size();
    for (u32 block_i = 0; block_i < block_count; ++block_i) {
      data_ptr = dest + data_offset;

      // Update header
      auto &slot = *reinterpret_cast<BitPackingSlot *>(header_ptr);
      slot.pack_size = stats[block_i].max_bits;
      slot.offset = data_offset;

      // Compress block
      bitpacking::pack(src, data_ptr, kBlockSize, slot.pack_size);
      data_offset +=
          std::ceil(static_cast<double>(slot.pack_size * kBlockSize) / 8);

      // Update iterators
      src += kBlockSize;
      header_ptr += BitPackingSlot::size();
    }

    return data_offset;
  }
  //---------------------------------------------------------------------------
  template <const u16 kBlockSize>
  void decompress(INTEGER *dest, const u32 total_size, const u8 *src) {
    const u32 block_count = total_size / kBlockSize;

    // Layout: HEADER | COMPRESSED DATA
    const u8 *header_ptr = src;
    const u8 *data_ptr = src;

    for (u32 block_i = 0; block_i < block_count; ++block_i) {

      // Read header
      auto &slot = *reinterpret_cast<const BitPackingSlot *>(header_ptr);
      data_ptr = src + slot.offset;

      // Decompress payload
      bitpacking::unpack(dest, data_ptr, kBlockSize, slot.pack_size);

      // Update iterators
      dest += kBlockSize;
      header_ptr += BitPackingSlot::size();
    }
  }
};
//---------------------------------------------------------------------------
} // namespace compression