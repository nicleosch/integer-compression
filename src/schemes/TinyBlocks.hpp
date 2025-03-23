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
struct TinyBlocksSlot {
  /// The reference of the corresponding frame.
  INTEGER reference;
  /// The offset into the data array.
  u32 offset;
  /// The number of bits used to store an integer in corresponding frame.
  u8 pack_size;
  /// The unpadded size of a slot.
  static u8 size() {
    return sizeof(reference) + sizeof(offset) + sizeof(pack_size);
  }
};
//---------------------------------------------------------------------------
class TinyBlocks {
public:
  //---------------------------------------------------------------------------
  template <const u16 kBlockSize>
  u32 compress(const INTEGER *src, const u32 total_size, u8 *dest,
               const Statistics *stats) {
    const u32 block_count = total_size / kBlockSize;

    // Layout: HEADER [kBlockSize] | COMPRESSED DATA
    u8 *header_ptr = dest;
    u8 *data_ptr = dest;

    u32 data_offset = block_count * TinyBlocksSlot::size();
    for (u32 block_i = 0; block_i < block_count; ++block_i) {
      data_ptr = dest + data_offset;

      // Update header
      auto &slot = *reinterpret_cast<TinyBlocksSlot *>(header_ptr);
      slot.reference = stats[block_i].min;
      slot.pack_size = stats[block_i].required_bits;
      slot.offset = data_offset;

      // Compress block
      data_offset += compressDispatch<kBlockSize>(src, data_ptr, slot);

      // Update iterators
      src += kBlockSize;
      header_ptr += TinyBlocksSlot::size();
    }

    return data_offset;
  }
  //---------------------------------------------------------------------------
  template <const u16 kBlockSize>
  void decompress(INTEGER *dest, const u32 total_size, const u8 *src) {
    decompress<kBlockSize>(dest, total_size, src, 0);
  }
  //---------------------------------------------------------------------------
  template <const u16 kBlockSize>
  void decompress(INTEGER *dest, const u32 total_size, const u8 *src,
                  const u32 block_offset) {
    const u32 block_count = total_size / kBlockSize;

    // Layout: HEADER [kBlockSize] | COMPRESSED DATA
    const u8 *header_ptr = src + block_offset * TinyBlocksSlot::size();
    const u8 *data_ptr = src;

    for (u32 block_i = 0; block_i < block_count; ++block_i) {

      // Read header
      auto &slot = *reinterpret_cast<const TinyBlocksSlot *>(header_ptr);
      data_ptr = src + slot.offset;

      // Decompress payload
      decompressDispatch<kBlockSize>(dest, data_ptr, slot);

      // Update iterators
      dest += kBlockSize;
      header_ptr += 9;
    }
  }

private:
  //---------------------------------------------------------------------------
  template <const u16 kBlockSize>
  u16 compressDispatch(const INTEGER *src, u8 *dest,
                       const TinyBlocksSlot &slot) {
    switch (slot.pack_size) {
    case 65: // OneValue
      return 0;
    default: // Regular Bit-Size
      compressImpl<kBlockSize>(src, dest, slot);
      return std::ceil(static_cast<double>(slot.pack_size * kBlockSize) / 8);
    }
  }
  //---------------------------------------------------------------------------
  template <const u16 kBlockSize>
  void decompressDispatch(INTEGER *dest, const u8 *src,
                          const TinyBlocksSlot &slot) {
    switch (slot.pack_size) {
    case 65: // OneValue
      broadcast<kBlockSize>(dest, slot.reference);
      return;
    default: // Regular Bit-Size
      decompressImpl<kBlockSize>(dest, src, slot);
      return;
    }
  }
  //---------------------------------------------------------------------------
  template <const u16 kLength>
  void compressImpl(const INTEGER *src, u8 *dest, const TinyBlocksSlot &slot) {
    // Normalize
    vector<INTEGER> normalized(kLength);
    for (u32 i = 0; i < kLength; ++i) {
      normalized[i] = src[i] - slot.reference;
    }

    // Compress
    bitpacking::pack(normalized.data(), dest, kLength, slot.pack_size);
  }
  //---------------------------------------------------------------------------
  template <const u16 kLength>
  void decompressImpl(INTEGER *dest, const u8 *src,
                      const TinyBlocksSlot &slot) {
    // Decompress
    bitpacking::unpack(dest, src, kLength, slot.pack_size);

    // Denormalize
    for (u32 i = 0; i < kLength; ++i) {
      dest[i] += slot.reference;
    }
  }
  //---------------------------------------------------------------------------
  template <const u16 kLength> void broadcast(INTEGER *dest, INTEGER value) {
    for (u16 i = 0; i < kLength; ++i) {
      dest[i] = value;
    }
  }
};
//---------------------------------------------------------------------------
} // namespace compression