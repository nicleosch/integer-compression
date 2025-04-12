#pragma once
//---------------------------------------------------------------------------
#include <cmath>
//---------------------------------------------------------------------------
#include "bitpacking/BitPacking.hpp"
#include "schemes/CompressionScheme.hpp"
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
template <const u16 kBlockSize> class TinyBlocks : public CompressionScheme {
public:
  //---------------------------------------------------------------------------
  u32 compress(const INTEGER *src, const u32 size, u8 *dest,
               const Statistics *stats) override {
    const u32 block_count = size / kBlockSize;

    // Layout: HEADER [kBlockSize] | COMPRESSED DATA
    u8 *header_ptr = dest;
    u8 *data_ptr = dest;

    u32 data_offset = block_count * TinyBlocksSlot::size();
    for (u32 block_i = 0; block_i < block_count; ++block_i) {
      data_ptr = dest + data_offset;

      // Update header
      auto &slot = *reinterpret_cast<TinyBlocksSlot *>(header_ptr);
      slot.reference = stats[block_i].min;
      slot.pack_size = stats[block_i].diff_bits;
      slot.offset = data_offset;

      // Compress block
      data_offset += compressDispatch(src, data_ptr, stats[block_i], slot);

      // Update iterators
      src += kBlockSize;
      header_ptr += TinyBlocksSlot::size();
    }

    return data_offset;
  }
  //---------------------------------------------------------------------------
  void decompress(INTEGER *dest, const u32 size, const u8 *src) override {
    decompress(dest, size, src, 0);
  }
  //---------------------------------------------------------------------------
  void decompress(INTEGER *dest, const u32 size, const u8 *src,
                  const u32 block_offset) {
    const u32 block_count = size / kBlockSize;

    // Layout: HEADER [kBlockSize] | COMPRESSED DATA
    const u8 *header_ptr = src + block_offset * TinyBlocksSlot::size();
    const u8 *data_ptr = src;

    for (u32 block_i = 0; block_i < block_count; ++block_i) {

      // Read header
      auto &slot = *reinterpret_cast<const TinyBlocksSlot *>(header_ptr);
      data_ptr = src + slot.offset;

      // Decompress payload
      decompressDispatch(dest, data_ptr, slot);

      // Update iterators
      dest += kBlockSize;
      header_ptr += 9;
    }
  }
  //---------------------------------------------------------------------------
  bool isPartitioningScheme() override { return true; }

private:
  //---------------------------------------------------------------------------
  u16 compressDispatch(const INTEGER *src, u8 *dest, const Statistics &stats,
                       TinyBlocksSlot &slot) {
    switch (slot.pack_size) {
    case 0: // OneValue
      return 0;
    case 65: // Monotonically Increasing
      slot.pack_size = (slot.pack_size & (step_mask + 1)) | stats.step_size;
      return 0;
    default: // Regular Bit-Size
      compressImpl(src, dest, slot);
      return std::ceil(static_cast<double>(slot.pack_size * kBlockSize) / 8);
    }
  }
  //---------------------------------------------------------------------------
  void decompressDispatch(INTEGER *dest, const u8 *src,
                          const TinyBlocksSlot &slot) {
    if (slot.pack_size == 0) // OneValue
      broadcast(dest, slot.reference);
    else if (slot.pack_size >= 65) // Monotonically Increasing
      decompressMonoInc(dest, slot);
    else // Regular Bit-Size
      decompressImpl(dest, src, slot);
  }
  //---------------------------------------------------------------------------
  void compressImpl(const INTEGER *src, u8 *dest, const TinyBlocksSlot &slot) {
    // Normalize
    vector<INTEGER> normalized(kBlockSize);
    for (u32 i = 0; i < kBlockSize; ++i) {
      normalized[i] = src[i] - slot.reference;
    }

    // Compress
    bitpacking::pack(normalized.data(), dest, kBlockSize, slot.pack_size);
  }
  //---------------------------------------------------------------------------
  void decompressImpl(INTEGER *dest, const u8 *src,
                      const TinyBlocksSlot &slot) {
    // Decompress
    bitpacking::unpack(dest, src, kBlockSize, slot.pack_size);

    // Denormalize
    for (u32 i = 0; i < kBlockSize; ++i) {
      dest[i] += slot.reference;
    }
  }
  //---------------------------------------------------------------------------
  void decompressMonoInc(INTEGER *dest, const TinyBlocksSlot &slot) {
    u8 step_size = slot.pack_size & step_mask;

    dest[0] = slot.reference;
    for (u16 i = 1; i < kBlockSize; ++i) {
      dest[i] = dest[i - 1] + step_size;
    }
  }
  //---------------------------------------------------------------------------
  void broadcast(INTEGER *dest, INTEGER value) {
    for (u16 i = 0; i < kBlockSize; ++i) {
      dest[i] = value;
    }
  }
  //---------------------------------------------------------------------------
  /// A mask to highlight the step size.
  /// Used for compressing monotonically increasing data.
  u8 step_mask = 0x40 - 1;
};
//---------------------------------------------------------------------------
} // namespace compression