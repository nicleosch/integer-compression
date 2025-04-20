#pragma once
//---------------------------------------------------------------------------
#include <cmath>
//---------------------------------------------------------------------------
#include "bitpacking/BitPacking.hpp"
#include "schemes/CompressionScheme.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
template <typename DataType, const u16 kBlockSize>
class TinyBlocks : public CompressionScheme<DataType> {
public:
  //---------------------------------------------------------------------------
  struct TinyBlocksSlot {
    /// The reference of the corresponding frame.
    DataType reference;
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
  CompressionDetails compress(const DataType *src, const u32 size, u8 *dest,
                              const Statistics<DataType> *stats) override {
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

    u64 header_size = header_ptr - dest;
    u64 payload_size = data_offset - header_size;
    return {header_size, payload_size};
  }
  //---------------------------------------------------------------------------
  void decompress(DataType *dest, const u32 size, const u8 *src) override {
    decompress(dest, size, src, 0);
  }
  //---------------------------------------------------------------------------
  void decompress(DataType *dest, const u32 size, const u8 *src,
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
      header_ptr += TinyBlocksSlot::size();
    }
  }
  //---------------------------------------------------------------------------
  bool isPartitioningScheme() override { return true; }

private:
  //---------------------------------------------------------------------------
  u16 compressDispatch(const DataType *src, u8 *dest,
                       const Statistics<DataType> &stats,
                       TinyBlocksSlot &slot) {
    switch (slot.pack_size) {
    case 0: // OneValue
      return 0;
    case 65: // Monotonically Increasing
      slot.pack_size = (slot.pack_size & (step_mask + 1)) | stats.step_size;
      return 0;
    default: // Regular Bit-Size
      return compressImpl(src, dest, slot);
    }
  }
  //---------------------------------------------------------------------------
  void decompressDispatch(DataType *dest, const u8 *src,
                          const TinyBlocksSlot &slot) {
    if (slot.pack_size == 0) // OneValue
      broadcast(dest, slot.reference);
    else if (slot.pack_size >= 65) // Monotonically Increasing
      decompressMonoInc(dest, slot);
    else // Regular Bit-Size
      decompressImpl(dest, src, slot);
  }
  //---------------------------------------------------------------------------
  u32 compressImpl(const DataType *src, u8 *dest, const TinyBlocksSlot &slot) {
    // Normalize
    vector<DataType> normalized(kBlockSize);
    for (u32 i = 0; i < kBlockSize; ++i) {
      normalized[i] = src[i] - slot.reference;
    }

    // Compress
    return bitpacking::pack<DataType, kBlockSize>(normalized.data(), dest,
                                                  slot.pack_size);
  }
  //---------------------------------------------------------------------------
  void decompressImpl(DataType *dest, const u8 *src,
                      const TinyBlocksSlot &slot) {
    // Decompress
    bitpacking::unpack<DataType, kBlockSize>(dest, src, slot.pack_size);

    // Denormalize
    for (u32 i = 0; i < kBlockSize; ++i) {
      dest[i] += slot.reference;
    }
  }
  //---------------------------------------------------------------------------
  void decompressMonoInc(DataType *dest, const TinyBlocksSlot &slot) {
    u8 step_size = slot.pack_size & step_mask;

    dest[0] = slot.reference;
    for (u16 i = 1; i < kBlockSize; ++i) {
      dest[i] = dest[i - 1] + step_size;
    }
  }
  //---------------------------------------------------------------------------
  void broadcast(DataType *dest, DataType value) {
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