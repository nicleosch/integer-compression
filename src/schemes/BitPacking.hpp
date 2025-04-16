#pragma once
//---------------------------------------------------------------------------
#include "bitpacking/BitPacking.hpp"
#include "schemes/CompressionScheme.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
/// This class implements bit-packing. For implementation, bit-packing operates
/// on smaller data blocks, with the block size specified as a template
/// parameter.
/// @tparam kBlockSize: The size of a block.
template <typename DataType, const u16 kBlockSize>
class BitPacking : public CompressionScheme<DataType> {
public:
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
  u32 compress(const DataType *src, const u32 size, u8 *dest,
               const Statistics<DataType> *stats) override {
    const u32 block_count = size / kBlockSize;

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
      data_offset +=
          bitpacking::pack<DataType, kBlockSize>(src, data_ptr, slot.pack_size);

      // Update iterators
      src += kBlockSize;
      header_ptr += BitPackingSlot::size();
    }

    return data_offset;
  }
  //---------------------------------------------------------------------------
  void decompress(DataType *dest, const u32 size, const u8 *src) override {
    decompress(dest, size, src, 0);
  }
  //---------------------------------------------------------------------------
  void decompress(DataType *dest, const u32 size, const u8 *src,
                  const u32 block_offset) {
    const u32 block_count = size / kBlockSize;

    // Layout: HEADER | COMPRESSED DATA
    const u8 *header_ptr = src + block_offset * BitPackingSlot::size();
    const u8 *data_ptr = src;

    for (u32 block_i = 0; block_i < block_count; ++block_i) {

      // Read header
      auto &slot = *reinterpret_cast<const BitPackingSlot *>(header_ptr);
      data_ptr = src + slot.offset;

      // Decompress payload
      bitpacking::unpack<DataType, kBlockSize>(dest, data_ptr, slot.pack_size);

      // Update iterators
      dest += kBlockSize;
      header_ptr += BitPackingSlot::size();
    }
  }
  //---------------------------------------------------------------------------
  bool isPartitioningScheme() override { return true; }
};
//---------------------------------------------------------------------------
} // namespace compression