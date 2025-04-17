#pragma once
//---------------------------------------------------------------------------
#include <cmath>
//---------------------------------------------------------------------------
#include "schemes/CompressionScheme.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
template <typename DataType, const u16 kBlockSize>
class FORn : public CompressionScheme<DataType> {
public:
  //---------------------------------------------------------------------------
  struct FORnSlot {
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

    u32 data_offset = block_count * FORnSlot::size();
    for (u32 block_i = 0; block_i < block_count; ++block_i) {
      data_ptr = dest + data_offset;

      // Update header
      auto &slot = *reinterpret_cast<FORnSlot *>(header_ptr);
      slot.pack_size = compressDispatch(src, data_ptr, &stats[block_i]);
      slot.reference = stats[block_i].min;
      slot.offset = data_offset;

      // Update iterators
      data_offset +=
          std::ceil(static_cast<double>(slot.pack_size * kBlockSize) / 8);
      src += kBlockSize;
      header_ptr += FORnSlot::size();
    }

    u32 header_size = header_ptr - dest;
    u32 payload_size = data_offset - header_size;
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
    const u8 *header_ptr = src + block_offset * FORnSlot::size();
    const u8 *data_ptr = src;

    for (u32 block_i = 0; block_i < block_count; ++block_i) {

      // Read header
      auto &slot = *reinterpret_cast<const FORnSlot *>(header_ptr);
      data_ptr = src + slot.offset;

      // Decompress payload
      decompressDispatch(dest, data_ptr, slot.reference, slot.pack_size);

      // Update iterators
      dest += kBlockSize;
      header_ptr += FORnSlot::size();
    }
  }
  //---------------------------------------------------------------------------
  bool isPartitioningScheme() override { return true; }

private:
  //---------------------------------------------------------------------------
  u8 compressDispatch(const DataType *src, u8 *dest,
                      const Statistics<DataType> *stats) {
    if (stats->diff_bits <= 8) {
      compressImpl<u8>(src, dest, stats);
      return 8;
    } else if (stats->diff_bits <= 16) {
      compressImpl<u16>(src, dest, stats);
      return 16;
    } else if (stats->diff_bits <= 32) {
      compressImpl<u32>(src, dest, stats);
      return 32;
    } else {
      compressImpl<u64>(src, dest, stats);
      return 64;
    }
  }
  //---------------------------------------------------------------------------
  void decompressDispatch(DataType *dest, const u8 *src, const u32 reference,
                          const u8 pack_size) {
    switch (pack_size) {
    case 8:
      decompressImpl<u8>(dest, src, reference);
      return;
    case 16:
      decompressImpl<u16>(dest, src, reference);
      return;
    case 32:
      decompressImpl<u32>(dest, src, reference);
      return;
    case 64:
      decompressImpl<u64>(dest, src, reference);
      return;
    default:
      return;
    }
  }
  //---------------------------------------------------------------------------
  template <typename T>
  void compressImpl(const DataType *src, u8 *dest,
                    const Statistics<DataType> *stats) {
    auto data = reinterpret_cast<T *>(dest);
    for (u32 i = 0; i < kBlockSize; ++i) {
      data[i] = static_cast<T>(src[i] - stats->min);
    }
  }
  //---------------------------------------------------------------------------
  template <typename T>
  void decompressImpl(DataType *dest, const u8 *src, const u32 reference) {
    const auto &data = reinterpret_cast<const T *>(src);
    for (u32 i = 0; i < kBlockSize; ++i) {
      dest[i] = data[i] + reference;
    }
  }
};
//---------------------------------------------------------------------------
} // namespace compression
