#pragma once
//---------------------------------------------------------------------------
#include <cmath>
//---------------------------------------------------------------------------
#include "common/Units.hpp"
#include "statistics/Statistics.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
struct FORnSlot {
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
class FORn {
public:
  //---------------------------------------------------------------------------
  template <const u16 kBlockSize>
  u32 compress(const INTEGER *src, const u32 total_size, u8 *dest,
               const Statistics *stats) {
    const u32 block_count = total_size / kBlockSize;

    // Layout: HEADER [kBlockSize] | COMPRESSED DATA
    u8 *header_ptr = dest;
    u8 *data_ptr = dest;

    u32 data_offset = block_count * FORnSlot::size();
    for (u32 block_i = 0; block_i < block_count; ++block_i) {
      data_ptr = dest + data_offset;

      // Update header
      auto &slot = *reinterpret_cast<FORnSlot *>(header_ptr);
      slot.pack_size =
          compressDispatch<kBlockSize>(src, data_ptr, &stats[block_i]);
      slot.reference = stats[block_i].min;
      slot.offset = data_offset;

      // Update iterators
      data_offset +=
          std::ceil(static_cast<double>(slot.pack_size * kBlockSize) / 8);
      src += kBlockSize;
      header_ptr += FORnSlot::size();
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
    const u8 *header_ptr = src + block_offset * FORnSlot::size();
    const u8 *data_ptr = src;

    for (u32 block_i = 0; block_i < block_count; ++block_i) {

      // Read header
      auto &slot = *reinterpret_cast<const FORnSlot *>(header_ptr);
      data_ptr = src + slot.offset;

      // Decompress payload
      decompressDispatch<kBlockSize>(dest, data_ptr, slot.reference,
                                     slot.pack_size);

      // Update iterators
      dest += kBlockSize;
      header_ptr += 9;
    }
  }

private:
  //---------------------------------------------------------------------------
  template <const u16 kLength>
  u8 compressDispatch(const INTEGER *src, u8 *dest, const Statistics *stats) {
    if (stats->required_bits <= 8) {
      compressImpl<u8, kLength>(src, dest, stats);
      return 8;
    } else if (stats->required_bits <= 16) {
      compressImpl<u16, kLength>(src, dest, stats);
      return 16;
    } else {
      compressImpl<u32, kLength>(src, dest, stats);
      return 32;
    }
  }
  //---------------------------------------------------------------------------
  template <const u16 kLength>
  void decompressDispatch(INTEGER *dest, const u8 *src, const u32 reference,
                          const u8 pack_size) {
    switch (pack_size) {
    case 8:
      decompressImpl<u8, kLength>(dest, src, reference);
      return;
    case 16:
      decompressImpl<u16, kLength>(dest, src, reference);
      return;
    case 32:
      decompressImpl<u32, kLength>(dest, src, reference);
      return;
    default:
      break;
    }
  }
  //---------------------------------------------------------------------------
  template <typename T, const u16 kLength>
  void compressImpl(const INTEGER *src, u8 *dest, const Statistics *stats) {
    auto data = reinterpret_cast<T *>(dest);
    for (u32 i = 0; i < kLength; ++i) {
      data[i] = static_cast<T>(src[i] - stats->min);
    }
  }
  //---------------------------------------------------------------------------
  template <typename T, const u16 kLength>
  void decompressImpl(INTEGER *dest, const u8 *src, const u32 reference) {
    const auto &data = reinterpret_cast<const T *>(src);
    for (u32 i = 0; i < kLength; ++i) {
      dest[i] = data[i] + reference;
    }
  }
};
//---------------------------------------------------------------------------
} // namespace compression
