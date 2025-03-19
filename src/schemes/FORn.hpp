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
  INTEGER reference;
  u32 offset;
  u8 pack_size;
};
//---------------------------------------------------------------------------
struct FORnLayout {
  u32 data_offset;
  // SLOTS [] | COMPRESSED DATA []
  u8 data[];
};
//---------------------------------------------------------------------------
class FORn {
public:
  //---------------------------------------------------------------------------
  template <const u16 kBlockSize>
  u32 compress(const INTEGER *src, const u32 total_size, u8 *dest,
               const Statistics *stats) {
    auto &layout = *reinterpret_cast<FORnLayout *>(dest);
    auto header = reinterpret_cast<FORnSlot *>(layout.data);

    const u32 block_count = total_size / kBlockSize;

    layout.data_offset = block_count * sizeof(FORnSlot);

    // Compress data
    u8 *data_ptr = layout.data + layout.data_offset;
    u32 offset = 0;
    for (u32 block_i = 0; block_i < block_count; ++block_i) {
      auto &slot = header[block_i];

      slot.pack_size =
          compressDispatch<kBlockSize>(src, data_ptr + offset, &stats[block_i]);
      slot.reference = stats[block_i].min;
      slot.offset = offset;

      offset += std::ceil(static_cast<double>(slot.pack_size * kBlockSize) / 8);
      src += kBlockSize;
    }

    return sizeof(FORnLayout) + block_count * sizeof(FORnSlot) + offset;
  }
  //---------------------------------------------------------------------------
  template <const u16 kBlockSize>
  void decompress(INTEGER *dest, const u32 total_size, const u8 *src) {
    const auto &layout = *reinterpret_cast<const FORnLayout *>(src);
    const auto &header = reinterpret_cast<const FORnSlot *>(layout.data);

    const u32 block_count = total_size / kBlockSize;

    const u8 *read_ptr;
    for (u32 block_i = 0; block_i < block_count; ++block_i) {
      auto &slot = header[block_i];
      read_ptr = layout.data + layout.data_offset + slot.offset;

      decompressDispatch<kBlockSize>(dest, read_ptr, slot.reference,
                                     slot.pack_size);

      dest += kBlockSize;
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
