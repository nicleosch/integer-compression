//---------------------------------------------------------------------------
// Inspired by https://github.com/maxi-k/btrblocks
//---------------------------------------------------------------------------
#pragma once
//---------------------------------------------------------------------------
#include "schemes/CompressionScheme.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
struct FORLayout {
  INTEGER reference;
  u8 pack_size;
  u8 data[];
};
//---------------------------------------------------------------------------
class FOR : public CompressionScheme {
public:
  u32 compress(const INTEGER *src, u8 *dest, const Statistics *stats,
               const u32 total_size, const u16 block_size) override;
  //---------------------------------------------------------------------------
  void decompress(INTEGER *dest, const u8 *src, const u32 total_size,
                  const u16 block_size) override;
  //---------------------------------------------------------------------------
  CompressionSchemeType getType() override;

protected:
  u8 compressDispatch(const INTEGER *src, u8 *dest, const Statistics *stats,
                      const u32 size) {
    if (stats->required_bits <= 8) {
      compressImpl<u8>(src, dest, stats, size);
      return 8;
    } else if (stats->required_bits <= 16) {
      compressImpl<u16>(src, dest, stats, size);
      return 16;
    } else {
      compressImpl<u32>(src, dest, stats, size);
      return 32;
    }
  }
  //---------------------------------------------------------------------------
  void decompressDispatch(INTEGER *dest, const u8 *src, const u32 reference,
                          const u32 size, const u8 pack_size) {
    switch (pack_size) {
    case 8:
      decompressImpl<u8>(dest, src, reference, size);
      return;
    case 16:
      decompressImpl<u16>(dest, src, reference, size);
      return;
    case 32:
      decompressImpl<u32>(dest, src, reference, size);
      return;
    default:
      break;
    }
  }
  //---------------------------------------------------------------------------
  template <typename T>
  void compressImpl(const INTEGER *src, u8 *dest, const Statistics *stats,
                    const u32 size) {
    auto data = reinterpret_cast<T *>(dest);
    for (u32 i = 0; i < size; ++i) {
      data[i] = static_cast<T>(src[i] - stats->min);
    }
  }
  //---------------------------------------------------------------------------
  template <typename T>
  void decompressImpl(INTEGER *dest, const u8 *src, const u32 reference,
                      const u32 size) {
    const auto &data = reinterpret_cast<const T *>(src);
    for (u32 i = 0; i < size; ++i) {
      dest[i] = data[i] + reference;
    }
  }
};
//---------------------------------------------------------------------------
} // namespace compression