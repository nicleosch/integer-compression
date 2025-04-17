#pragma once
//---------------------------------------------------------------------------
#include "schemes/CompressionScheme.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
template <typename DataType> class FOR : public CompressionScheme<DataType> {
public:
  //---------------------------------------------------------------------------
  struct FORLayout {
    /// The reference of the corresponding frame.
    DataType reference;
    /// The number of bits used to store an integer in the frame.
    u8 pack_size;
    /// The compressed data.
    u8 data[];
  };
  //---------------------------------------------------------------------------
  CompressionDetails compress(const DataType *src, const u32 size, u8 *dest,
                              const Statistics<DataType> *stats) override {
    auto &layout = *reinterpret_cast<FORLayout *>(dest);
    layout.reference = stats->min;
    layout.pack_size = compressDispatch(src, size, layout.data, stats);

    u32 header_size = offsetof(FORLayout, data);
    u32 payload_size = layout.pack_size / 8 * size;
    return {header_size, payload_size};
  }
  //---------------------------------------------------------------------------
  void decompress(DataType *dest, const u32 size, const u8 *src) override {
    decompress(dest, size, src, 0);
  }
  //---------------------------------------------------------------------------
  void decompress(DataType *dest, const u32 size, const u8 *src,
                  const u32 offset) {
    const auto &layout = *reinterpret_cast<const FORLayout *>(src);
    decompressDispatch(dest, size,
                       layout.data + offset * (layout.pack_size / 8),
                       layout.reference, layout.pack_size);
  }
  //---------------------------------------------------------------------------
  bool isPartitioningScheme() override { return false; }

private:
  //---------------------------------------------------------------------------
  u8 compressDispatch(const DataType *src, const u32 size, u8 *dest,
                      const Statistics<DataType> *stats) {
    if (stats->diff_bits <= 8) {
      compressImpl<u8>(src, size, dest, stats);
      return 8;
    } else if (stats->diff_bits <= 16) {
      compressImpl<u16>(src, size, dest, stats);
      return 16;
    } else {
      compressImpl<u32>(src, size, dest, stats);
      return 32;
    }
  }
  //---------------------------------------------------------------------------
  void decompressDispatch(DataType *dest, const u32 size, const u8 *src,
                          const u32 reference, const u8 pack_size) {
    switch (pack_size) {
    case 8:
      decompressImpl<u8>(dest, size, src, reference);
      return;
    case 16:
      decompressImpl<u16>(dest, size, src, reference);
      return;
    case 32:
      decompressImpl<u32>(dest, size, src, reference);
      return;
    default:
      break;
    }
  }
  //---------------------------------------------------------------------------
  template <typename TargetType>
  void compressImpl(const DataType *src, const u32 size, u8 *dest,
                    const Statistics<DataType> *stats) {
    auto data = reinterpret_cast<TargetType *>(dest);
    for (u32 i = 0; i < size; ++i) {
      data[i] = static_cast<TargetType>(src[i] - stats->min);
    }
  }
  //---------------------------------------------------------------------------
  template <typename TargetType>
  void decompressImpl(DataType *dest, const u32 size, const u8 *src,
                      const u32 reference) {
    const auto &data = reinterpret_cast<const TargetType *>(src);
    for (u32 i = 0; i < size; ++i) {
      dest[i] = data[i] + reference;
    }
  }
};
//---------------------------------------------------------------------------
} // namespace compression