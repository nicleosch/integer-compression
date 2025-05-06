#pragma once
//---------------------------------------------------------------------------
#include "common/Utils.hpp"
#include "schemes/CompressionScheme.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
template <typename DataType> class Delta : public CompressionScheme<DataType> {
public:
  //---------------------------------------------------------------------------
  struct __attribute__((packed)) DeltaLayout {
    /// The first value of the sequence.
    DataType base;
    /// The number of bits required to store the deltas.
    u8 pack_size;
    /// The compressed data.
    u8 data[];
  };
  //---------------------------------------------------------------------------
  CompressionDetails compress(const DataType *src, const u32 size, u8 *dest,
                              const Statistics<DataType> *stats) override {
    auto &layout = *reinterpret_cast<DeltaLayout *>(dest);
    layout.pack_size = compressDispatch(src, size, layout.data, stats);
    layout.base = src[0];

    u64 header_size = offsetof(DeltaLayout, data);
    u64 payload_size = layout.pack_size / 8 * size;
    return {header_size, payload_size};
  }
  //---------------------------------------------------------------------------
  void decompress(DataType *dest, const u32 size, const u8 *src) override {
    const auto &layout = *reinterpret_cast<const DeltaLayout *>(src);
    decompressDispatch(dest, size, layout);
  }
  //---------------------------------------------------------------------------
  bool isPartitioningScheme() override { return false; }

private:
  //---------------------------------------------------------------------------
  u8 compressDispatch(const DataType *src, const u32 size, u8 *dest,
                      const Statistics<DataType> *stats) {
    if (stats->delta_bits <= 8) {
      compressImpl<u8>(src, size, dest);
      return 8;
    } else if (stats->delta_bits <= 16) {
      compressImpl<u16>(src, size, dest);
      return 16;
    } else if (stats->delta_bits <= 32) {
      compressImpl<u32>(src, size, dest);
      return 32;
    } else {
      compressImpl<u64>(src, size, dest);
      return 64;
    }
  }
  //---------------------------------------------------------------------------
  void decompressDispatch(DataType *dest, const u32 size,
                          const DeltaLayout &layout) {
    switch (layout.pack_size) {
    case 8:
      decompressImpl<u8>(dest, size, layout.data, layout.base);
      return;
    case 16:
      decompressImpl<u16>(dest, size, layout.data, layout.base);
      return;
    case 32:
      decompressImpl<u32>(dest, size, layout.data, layout.base);
      return;
    case 64:
      decompressImpl<u64>(dest, size, layout.data, layout.base);
      return;
    default:
      return;
    }
  }
  //---------------------------------------------------------------------------
  template <typename TargetType>
  void compressImpl(const DataType *src, const u32 size, u8 *dest) {
    utils::unalignedStore<TargetType>(dest, 0);
    for (u32 i = 1; i < size; ++i) {
      utils::unalignedStore<TargetType>(
          dest + i * sizeof(TargetType),
          static_cast<TargetType>(src[i] - src[i - 1]));
    }
  }
  //---------------------------------------------------------------------------
  template <typename TargetType>
  void decompressImpl(DataType *dest, const u32 size, const u8 *src,
                      const DataType base) {
    dest[0] = base;

    for (u32 i = 1; i < size; ++i) {
      dest[i] = utils::unalignedLoad<TargetType>(src + i * sizeof(TargetType)) +
                dest[i - 1];
    }
  }
};
//---------------------------------------------------------------------------
} // namespace compression