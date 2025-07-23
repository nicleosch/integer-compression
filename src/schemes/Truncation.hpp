#pragma once
//---------------------------------------------------------------------------
#include "schemes/CompressionScheme.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
template <typename DataType, u16 kBlockSize>
class Truncation : public CompressionScheme<DataType> {
public:
  //---------------------------------------------------------------------------
  struct Header {
    /// The number of bytes used to store an integer in corresponding frame.
    u32 cbytes;
  };
  //---------------------------------------------------------------------------
  CompressionDetails compress(const DataType *src, const u32 size, u8 *dest,
                              const Statistics<DataType> *stats) override {
    const u32 block_count = size / kBlockSize;
    //---------------------------------------------------------------------------
    auto &header = *reinterpret_cast<Header *>(dest);
    if (stats->diff_bits <= 8) {
      header.cbytes = 1;
    } else if (stats->diff_bits <= 16) {
      header.cbytes = 2;
    } else if (stats->diff_bits <= 32) {
      header.cbytes = 4;
    } else {
      header.cbytes = 8;
    }
    dest += sizeof(header);
    //---------------------------------------------------------------------------
    for (u32 b = 0; b < block_count; ++b) {
      compressDispatch(src, dest, header.cbytes);
      dest += header.cbytes * kBlockSize;
      src += kBlockSize;
    }
    //---------------------------------------------------------------------------
    return {4, header.cbytes * size};
  }
  //---------------------------------------------------------------------------
  void decompress(DataType *dest, const u32 size, const u8 *src) override {
    const u32 block_count = size / kBlockSize;
    //---------------------------------------------------------------------------
    const auto &header = *reinterpret_cast<const Header *>(src);
    src += sizeof(header);
    //---------------------------------------------------------------------------
    for (u32 b = 0; b < block_count; ++b) {
      decompressDispatch(dest, src, header.cbytes);
      dest += kBlockSize;
      src += header.cbytes * kBlockSize;
    }
  }
  //---------------------------------------------------------------------------
  bool isPartitioningScheme() override { return false; }

private:
  //---------------------------------------------------------------------------
  void compressDispatch(const DataType *src, u8 *dest, const u32 cbytes) {
    switch (cbytes) {
    case 1:
      compressImpl<u8>(src, dest);
      break;
    case 2:
      compressImpl<u8>(src, dest);
      break;
    case 4:
      compressImpl<u8>(src, dest);
      break;
    case 8:
      compressImpl<u8>(src, dest);
      break;
    default:
      std::runtime_error("Truncation Compression Error: Size \"" +
                         std::to_string(cbytes) + "\" not supported.");
    }
  }
  //---------------------------------------------------------------------------
  void decompressDispatch(DataType *dest, const u8 *src, const u32 cbytes) {
    switch (cbytes) {
    case 1:
      decompressImpl<u8>(dest, src);
      return;
    case 2:
      decompressImpl<u16>(dest, src);
      return;
    case 4:
      decompressImpl<u32>(dest, src);
      return;
    case 8:
      decompressImpl<u64>(dest, src);
      return;
    default:
      std::runtime_error("Truncation Decompression Error: Size \"" +
                         std::to_string(cbytes) + "\" not supported.");
    }
  }
  //---------------------------------------------------------------------------
  template <typename T> void compressImpl(const DataType *src, u8 *dest) {
    auto data = reinterpret_cast<T *>(dest);
    for (u32 i = 0; i < kBlockSize; ++i) {
      data[i] = static_cast<T>(src[i]);
    }
  }
  //---------------------------------------------------------------------------
  template <typename T> void decompressImpl(DataType *dest, const u8 *src) {
    const auto &data = reinterpret_cast<const T *>(src);
    for (u32 i = 0; i < kBlockSize; ++i) {
      dest[i] = data[i];
    }
  }
};
//---------------------------------------------------------------------------
} // namespace compression
