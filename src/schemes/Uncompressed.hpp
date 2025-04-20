#pragma once
//---------------------------------------------------------------------------
#include <cstring>
//---------------------------------------------------------------------------
#include "schemes/CompressionScheme.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
template <typename DataType>
class Uncompressed : public CompressionScheme<DataType> {
public:
  //---------------------------------------------------------------------------
  CompressionDetails compress(const DataType *src, const u32 size, u8 *dest,
                              const Statistics<DataType> *stats) override {
    u64 total_size = size * sizeof(DataType);
    std::memcpy(dest, src, total_size);
    return {0, total_size};
  }
  //---------------------------------------------------------------------------
  void decompress(DataType *dest, const u32 size, const u8 *src) override {
    decompress(dest, size, src, 0);
  }
  //---------------------------------------------------------------------------
  void decompress(DataType *dest, const u32 size, const u8 *src,
                  const u32 block_offset) {
    std::memcpy(dest, src + block_offset * sizeof(DataType),
                size * sizeof(DataType));
  }
  //---------------------------------------------------------------------------
  bool isPartitioningScheme() override { return false; }
};
//---------------------------------------------------------------------------
} // namespace compression