#pragma once
//---------------------------------------------------------------------------
#include "schemes/CompressionScheme.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
struct RLELayout {
  u32 value_offset;
  u8 data[];
};
//---------------------------------------------------------------------------
class RLE : public CompressionScheme {
public:
  //---------------------------------------------------------------------------
  u32 compress(const INTEGER *src, const u32 size, u8 *dest,
               const Statistics *stats) override;
  //---------------------------------------------------------------------------
  void decompress(INTEGER *dest, const u32 size, const u8 *src) override;
  //---------------------------------------------------------------------------
  bool isPartitioningScheme() override;
};
//---------------------------------------------------------------------------
} // namespace compression