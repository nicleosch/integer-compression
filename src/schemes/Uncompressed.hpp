#pragma once
//---------------------------------------------------------------------------
#include "schemes/CompressionScheme.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
class Uncompressed : public CompressionScheme {
public:
  //---------------------------------------------------------------------------
  u32 compress(const INTEGER *src, const u32 size, u8 *dest,
               const Statistics *stats) override;
  //---------------------------------------------------------------------------
  void decompress(INTEGER *dest, const u32 size, const u8 *src) override;
  //---------------------------------------------------------------------------
  void decompress(INTEGER *dest, const u32 size, const u8 *src,
                  const u32 block_offset);
  //---------------------------------------------------------------------------
  bool isPartitioningScheme() override;
};
//---------------------------------------------------------------------------
} // namespace compression