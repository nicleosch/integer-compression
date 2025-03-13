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
  u8 compressImpl(const INTEGER *src, u8 *dest, const Statistics *stats,
                  const u32 total_size,
                  u8 (*bitpack_func)(const INTEGER *, u8 *, const u32,
                                     INTEGER));
  //---------------------------------------------------------------------------
  void decompressImpl(INTEGER *dest, const u8 *src, const u32 total_size,
                      const u32 reference, const u8 pack_size);
};
//---------------------------------------------------------------------------
} // namespace compression