#pragma once
//---------------------------------------------------------------------------
#include "schemes/FOR.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
struct FORnSlot {
  INTEGER reference;
  u32 offset;
  u8 pack_size;
};
//---------------------------------------------------------------------------
struct FORnSlots {
  FORnSlot slots[];
};
//---------------------------------------------------------------------------
struct FORnLayout {
  u32 data_offset;
  u8 data[];
};
//---------------------------------------------------------------------------
class FORn : public FOR {
public:
  u32 compress(const INTEGER *src, u8 *dest, const Statistics *stats,
               const u32 total_size, const u16 block_size) override;
  //---------------------------------------------------------------------------
  void decompress(INTEGER *dest, const u8 *src, const u32 total_size,
                  const u16 block_size) override;
  //---------------------------------------------------------------------------
  CompressionSchemeType getType() override;

protected:
  u32 compressImpl(const INTEGER *src, u8 *dest, const Statistics *stats,
                   const u32 total_size, const u16 block_size,
                   u8 (*pack_func)(const INTEGER *, u8 *, const u32, INTEGER));
};
//---------------------------------------------------------------------------
} // namespace compression
