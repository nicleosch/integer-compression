#pragma once
//---------------------------------------------------------------------------
#include "schemes/CompressionScheme.hpp"
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
class FORn : public CompressionScheme {
  public:
    u32 compress(
      const INTEGER* src,
      u8* dest,
      const Statistics* stats,
      const u32 total_size,
      const u16 block_size
    ) override;
    //---------------------------------------------------------------------------
    void decompress(
        INTEGER* dest,
        const u8* src,
        const u32 total_size,
        const u16 block_size
    ) override;
    //---------------------------------------------------------------------------
    CompressionSchemeType getType() override;

  private:
    u8 compressBlock(
      const INTEGER* src,
      u8* dest,
      FORnSlot& slot,
      const Statistics& stats,
      const u32 size
    );

};
//---------------------------------------------------------------------------
}  // namespace compression
