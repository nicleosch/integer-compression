#pragma once
//---------------------------------------------------------------------------
#include "schemes/FORn.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
class AdaptiveFORn : public FORn {
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
};
//---------------------------------------------------------------------------
}  // namespace compression