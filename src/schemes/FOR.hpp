#pragma once
//---------------------------------------------------------------------------
#include "schemes/CompressionScheme.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
struct FORLayout {
  INTEGER reference;
  u8 next_scheme;
  u8 data[];
};
//---------------------------------------------------------------------------
class FOR : public CompressionScheme {
  public:
    void compress(
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
}  // compression