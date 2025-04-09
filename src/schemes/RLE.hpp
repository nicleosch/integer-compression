#pragma once
//---------------------------------------------------------------------------
#include "common/Utils.hpp"
#include "statistics/Statistics.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
struct RLELayout {
  u32 value_offset;
  u8 data[];
};
//---------------------------------------------------------------------------
class RLE {
public:
  //---------------------------------------------------------------------------
  u32 compress(const INTEGER *src, const u32 total_size, u8 *dest,
               const Statistics *stats);
  //---------------------------------------------------------------------------
  void decompress(INTEGER *dest, const u32 total_size, const u8 *src);
};
//---------------------------------------------------------------------------
} // namespace compression