#pragma once
//---------------------------------------------------------------------------
#include "common/Units.hpp"
#include "schemes/FORn.hpp"
#include "schemes/TinyBlocks.hpp"
#include "storage/Column.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
namespace decompressor {
//---------------------------------------------------------------------------
/// @brief Decompresses given Frame-Of-Reference compressed source data to
/// provided destination.
void decompressFOR(vector<INTEGER> &dest, u32 total_size, u8 *src);
//---------------------------------------------------------------------------
/// @brief Decompresses given blockwise Frame-Of-Reference compressed source
/// data to provided destination.
template <const u16 kBlockSize>
void decompressFORn(vector<INTEGER> &dest, u32 total_size, u8 *src) {
  // allocate space
  dest.reserve(total_size);

  // decompress
  FORn cForN;
  cForN.decompress<kBlockSize>(dest.data(), total_size, src);
}
//---------------------------------------------------------------------------
/// @brief Decompresses given TinyBlocks-compressed source data to
/// provided destination.
template <const u16 kBlockSize>
void decompressTinyBlocks(vector<INTEGER> &dest, u32 total_size, u8 *src) {
  // allocate space
  dest.reserve(total_size);

  // decompress
  TinyBlocks cTinyBlocks;
  cTinyBlocks.decompress<kBlockSize>(dest.data(), total_size, src);
}
//---------------------------------------------------------------------------
} // namespace decompressor
//---------------------------------------------------------------------------
} // namespace compression