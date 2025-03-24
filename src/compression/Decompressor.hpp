#pragma once
//---------------------------------------------------------------------------
#include <cstring>
//---------------------------------------------------------------------------
#include "common/Units.hpp"
#include "schemes/FOR.hpp"
#include "schemes/FORn.hpp"
#include "schemes/TinyBlocks.hpp"
#include "storage/Column.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
namespace decompressor {
//---------------------------------------------------------------------------
/// @brief Copies the given uncompressed source data to provided destination.
void decompressUncompressed(vector<INTEGER> &dest, u32 total_size, u8 *src);
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
namespace morsel {
//---------------------------------------------------------------------------
/// @brief Copies the given uncompressed source data into small
/// (i.e. L1 resident) buffers called morsels.
template <const u16 kMorselSize>
void decompressUncompressed(u8 *src, u32 total_size) {
  // allocate space
  vector<INTEGER> dest(kMorselSize);

  auto morsel_count = total_size / kMorselSize;

  // copy
  for (u32 i = 0; i < morsel_count; ++i) {
    std::memcpy(dest.data(), src + i * kMorselSize,
                kMorselSize * sizeof(INTEGER));
  }
};
//---------------------------------------------------------------------------
/// @brief Decompresses given Frame-Of-Reference compressed source data into
/// small (i.e. L1 resident) buffers called morsels.
template <const u16 kMorselSize> void decompressFOR(u8 *src, u32 total_size) {
  // allocate space
  vector<INTEGER> dest(kMorselSize);

  auto morsel_count = total_size / kMorselSize;

  // decompress
  FOR cFor;
  for (u32 i = 0; i < morsel_count; ++i) {
    cFor.decompress(dest.data(), kMorselSize, src, i * kMorselSize);
  }
}
//---------------------------------------------------------------------------
/// @brief Decompresses given blockwise Frame-Of-Reference compressed source
/// data into small (i.e. L1 resident) buffers called morsels.
template <const u16 kMorselSize, const u16 kBlockSize>
void decompressFORn(u8 *src, u32 total_size) {
  // allocate space
  vector<INTEGER> dest(kMorselSize);

  auto morsel_count = total_size / kMorselSize;
  auto blocks_per_morsel = kMorselSize / kBlockSize;

  // decompress
  FORn cForN;
  for (u32 i = 0; i < morsel_count; ++i) {
    cForN.decompress<kBlockSize>(dest.data(), kMorselSize, src,
                                 i * blocks_per_morsel);
  }
}
//---------------------------------------------------------------------------
/// @brief Decompresses given TinyBlocks-compressed source data into small
/// (i.e. L1 resident) buffers called morsels.
template <const u16 kMorselSize, const u16 kBlockSize>
void decompressTinyBlocks(u8 *src, u32 total_size) {
  // allocate space
  vector<INTEGER> dest(kMorselSize);

  auto morsel_count = total_size / kMorselSize;
  auto blocks_per_morsel = kMorselSize / kBlockSize;

  // decompress
  TinyBlocks cTinyBlocks;
  for (u32 i = 0; i < morsel_count; ++i) {
    cTinyBlocks.decompress<kBlockSize>(dest.data(), kMorselSize, src,
                                       i * blocks_per_morsel);
  }
}
//---------------------------------------------------------------------------
} // namespace morsel
//---------------------------------------------------------------------------
} // namespace decompressor
//---------------------------------------------------------------------------
} // namespace compression