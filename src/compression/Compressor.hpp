#pragma once
//---------------------------------------------------------------------------
#include <algorithm>
#include <memory>
//---------------------------------------------------------------------------
#include "common/Units.hpp"
#include "schemes/FORn.hpp"
#include "schemes/TinyBlocks.hpp"
#include "statistics/Statistics.hpp"
#include "storage/Column.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
namespace compressor {
//---------------------------------------------------------------------------
/// @brief Copies the uncompressed column to the target. Used for comparison
/// with compression schemes.
/// @return The size of the compressed data in bytes.
u32 compressUncompressed(storage::Column col, std::unique_ptr<u8[]> &dest);
//---------------------------------------------------------------------------
/// @brief Compresses given column to provided destination using LZ4.
/// @return The size of the compressed data in bytes.
u32 compressLZ4(storage::Column col, std::unique_ptr<u8[]> &dest);
//---------------------------------------------------------------------------
/// @brief Compresses given column to provided destination using
/// Frame-Of-Reference compression.
/// @return The size of the compressed data in bytes.
u32 compressFOR(storage::Column col, std::unique_ptr<u8[]> &dest);
//---------------------------------------------------------------------------
/// @brief Compresses given column to provided destination using
/// blockwise Frame-Of-Reference compression.
/// @return The size of the compressed data in bytes.
template <const u16 kBlockSize>
u32 compressFORn(storage::Column col, std::unique_ptr<u8[]> &dest) {
  // allocate space
  dest = std::make_unique<u8[]>(col.size() * sizeof(INTEGER));

  // compute statistics
  vector<Statistics> stats;
  auto block_count = col.size() / kBlockSize;
  for (size_t i = 0; i < block_count; ++i) {
    stats.push_back(
        Statistics::generateFrom(col.data() + i * kBlockSize, kBlockSize));
  }

  // compress
  FORn cForN;
  return cForN.compress<kBlockSize>(col.data(), col.size(), dest.get(),
                                    stats.data());
}
//---------------------------------------------------------------------------
/// @brief Compresses given column to provided destination using TinyBlocks-
/// compression. Compared to the regular FORn, this compression scheme packs
/// data into arbitrary bit widths.
/// @return The size of the compressed data in bytes.
template <const u16 kBlockSize>
u32 compressTinyBlocks(storage::Column col, std::unique_ptr<u8[]> &dest) {
  // allocate space
  dest = std::make_unique<u8[]>(col.size() * sizeof(INTEGER));

  // compute statistics
  vector<Statistics> stats;
  auto block_count = col.size() / kBlockSize;
  for (size_t i = 0; i < block_count; ++i) {
    stats.push_back(
        Statistics::generateFrom(col.data() + i * kBlockSize, kBlockSize));
  }

  // compress
  TinyBlocks cTinyBlocks;
  return cTinyBlocks.compress<kBlockSize>(col.data(), col.size(), dest.get(),
                                          stats.data());
}
//---------------------------------------------------------------------------
} // namespace compressor
//---------------------------------------------------------------------------
} // namespace compression