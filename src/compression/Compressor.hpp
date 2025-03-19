#pragma once
//---------------------------------------------------------------------------
#include <algorithm>
#include <memory>
//---------------------------------------------------------------------------
#include "common/Units.hpp"
#include "schemes/AdaptiveFORn.hpp"
#include "schemes/FORn.hpp"
#include "statistics/Statistics.hpp"
#include "storage/Column.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
namespace compressor {
//---------------------------------------------------------------------------
/// @brief Compresses given column to provided destination using
/// Frame-Of-Reference encoding and subsequent bitpacking.
/// @return The size of the compressed data in bytes.
u32 compressFOR(storage::Column col, std::unique_ptr<u8[]> &dest);
//---------------------------------------------------------------------------
/// @brief Compresses given column to provided destination using
/// blockwise Frame-Of-Reference encoding and subsequent blockwise bitpacking.
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
/// @brief Compresses given column to provided destination using
/// blockwise Frame-Of-Reference encoding and subsequent blockwise bitpacking.
/// Compared to the regular FORn, this compression scheme packs data into
/// arbitrary bit widths.
/// @return The size of the compressed data in bytes.
template <const u16 kBlockSize>
u32 compressAdaptiveFORn(storage::Column col, std::unique_ptr<u8[]> &dest) {
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
  AdaptiveFORn cAdaptiveForN;
  return cAdaptiveForN.compress<kBlockSize>(col.data(), col.size(), dest.get(),
                                            stats.data());
}
//---------------------------------------------------------------------------
} // namespace compressor
//---------------------------------------------------------------------------
} // namespace compression
