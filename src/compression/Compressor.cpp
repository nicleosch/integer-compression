#include <cmath>
//---------------------------------------------------------------------------
#include "compression/Compressor.hpp"
#include "schemes/AdaptiveFORn.hpp"
#include "schemes/FOR.hpp"
#include "schemes/FORn.hpp"
#include "statistics/Statistics.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
static void computeStats(storage::Column &col,
                         vector<compression::Statistics> &stats,
                         u32 block_size) {
  auto block_count = std::ceil(static_cast<double>(col.size()) / block_size);
  for (size_t i = 0; i < block_count; ++i) {
    stats.push_back(compression::Statistics::generateFrom(
        col.data() + i * block_size,
        std::min(block_size, static_cast<u32>(col.size() - i * block_size))));
  }
};
//---------------------------------------------------------------------------
u32 compressor::compressFOR(storage::Column col, std::unique_ptr<u8[]> &dest) {
  // allocate space
  dest = std::make_unique<u8[]>(col.size() * sizeof(INTEGER));

  // compute statistics
  auto stats = Statistics::generateFrom(col.data(), col.size());

  // compress
  FOR cFor;
  return cFor.compress(col.data(), dest.get(), &stats, col.size(), 0);
};
//---------------------------------------------------------------------------
u32 compressor::compressFORn(storage::Column col, std::unique_ptr<u8[]> &dest,
                             u32 block_size) {
  // allocate space
  dest = std::make_unique<u8[]>(col.size() * sizeof(INTEGER));

  // compute statistics
  vector<compression::Statistics> stats;
  computeStats(col, stats, block_size);

  // compress
  FORn cForN;
  return cForN.compress(col.data(), dest.get(), stats.data(), col.size(),
                        block_size);
};
//---------------------------------------------------------------------------
u32 compressor::compressAdaptiveFORn(storage::Column col,
                                     std::unique_ptr<u8[]> &dest,
                                     u32 block_size) {
  // allocate space
  dest = std::make_unique<u8[]>(col.size() * sizeof(INTEGER));

  // compute statistics
  vector<compression::Statistics> stats;
  computeStats(col, stats, block_size);

  // compress
  AdaptiveFORn cAdaptiveForN;
  return cAdaptiveForN.compress(col.data(), dest.get(), stats.data(),
                                col.size(), block_size);
};
//---------------------------------------------------------------------------
} // namespace compression