#include <cmath>
//---------------------------------------------------------------------------
#include "compression/Compressor.hpp"
#include "schemes/FOR.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
u32 compressor::compressFOR(storage::Column col, std::unique_ptr<u8[]> &dest) {
  // allocate space
  dest = std::make_unique<u8[]>(col.size() * sizeof(INTEGER));

  // compute statistics
  auto stats = Statistics::generateFrom(col.data(), col.size());

  // compress
  FOR cFor;
  return cFor.compress(col.data(), col.size(), dest.get(), &stats);
};
//---------------------------------------------------------------------------
} // namespace compression