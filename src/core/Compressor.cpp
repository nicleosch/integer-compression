#include <cmath>
#include <cstring>
#include <lz4.h>
//---------------------------------------------------------------------------
#include "core/Compressor.hpp"
#include "schemes/FOR.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
u32 compressor::compressUncompressed(storage::Column col,
                                     std::unique_ptr<u8[]> &dest) {
  // allocate space
  dest = std::make_unique<u8[]>(col.size() * sizeof(INTEGER));

  // copy
  std::memcpy(dest.get(), col.data(), col.size() * sizeof(INTEGER));

  return col.size() * sizeof(INTEGER);
};
//---------------------------------------------------------------------------
u32 compressor::compressLZ4(storage::Column col, std::unique_ptr<u8[]> &dest) {
  auto size = col.size() * sizeof(INTEGER);
  // allocate space
  dest = std::make_unique<u8[]>(size);

  // compress
  return static_cast<u32>(
      LZ4_compress_default(reinterpret_cast<const char *>(col.data()),
                           reinterpret_cast<char *>(dest.get()), size, size));
}
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