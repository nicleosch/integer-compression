#include <lz4.h>
//---------------------------------------------------------------------------
#include "compression/Decompressor.hpp"
#include "schemes/FOR.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
void decompressor::decompressUncompressed(vector<INTEGER> &dest, u32 total_size,
                                          u8 *src) {
  // allocate space
  dest.reserve(total_size);

  // copy
  std::memcpy(dest.data(), src, total_size * sizeof(INTEGER));
};
//---------------------------------------------------------------------------
void decompressor::decompressLZ4(vector<INTEGER> &dest, u32 total_size, u8 *src,
                                 u32 compressed_size) {
  // allocate space
  dest.reserve(total_size);

  // decompress
  LZ4_decompress_safe(reinterpret_cast<const char *>(src),
                      reinterpret_cast<char *>(dest.data()), compressed_size,
                      total_size * sizeof(INTEGER));
}
//---------------------------------------------------------------------------
void decompressor::decompressFOR(vector<INTEGER> &dest, u32 total_size,
                                 u8 *src) {
  // allocate space
  dest.reserve(total_size);

  // decompress
  FOR cFor;
  cFor.decompress(dest.data(), total_size, src);
};
//---------------------------------------------------------------------------
} // namespace compression