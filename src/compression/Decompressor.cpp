#include "compression/Decompressor.hpp"
#include "schemes/AdaptiveFORn.hpp"
#include "schemes/FOR.hpp"
#include "schemes/FORn.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
void decompressor::decompressFOR(vector<INTEGER> &dest, u8 *src,
                                 u32 total_size) {
  // allocate space
  dest.reserve(total_size);

  // decompress
  FOR cFor;
  cFor.decompress(dest.data(), src, total_size, 0);
};
//---------------------------------------------------------------------------
void decompressor::decompressFORn(vector<INTEGER> &dest, u8 *src,
                                  u32 total_size, u32 block_size) {
  // allocate space
  dest.reserve(total_size);

  // decompress
  FORn cForN;
  cForN.decompress(dest.data(), src, total_size, block_size);
}
//---------------------------------------------------------------------------
void decompressor::decompressAdaptiveFORn(vector<INTEGER> &dest, u8 *src,
                                          u32 total_size, u32 block_size) {
  // allocate space
  dest.reserve(total_size);

  // decompress
  AdaptiveFORn cAdaptiveForN;
  cAdaptiveForN.decompress(dest.data(), src, total_size, block_size);
}
} // namespace compression