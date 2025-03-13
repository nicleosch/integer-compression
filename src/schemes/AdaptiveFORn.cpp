#include "schemes/AdaptiveFORn.hpp"
#include "common/BitPacking.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
u32 AdaptiveFORn::compress(const INTEGER *src, u8 *dest,
                           const Statistics *stats, const u32 total_size,
                           const u16 block_size) {
  return FORn::compressImpl(src, dest, stats, total_size, block_size,
                            BitPacking::packArbitrary);
}
//---------------------------------------------------------------------------
void AdaptiveFORn::decompress(INTEGER *dest, const u8 *src,
                              const u32 total_size, const u16 block_size) {
  return FORn::decompress(dest, src, total_size, block_size);
}
//---------------------------------------------------------------------------
CompressionSchemeType AdaptiveFORn::getType() {
  return CompressionSchemeType::kAdaptiveFORn;
}
//---------------------------------------------------------------------------
} // namespace compression