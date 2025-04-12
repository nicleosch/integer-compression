#include <cstring>
//---------------------------------------------------------------------------
#include "schemes/Uncompressed.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
u32 Uncompressed::compress(const INTEGER *src, const u32 size, u8 *dest,
                           const Statistics *stats) {
  auto total_size = size * sizeof(INTEGER);
  std::memcpy(dest, src, total_size);
  return total_size;
}
//---------------------------------------------------------------------------
void Uncompressed::decompress(INTEGER *dest, const u32 size, const u8 *src) {
  decompress(dest, size, src, 0);
}
//---------------------------------------------------------------------------
void Uncompressed::decompress(INTEGER *dest, const u32 size, const u8 *src,
                              const u32 block_offset) {
  std::memcpy(dest, src + block_offset * sizeof(INTEGER),
              size * sizeof(INTEGER));
}
//---------------------------------------------------------------------------
bool Uncompressed::isPartitioningScheme() { return false; }
//---------------------------------------------------------------------------
} // namespace compression