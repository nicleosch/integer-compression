#pragma once
//---------------------------------------------------------------------------
#include "algebra/Predicate.hpp"
#include "bitpacking/BitPacking.hpp"
#include "extern/fastpfor/Delta.hpp"
#include "statistics/Statistics.hpp"
#include "tinyblocks/Metadata.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
namespace tinyblocks {
//---------------------------------------------------------------------------
namespace delta {
//---------------------------------------------------------------------------
template <typename T, const u16 kBlockSize>
u32 compress(const T *src, u8 *dest, Slot<T> &slot) {
  // FOR
  auto buffer = std::make_unique<T[]>(kBlockSize);
  for (u32 i = 0; i < kBlockSize; ++i) {
    buffer[i] = src[i] - slot.reference;
  }
  //---------------------------------------------------------------------------
  // Delta
  external::fastpfor::delta::compress<T, kBlockSize>(buffer.get());
  //---------------------------------------------------------------------------
  // Bitpacking
  //---------------------------------------------------------------------------
  // Prepare
  T min = 0;
  T max = 0;
  for (u16 i = 0; i < kBlockSize; ++i) {
    if (buffer[i] < min)
      min = buffer[i];
    if (buffer[i] > max)
      max = buffer[i];
  }
  u8 pack_size = utils::requiredBits(max - min);
  //---------------------------------------------------------------------------
  slot.opcode = {Scheme::DELTA, pack_size};
  //---------------------------------------------------------------------------
  // Compress
  u32 compressed_size =
      bitpacking::pack<T, kBlockSize>(buffer.get(), dest, pack_size);
  //---------------------------------------------------------------------------
  return compressed_size;
}
//---------------------------------------------------------------------------
template <typename T, const u16 kBlockSize>
void decompress(T *dest, const u8 *src, const Slot<T> &slot) {
  // Bitpacking
  bitpacking::unpack<T, kBlockSize>(dest, src, slot.opcode.payload);
  //---------------------------------------------------------------------------
  // Delta
  external::fastpfor::delta::decompress<T, kBlockSize>(dest);
  //---------------------------------------------------------------------------
  // FOR
  for (u32 i = 0; i < kBlockSize; ++i) {
    dest[i] += slot.reference;
  }
}
//---------------------------------------------------------------------------
template <typename T, const u16 kBlockSize>
void filter(const u8 *data, algebra::Predicate<T> &predicate, Match *matches) {
  // TODO: Implement filter
  throw std::runtime_error("Not implemented yet.");
}
//---------------------------------------------------------------------------
} // namespace delta
//---------------------------------------------------------------------------
} // namespace tinyblocks
//---------------------------------------------------------------------------
} // namespace compression