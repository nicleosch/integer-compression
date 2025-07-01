#pragma once
//---------------------------------------------------------------------------
#include "algebra/Predicate.hpp"
#include "statistics/Statistics.hpp"
#include "tinyblocks/Metadata.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
namespace tinyblocks {
//---------------------------------------------------------------------------
namespace monotonic {
//---------------------------------------------------------------------------
template <typename T, const u16 kBlockSize>
u32 compress(const T *src, u8 *dest, const Statistics<T> &stats,
             Slot<T> &slot) {
  assert(stats.step_size >= 0 && stats.step_size < 256);
  slot.opcode = {Scheme::MONOTONIC, static_cast<u8>(stats.step_size)};
  return 0;
}
//---------------------------------------------------------------------------
template <typename T, const u16 kBlockSize>
void decompress(T *dest, const Slot<T> &slot) {
  for (u16 i = 0; i < kBlockSize; ++i) {
    dest[i] = slot.reference + i * slot.opcode.payload;
  }
}
//---------------------------------------------------------------------------
template <typename T, const u16 kBlockSize>
void filter(const T *data, algebra::Predicate<T> &predicate, u8 *matches) {
  // TODO: Implement filter
  throw std::runtime_error("Not implemented yet.");
}
//---------------------------------------------------------------------------
} // namespace monotonic
//---------------------------------------------------------------------------
} // namespace tinyblocks
//---------------------------------------------------------------------------
} // namespace compression