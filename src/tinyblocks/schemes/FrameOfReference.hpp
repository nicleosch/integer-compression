#pragma once
//---------------------------------------------------------------------------
#include "algebra/Predicate.hpp"
#include "bitpacking/BitPacking.hpp"
#include "statistics/Statistics.hpp"
#include "tinyblocks/Metadata.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
namespace tinyblocks {
//---------------------------------------------------------------------------
namespace frameofreference {
//---------------------------------------------------------------------------
template <typename T, const u16 kBlockSize>
u32 compress(const T *src, u8 *dest, const Statistics<T> &stats,
             Slot<T> &slot) {
  assert(stats.diff_bits >= 0 && stats.diff_bits <= sizeof(T) * 8);
  slot.opcode = {Scheme::FOR, stats.diff_bits};
  //---------------------------------------------------------------------------
  // Normalize
  vector<T> normalized(kBlockSize);
  for (u32 i = 0; i < kBlockSize; ++i) {
    normalized[i] = src[i] - slot.reference;
  }
  //---------------------------------------------------------------------------
  // Compress
  u8 pack_size = slot.opcode.payload;
  return bitpacking::pack<T, kBlockSize>(normalized.data(), dest, pack_size);
}
//---------------------------------------------------------------------------
template <typename T, const u16 kBlockSize>
void decompress(T *dest, const u8 *src, const Slot<T> &slot) {
  // Decompress
  bitpacking::unpack<T, kBlockSize>(dest, src, slot.opcode.payload);
  //---------------------------------------------------------------------------
  // Denormalize
  for (u32 i = 0; i < kBlockSize; ++i) {
    dest[i] += slot.reference;
  }
}
//---------------------------------------------------------------------------
template <typename T, const u16 kBlockSize>
void filter(const u8 *data, const Slot<T> &slot,
            algebra::Predicate<T> &predicate, Match *matches) {
  // Normalize the value to filter
  predicate.setValue(predicate.getValue() - slot.reference);
  bitpacking::filter<T, kBlockSize>(data, matches, slot.opcode.payload,
                                    predicate);
  // Revert the changes to the filter predicate
  predicate.setValue(predicate.getValue() + slot.reference);
}
//---------------------------------------------------------------------------
} // namespace frameofreference
//---------------------------------------------------------------------------
} // namespace tinyblocks
//---------------------------------------------------------------------------
} // namespace compression