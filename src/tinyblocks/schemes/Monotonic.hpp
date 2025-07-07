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
void filter(const u8 *data, const Slot<T> &slot,
            const algebra::Predicate<T> &predicate, Match *matches) {
  auto step = slot.opcode.payload;
  auto value = predicate.getValue();
  switch (predicate.getType()) {
  case algebra::PredicateType::EQ: { // Equality Predicate
    if (step > 0) {
      if ((value - slot.reference) % step == 0)
        ++matches[(value - slot.reference) / step];
    } else {
      std::fill(matches, matches + kBlockSize, 1);
    }
    return;
  }
  case algebra::PredicateType::GT: { // GreaterThan Predicate
    u32 min = (value - slot.reference) / step + 1;
    for (u32 i = min; i < kBlockSize; ++i) {
      ++matches[i];
    }
    return;
  }
  case algebra::PredicateType::LT: { // LessThan Predicate
    u32 numerator = value - slot.reference;
    u32 denominator = step;
    u32 max = (numerator + denominator - 1) / denominator;
    for (u32 i = 0; i < max; ++i) {
      ++matches[i];
    }
    return;
  }
  case algebra::PredicateType::INEQ: { // Inequality Predicate
    assert(!(step == 0));              // Should be pre-filtered
    std::fill(matches, matches + kBlockSize, 1);
    if ((value - slot.reference) % step == 0)
      matches[(value - slot.reference) / step] = 0;
    return;
  }
  }
}
//---------------------------------------------------------------------------
} // namespace monotonic
//---------------------------------------------------------------------------
} // namespace tinyblocks
//---------------------------------------------------------------------------
} // namespace compression