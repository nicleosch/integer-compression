#include "statistics/Statistics.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
Statistics Statistics::generateFrom(INTEGER *src, u32 count) {
  auto stats = Statistics(src, count);
  stats.min = stats.max = src[0];
  for (u32 i = 0; i < count; ++i) {
    INTEGER current = src[i];
    if (current < stats.min)
      stats.min = current;
    if (current > stats.max)
      stats.max = current;
  }
  stats.required_bits = requiredBits(stats.max - stats.min);
  return stats;
}
//---------------------------------------------------------------------------
u8 Statistics::requiredBits(INTEGER value) {
  if (value == 0)
    return 65;
  return static_cast<u8>(sizeof(INTEGER) * 8) -
         __builtin_clz(static_cast<u32>(value));
}
//---------------------------------------------------------------------------
} // namespace compression