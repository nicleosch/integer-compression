#include "statistics/Statistics.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
Statistics Statistics::generateFrom(INTEGER *src, u32 count) {
  auto stats = Statistics(src, count);

  stats.min = stats.max = src[0];
  INTEGER step = src[1] - src[0];

  for (u32 i = 0; i < count; ++i) {
    INTEGER current = src[i];
    if (current < stats.min)
      stats.min = current;
    if (current > stats.max)
      stats.max = current;
    if (i > 0 && src[i] - src[i - 1] != step)
      step = 0;
  }

  if (step > 0 && step < 64) {
    stats.required_bits = 65;
    stats.step_size = static_cast<u8>(step);
  } else {
    stats.required_bits = requiredBits(stats.max - stats.min);
  }

  return stats;
}
//---------------------------------------------------------------------------
u8 Statistics::requiredBits(INTEGER value) {
  if (value == 0)
    return 0;
  return static_cast<u8>(sizeof(INTEGER) * 8) -
         __builtin_clz(static_cast<u32>(value));
}
//---------------------------------------------------------------------------
} // namespace compression