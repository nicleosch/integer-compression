#include "statistics/Statistics.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
Statistics Statistics::generateFrom(INTEGER *src, u32 count) {
  auto stats = Statistics(src, count);

  stats.min = stats.max = src[0];
  INTEGER step = src[1] - src[0];
  INTEGER max_diff = step;

  for (u32 i = 0; i < count; ++i) {
    INTEGER current = src[i];
    if (current < stats.min)
      stats.min = current;
    if (current > stats.max)
      stats.max = current;
    if (i > 0 && src[i] - src[i - 1] != step) {
      if (src[i] - src[i - 1] > max_diff)
        max_diff = src[i] - src[i - 1];
      step = 0;
    }
  }

  if (step > 0 && step < 64) {
    stats.diff_bits = 65;
    stats.step_size = static_cast<u8>(step);
  } else {
    stats.diff_bits = requiredBits(stats.max - stats.min);
  }
  stats.delta_bits = requiredBits(max_diff);
  stats.max_bits = requiredBits(stats.max);

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