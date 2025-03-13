#pragma once
//---------------------------------------------------------------------------
#include "common/Units.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
class Statistics {
public:
  Statistics() = delete;
  //---------------------------------------------------------------------------
  Statistics(INTEGER *src, u32 count) : src(src), count(count){};
  //---------------------------------------------------------------------------
  INTEGER *src;
  u32 count;
  INTEGER min;
  INTEGER max;
  //---------------------------------------------------------------------------
  /// @brief Generates statistics from given data.
  /// @param src The data to generate the statistic from.
  /// @param count The total number of integers.
  /// @return The statistics generated from the data.
  static Statistics generateFrom(INTEGER *src, u32 count) {
    auto stats = Statistics(src, count);
    stats.min = stats.max = src[0];
    for (u32 i = 0; i < count; ++i) {
      INTEGER current = src[i];
      if (current < stats.min)
        stats.min = current;
      if (current > stats.max)
        stats.max = current;
    }
    return stats;
  }
};
//---------------------------------------------------------------------------
} // namespace compression