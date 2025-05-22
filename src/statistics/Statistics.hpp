#pragma once
//---------------------------------------------------------------------------
#include <cmath>
//---------------------------------------------------------------------------
#include "common/Types.hpp"
#include "common/Utils.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
template <typename T> class Statistics {
public:
  Statistics() = delete;
  //---------------------------------------------------------------------------
  Statistics(T *src, u32 count) : src(src), count(count){};
  //---------------------------------------------------------------------------
  /// The pointer to the data to generate the statistics from.
  T *src;
  /// The minimum.
  T min;
  /// The maximum
  T max;
  /// The step size, if it's the same for all deltas, else 0.
  T step_size;
  /// The number of values in the input.
  u32 count;
  /// Whether there are only positive deltas
  bool delta;
  /// The number of bits required to store the maximum delta between two
  /// consecutive values.
  u8 delta_bits;
  /// The number of bits required to store the difference between maximum
  /// and minimum.
  u8 diff_bits;
  /// The number of bits required to store the maximum.
  u8 max_bits;
  //---------------------------------------------------------------------------
  /// @brief Generates statistics from given data.
  /// @param src The data to generate the statistic from.
  /// @param count The total number of integers.
  /// @return The statistics generated from the data.
  static Statistics generateFrom(T *src, u32 count) {
    auto stats = Statistics(src, count);

    stats.delta = true;
    stats.min = stats.max = src[0];
    T step = src[1] - src[0];
    T max_diff = step;

    for (u32 i = 0; i < count; ++i) {
      T current = src[i];
      if (current < stats.min)
        stats.min = current;
      if (current > stats.max)
        stats.max = current;
      if (i > 0 && src[i] - src[i - 1] != step) {
        if (std::abs(src[i] - src[i - 1]) > std::abs(max_diff) && max_diff >= 0)
          max_diff = src[i] - src[i - 1];
        step = -1;
      }
      if (i > 0 && src[i] - src[i - 1] < 0)
        stats.delta = false;
    }

    stats.diff_bits = utils::requiredBits<T>(stats.max - stats.min);
    stats.delta_bits = utils::requiredBits<T>(max_diff);
    stats.max_bits = utils::requiredBits<T>(stats.max);
    stats.step_size = step;

    return stats;
  }
};
//---------------------------------------------------------------------------
} // namespace compression