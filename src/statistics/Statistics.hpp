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
  /// The number of values in the input.
  u32 count;
  /// The number of bits required to store the maximum delta between two
  /// consecutive values.
  u8 delta_bits;
  /// The number of bits required to store the difference between maximum
  /// and minimum.
  u8 diff_bits;
  /// The opcode used for TinyBlocks-Compression.
  u8 opcode;
  /// The number of bits required to store the maximum.
  u8 max_bits;
  /// The step size, if it's the same for all deltas, else 0.
  u8 step_size;
  //---------------------------------------------------------------------------
  /// @brief Generates statistics from given data.
  /// @param src The data to generate the statistic from.
  /// @param count The total number of integers.
  /// @return The statistics generated from the data.
  static Statistics generateFrom(T *src, u32 count) {
    auto stats = Statistics(src, count);

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
        step = 0;
      }
    }

    stats.diff_bits = utils::requiredBits<T>(stats.max - stats.min);
    stats.delta_bits = utils::requiredBits<T>(max_diff);
    stats.max_bits = utils::requiredBits<T>(stats.max);

    if (step > 0 && step < 64) {
      stats.opcode = 65;
      stats.step_size = static_cast<u8>(step);
    } else if (stats.max - stats.min == 0) {
      stats.opcode = 0;
    } else {
      stats.opcode = stats.diff_bits;
    }

    return stats;
  }
};
//---------------------------------------------------------------------------
} // namespace compression