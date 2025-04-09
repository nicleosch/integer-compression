#pragma once
//---------------------------------------------------------------------------
#include "common/Types.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
class Statistics {
public:
  Statistics() = delete;
  //---------------------------------------------------------------------------
  Statistics(INTEGER *src, u32 count) : src(src), count(count){};
  //---------------------------------------------------------------------------
  /// The pointer to the data to generate the statistics from.
  INTEGER *src;
  /// The number of values in the input.
  u32 count;
  /// The minimum.
  INTEGER min;
  /// The maximum
  INTEGER max;
  /// The number of bits required to store the maximum delta between two
  /// consecutive values.
  u8 delta_bits;
  /// The number of bits required to store the difference between maximum
  /// and minimum.
  u8 diff_bits;
  /// The number of bits required to store the maximum.
  u8 max_bits;
  /// The step size, if it's the same for all deltas, else 0.
  u8 step_size;
  //---------------------------------------------------------------------------
  /// @brief Generates statistics from given data.
  /// @param src The data to generate the statistic from.
  /// @param count The total number of integers.
  /// @return The statistics generated from the data.
  static Statistics generateFrom(INTEGER *src, u32 count);

private:
  static u8 requiredBits(INTEGER value);
};
//---------------------------------------------------------------------------
} // namespace compression