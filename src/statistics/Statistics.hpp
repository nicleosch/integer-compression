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
  INTEGER *src;
  u32 count;
  INTEGER min;
  INTEGER max;
  u8 required_bits;
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