#pragma once
//---------------------------------------------------------------------------
#include <cassert>
#include <cmath>
#include <type_traits>
//---------------------------------------------------------------------------
#include "common/Types.hpp"
#include "common/Utils.hpp"
//---------------------------------------------------------------------------
namespace compression {
template <typename T> class MiniStatistics {
public:
  using U = typename std::make_unsigned<T>::type;
  //---------------------------------------------------------------------------
  static MiniStatistics generateFrom(const T *src, u32 count) {
    assert(count > 0);
    //---------------------------------------------------------------------------
    MiniStatistics stats(src, count);
    stats.min = src[0];
    stats.max = src[0];
    T step = src[1] - src[0];
    for (u32 i = 1; i < count; ++i) {
      T current = src[i];
      if (current < stats.min)
        stats.min = current;
      if (current > stats.max)
        stats.max = current;
      if (current - src[i - 1] != step)
        step = -1;
    }
    // TODO: The difference between two values of T might not fit into a T
    // value. This is a bug and needs to be fixed.
    stats.diff_bits = utils::requiredBits<U>(static_cast<U>(stats.max) -
                                             static_cast<U>((stats.min)));
    stats.step_size = step;
    //---------------------------------------------------------------------------
    return stats;
  }
  //---------------------------------------------------------------------------
  MiniStatistics() = delete;
  //---------------------------------------------------------------------------
  MiniStatistics(const T *src, u32 count) : src(src), count(count){};
  //---------------------------------------------------------------------------
  /// The pointer to the data to generate the statistics from.
  const T *src;
  /// The minimum.
  T min;
  /// The maximum.
  T max;
  /// The step size, if it's the same for all deltas, else -1.
  T step_size;
  /// The number of values in the input.
  u32 count;
  /// The number of bits required to store the difference between maximum
  /// and minimum.
  u8 diff_bits;
};
//---------------------------------------------------------------------------
template <typename T> class Statistics {
public:
  using U = typename std::make_unsigned<T>::type;
  //---------------------------------------------------------------------------
  static Statistics generateFrom(const T *src, u32 count) {
    assert(count > 0);
    //---------------------------------------------------------------------------
    Statistics stats(src, count);
    //---------------------------------------------------------------------------
    stats.delta = true;
    stats.min = stats.max = src[0];
    T step = src[1] - src[0];
    T max_diff = step;
    //---------------------------------------------------------------------------
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
    //---------------------------------------------------------------------------
    // TODO: The difference between two values of T might not fit into a T
    // value. This is a bug and needs to be fixed.
    stats.diff_bits = utils::requiredBits<U>(static_cast<U>(stats.max) -
                                             static_cast<U>((stats.min)));
    stats.delta_bits = utils::requiredBits<U>(max_diff);
    stats.max_bits = utils::requiredBits<U>(stats.max);
    stats.step_size = step;
    //---------------------------------------------------------------------------
    return stats;
  };
  //---------------------------------------------------------------------------
  Statistics() = delete;
  //---------------------------------------------------------------------------
  Statistics(const T *src, u32 count) : src(src), count(count){};
  //---------------------------------------------------------------------------
  /// The pointer to the data to generate the statistics from.
  const T *src;
  /// The minimum.
  T min;
  /// The maximum
  T max;
  /// The step size, if it's the same for all deltas, else -1.
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
};
//---------------------------------------------------------------------------
} // namespace compression