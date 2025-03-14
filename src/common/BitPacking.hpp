#pragma once
//---------------------------------------------------------------------------
#include <cassert>
//---------------------------------------------------------------------------
#include "common/Units.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
class BitPacking {
public:
  /// @brief Bitpack the given data into fixed bit-widths (8, 16 or 32).
  /// @param src The integers to be packed.
  /// @param dest The destination to pack the data to.
  /// @param size The amount of integers to be packed.
  /// @param diff The difference between min and max in the source data.
  /// @return The amount of bits used to pack the data.
  static u8 packFixed(const INTEGER *src, u8 *dest, const u32 size,
                      INTEGER diff) {
    auto pack_size = packSizeFor(diff);

    if (pack_size <= 8) {
      compress<u8>(src, dest, size, 8);
      return 8;
    } else if (pack_size <= 16) {
      compress<u16>(src, dest, size, 16);
      return 16;
    } else {
      compress<u32>(src, dest, size, 32);
      return 32;
    }
  }
  //---------------------------------------------------------------------------
  /// @brief Bitpack the given data into arbitrary bit-widths.
  /// @param src The integers to be packed.
  /// @param dest The destination to pack the data to.
  /// @param size The amount of integers to be packed.
  /// @param diff The difference between min and max in the source data.
  /// @return The amount of bits used to pack the data.
  static u8 packArbitrary(const INTEGER *src, u8 *dest, const u32 size,
                          INTEGER diff) {
    auto pack_size = packSizeFor(diff);

    if (pack_size <= 8) {
      compress<u8>(src, dest, size, pack_size);
    } else if (pack_size <= 16) {
      compress<u16>(src, dest, size, pack_size);
    } else {
      compress<u32>(src, dest, size, pack_size);
    }

    return pack_size;
  }
  //---------------------------------------------------------------------------
  /// @brief Unpack the given data.
  /// @param dest The resulting integers.
  /// @param src The data to unpack.
  /// @param size The amount of integers to be unpacked.
  /// @param pack_size The amount of bits used for packing.
  static void unpack(INTEGER *dest, const u8 *src, const u32 size,
                     const u8 pack_size) {
    if (pack_size <= 8) {
      decompress<u8>(dest, src, size, pack_size);
    } else if (pack_size <= 16) {
      decompress<u16>(dest, src, size, pack_size);
    } else {
      decompress<u32>(dest, src, size, pack_size);
    }
  }

private:
  static u8 packSizeFor(INTEGER value) {
    if (value < 0)
      return 32; // TODO: Improve handling negative integers
    if (value == 0)
      value += 1;
    return static_cast<u8>(sizeof(INTEGER) * 8) -
           __builtin_clz(static_cast<u32>(value));
  }
  //---------------------------------------------------------------------------
  template <typename T>
  static void compress(const INTEGER *src, u8 *dest, const u32 size,
                       const u8 pack_size) {
    auto data = reinterpret_cast<T *>(dest);

    // The size of T in bits.
    u8 t_size = sizeof(T) * 8;
    // The remaining free bits in T.
    u8 rest = t_size;

    // The index in the destination array.
    u32 dest_i = 0;
    for (u32 src_i = 0; src_i < size; ++src_i) {
      if (rest >= pack_size) { // There is space for another value.
        rest -= pack_size;
        data[dest_i] |= (static_cast<T>(src[src_i]) << rest);
      } else { // There is not enough space for another value, thus wrap the
               // value.
        ++dest_i;

        // The number of bits wrapped onto the right T.
        u8 right_bits = pack_size - rest;
        // The number of bits that stay on the left T.
        u8 left_bits = pack_size - right_bits;

        // A mask for the part that wraps to the right T.
        T right_mask = (1ULL << (right_bits + 1)) - 1;
        // A mask for the part that stays on the left T.
        T left_mask = ((1ULL << (pack_size + 1)) - 1) << right_bits;

        auto value = static_cast<T>(src[src_i]);
        data[dest_i - 1] |= ((value & left_mask) >> (pack_size - left_bits));
        data[dest_i] |= ((value & right_mask) << (t_size - right_bits));

        rest = t_size - right_bits;
      }
    }
  }
  //---------------------------------------------------------------------------
  template <typename T>
  static void decompress(INTEGER *dest, const u8 *src, const u32 size,
                         const u8 pack_size) {
    const auto &data = reinterpret_cast<const T *>(src);

    // The size of T in bits.
    u8 t_size = sizeof(T) * 8;
    // The remaining free bits in T.
    u8 rest = t_size;
    // A mask to only read a pack at a time.
    T pack_mask = (1ULL << (pack_size)) - 1;

    // The index in the source array.
    u32 src_i = 0;
    for (u32 dest_i = 0; dest_i < size; ++dest_i) {
      if (rest >= pack_size) { // There is another value in current T.
        rest -= pack_size;
        dest[dest_i] = static_cast<INTEGER>((data[src_i] >> rest) & pack_mask);
      } else { // The value is wrapped to the next T.
        ++src_i;

        // The number of bits wrapped onto the right T.
        u8 right_bits = pack_size - rest;
        // The number of bits that stay on the left T.
        u8 left_bits = pack_size - right_bits;

        // A mask for the part that is located in the right T.
        T right_mask = ((1ULL << (right_bits + 1)) - 1)
                       << (t_size - right_bits);
        // A mask for the part that is located in the left T.
        T left_mask = (1ULL << (left_bits)) - 1;

        T left_part = (data[src_i - 1] & left_mask) << right_bits;
        T right_part = (data[src_i] & right_mask) >> (t_size - right_bits);
        dest[dest_i] = static_cast<INTEGER>(left_part | right_part);

        rest = t_size - right_bits;
      }
    }
  }
};
//---------------------------------------------------------------------------
} // namespace compression