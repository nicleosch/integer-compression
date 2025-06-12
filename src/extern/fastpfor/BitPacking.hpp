#pragma once
//---------------------------------------------------------------------------
#include <simdfastpfor.h>
//---------------------------------------------------------------------------
#include "common/Types.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
namespace pfor {
//---------------------------------------------------------------------------
constexpr u8 kScalarBlockSize = 32;
//---------------------------------------------------------------------------
/// @brief Bitpack the given data of arbitrary size into arbitrary bit-widths.
/// In particular, pack as many integers as posssible using SIMD-BinaryPacking
/// and the rest using Scalar-Binarypacking.
/// Note: This function is derived from "packmeupwithoutmasksimd" in the
/// simdfastpfor.h header but adjusted to this project's types.
template <typename T>
inline u32 packAdaptive(vector<T> &src, u32 *dest, const u8 bit);
//---------------------------------------------------------------------------
template <>
inline u32 packAdaptive<INTEGER>(vector<INTEGER> &src, u32 *dest,
                                 const u8 bit) {
  auto write_ptr = dest;
  const u32 size = static_cast<u32>(src.size());
  if (size == 0)
    return 0;
  src.resize((size + 32 - 1) / 32 * 32);
  u32 j = 0;
  for (; j + 128 <= size; j += 128) {
    FastPForLib::usimdpackwithoutmask(reinterpret_cast<u32 *>(&src[j]),
                                      reinterpret_cast<__m128i *>(write_ptr),
                                      bit);
    write_ptr += 4 * bit;
  }
  for (; j < size; j += 32) {
    FastPForLib::fastpackwithoutmask(reinterpret_cast<u32 *>(&src[j]),
                                     write_ptr, bit);
    write_ptr += bit;
  }
  write_ptr -= (j - size) * bit / 32;
  src.resize(size);
  return (write_ptr - dest) * sizeof(u32);
}
//---------------------------------------------------------------------------
template <>
inline u32 packAdaptive<BIGINT>(vector<BIGINT> &src, u32 *dest, const u8 bit) {
  throw std::runtime_error(
      "TODO: Adaptive Packing for BigInt not implemented yet.");
}
//---------------------------------------------------------------------------
/// @brief Unpack the given data.
/// In particular, unpack as many integers as posssible using SIMD-BinaryPacking
/// and the rest using Scalar-Binarypacking.
/// Note: This function is derived from "unpackmesimd" in the
/// simdfastpfor.h header but we adjusted the code to this project's types.
template <typename T>
inline u32 unpackAdaptive(vector<T> &dest, const u32 *src, const u8 bit);
//---------------------------------------------------------------------------
template <>
inline u32 unpackAdaptive<INTEGER>(vector<INTEGER> &dest, const u32 *src,
                                   const u8 bit) {
  const u32 size = static_cast<u32>(dest.size());
  dest.resize((size + 32 - 1) / 32 * 32);
  u32 j = 0;
  for (; j + 128 <= size; j += 128) {
    FastPForLib::usimdunpack(reinterpret_cast<const __m128i *>(src),
                             reinterpret_cast<u32 *>(&dest[j]), bit);
    src += 4 * bit;
  }
  for (; j + 31 < size; j += 32) {
    FastPForLib::fastunpack(src, reinterpret_cast<u32 *>(&dest[j]), bit);
    src += bit;
  }
  u32 buffer[kScalarBlockSize];
  u32 remaining = size - j;
  memcpy(buffer, src, (remaining * bit + 31) / 32 * sizeof(u32));
  u32 *bpointer = buffer;
  src += (dest.size() - j) / 32 * bit;
  for (; j < size; j += 32) {
    FastPForLib::fastunpack(bpointer, reinterpret_cast<u32 *>(&dest[j]), bit);
    bpointer += bit;
  }
  src -= (j - size) * bit / 32;
  dest.resize(size);
  return size * sizeof(u32);
}
//---------------------------------------------------------------------------
template <>
inline u32 unpackAdaptive<BIGINT>(vector<BIGINT> &dest, const u32 *src,
                                  const u8 bit) {
  throw std::runtime_error(
      "TODO: Adaptive Unpacking for BigInt not implemented yet.");
}
//---------------------------------------------------------------------------
} // namespace pfor
//---------------------------------------------------------------------------
} // namespace compression