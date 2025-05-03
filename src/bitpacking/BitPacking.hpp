#pragma once
//---------------------------------------------------------------------------
#include <cmath>
#include <simdcomp.h>
//---------------------------------------------------------------------------
#include "bitpacking/simd64bit/BitPacking.hpp"
#include "common/Types.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
namespace bitpacking {
//---------------------------------------------------------------------------
/// @brief Bitpack the given data into arbitrary bit-widths.
/// @param src The integers to be packed.
/// @param dest The destination to pack the data to.
/// @param pack_size The amount of bits to be used to pack an integer.
/// @tparam DataType The type of integer to be packed.
/// @tparam kSize The amount of integers to be unpacked.
/// @returns The size of the compressed data in bytes.
template <typename DataType, u32 kSize>
u32 pack(const DataType *src, u8 *dest, const u8 pack_size);
//---------------------------------------------------------------------------
/// @brief Unpack the given data.
/// @param dest The resulting integers.
/// @param src The data to unpack.
/// @param pack_size The amount of bits used for packing.
/// @tparam DataType The type of integer to be unpacked.
/// @tparam kSize The amount of integers to be unpacked.
template <typename DataType, u32 kSize>
void unpack(DataType *dest, const u8 *src, const u8 pack_size);
//---------------------------------------------------------------------------
template <>
inline u32 pack<INTEGER, 64>(const INTEGER *src, u8 *dest, const u8 pack_size) {
  auto out = simdpack_shortlength(reinterpret_cast<const u32 *>(src), 64,
                                  reinterpret_cast<__m128i *>(dest), pack_size);
  return reinterpret_cast<u8 *>(out) - dest;
}
//---------------------------------------------------------------------------
template <>
inline u32 pack<INTEGER, 128>(const INTEGER *src, u8 *dest,
                              const u8 pack_size) {
  simdpackwithoutmask(reinterpret_cast<const u32 *>(src),
                      reinterpret_cast<__m128i *>(dest), pack_size);
  return std::ceil(static_cast<double>(pack_size * 128) / 8);
}
//---------------------------------------------------------------------------
template <>
inline u32 pack<INTEGER, 256>(const INTEGER *src, u8 *dest,
                              const u8 pack_size) {
  avxpackwithoutmask(reinterpret_cast<const u32 *>(src),
                     reinterpret_cast<__m256i *>(dest), pack_size);
  return std::ceil(static_cast<double>(pack_size * 256) / 8);
}
//---------------------------------------------------------------------------
template <>
inline u32 pack<INTEGER, 512>(const INTEGER *src, u8 *dest,
                              const u8 pack_size) {
  avx512packwithoutmask(reinterpret_cast<const u32 *>(src),
                        reinterpret_cast<__m512i *>(dest), pack_size);
  return std::ceil(static_cast<double>(pack_size * 512) / 8);
}
//---------------------------------------------------------------------------
template <>
inline u32 pack<BIGINT, 64>(const BIGINT *src, u8 *dest, const u8 pack_size) {
  auto out = simd64::simdpack_shortlength<64>(
      reinterpret_cast<const u64 *>(src), reinterpret_cast<__m128i *>(dest),
      pack_size);
  return reinterpret_cast<u8 *>(out) - dest;
}
//---------------------------------------------------------------------------
template <>
inline u32 pack<BIGINT, 128>(const BIGINT *src, u8 *dest, const u8 pack_size) {
  simd64::simdpackwithoutmask(reinterpret_cast<const u64 *>(src),
                              reinterpret_cast<__m128i *>(dest), pack_size);
  return std::ceil(static_cast<double>(pack_size * 128) / 8);
}
//---------------------------------------------------------------------------
template <>
inline u32 pack<BIGINT, 256>(const BIGINT *src, u8 *dest, const u8 pack_size) {
  simd64::avxpackwithoutmask(reinterpret_cast<const u64 *>(src),
                             reinterpret_cast<__m256i *>(dest), pack_size);
  return std::ceil(static_cast<double>(pack_size * 256) / 8);
}
//---------------------------------------------------------------------------
template <>
inline u32 pack<BIGINT, 512>(const BIGINT *src, u8 *dest, const u8 pack_size) {
  simd64::avx512packwithoutmask(reinterpret_cast<const u64 *>(src),
                                reinterpret_cast<__m512i *>(dest), pack_size);
  return std::ceil(static_cast<double>(pack_size * 512) / 8);
}
//---------------------------------------------------------------------------
template <>
inline void unpack<INTEGER, 64>(INTEGER *dest, const u8 *src,
                                const u8 pack_size) {
  simdunpack_shortlength(reinterpret_cast<const __m128i *>(src), 64,
                         reinterpret_cast<u32 *>(dest), pack_size);
}
//---------------------------------------------------------------------------
template <>
inline void unpack<INTEGER, 128>(INTEGER *dest, const u8 *src,
                                 const u8 pack_size) {
  simdunpack(reinterpret_cast<const __m128i *>(src),
             reinterpret_cast<u32 *>(dest), pack_size);
}
//---------------------------------------------------------------------------
template <>
inline void unpack<INTEGER, 256>(INTEGER *dest, const u8 *src,
                                 const u8 pack_size) {
  avxunpack(reinterpret_cast<const __m256i *>(src),
            reinterpret_cast<u32 *>(dest), pack_size);
}
//---------------------------------------------------------------------------
template <>
inline void unpack<INTEGER, 512>(INTEGER *dest, const u8 *src,
                                 const u8 pack_size) {
  avx512unpack(reinterpret_cast<const __m512i *>(src),
               reinterpret_cast<u32 *>(dest), pack_size);
}
//---------------------------------------------------------------------------
template <>
inline void unpack<BIGINT, 64>(BIGINT *dest, const u8 *src,
                               const u8 pack_size) {
  simd64::simdunpack_shortlength<64>(reinterpret_cast<const __m128i *>(src),
                                     reinterpret_cast<u64 *>(dest), pack_size);
}
//---------------------------------------------------------------------------
template <>
inline void unpack<BIGINT, 128>(BIGINT *dest, const u8 *src,
                                const u8 pack_size) {
  simd64::simdunpack(reinterpret_cast<const __m128i *>(src),
                     reinterpret_cast<u64 *>(dest), pack_size);
}
//---------------------------------------------------------------------------
template <>
inline void unpack<BIGINT, 256>(BIGINT *dest, const u8 *src,
                                const u8 pack_size) {
  simd64::avxunpack(reinterpret_cast<const __m256i *>(src),
                    reinterpret_cast<u64 *>(dest), pack_size);
}
//---------------------------------------------------------------------------
template <>
inline void unpack<BIGINT, 512>(BIGINT *dest, const u8 *src,
                                const u8 pack_size) {
  simd64::avx512unpack(reinterpret_cast<const __m512i *>(src),
                       reinterpret_cast<u64 *>(dest), pack_size);
}
//---------------------------------------------------------------------------
/// @brief Bitpack the given data of arbitrary size into arbitrary bit-widths.
/// @param src The integers to be packed.
/// @param dest The destination to pack the data to.
/// @param pack_size The amount of bits to be used to pack an integer.
/// @tparam DataType The type of integer to be packed.
/// @returns The size of the compressed data in bytes.
template <typename DataType>
u32 pack(const DataType *src, const u16 size, u8 *dest, const u8 pack_size);
//---------------------------------------------------------------------------
/// @brief Unpack the given data.
/// @param dest The resulting integers.
/// @param src The data to unpack.
/// @param pack_size The amount of bits used for packing.
/// @tparam DataType The type of integer to be unpacked.
/// @tparam kSize The amount of integers to be unpacked.
template <typename DataType>
void unpack(DataType *dest, const u16 size, const u8 *src, const u8 pack_size);
//---------------------------------------------------------------------------
template <>
inline u32 pack<INTEGER>(const INTEGER *src, const u16 size, u8 *dest,
                         const u8 pack_size) {
  auto out = simdpack_length(reinterpret_cast<const u32 *>(src), size,
                             reinterpret_cast<__m128i *>(dest), pack_size);
  return reinterpret_cast<u8 *>(out) - dest;
}
//---------------------------------------------------------------------------
template <>
inline u32 pack<BIGINT>(const BIGINT *src, const u16 size, u8 *dest,
                        const u8 pack_size) {
  auto out =
      simd64::simdpack_length(reinterpret_cast<const u64 *>(src), size,
                              reinterpret_cast<__m128i *>(dest), pack_size);
  return reinterpret_cast<u8 *>(out) - dest;
}
//---------------------------------------------------------------------------
template <>
inline void unpack<INTEGER>(INTEGER *dest, const u16 size, const u8 *src,
                            const u8 pack_size) {
  simdunpack_length(reinterpret_cast<const __m128i *>(src), size,
                    reinterpret_cast<u32 *>(dest), pack_size);
}
//---------------------------------------------------------------------------
template <>
inline void unpack<BIGINT>(BIGINT *dest, const u16 size, const u8 *src,
                           const u8 pack_size) {
  simd64::simdunpack_length(reinterpret_cast<const __m128i *>(src), size,
                            reinterpret_cast<u64 *>(dest), pack_size);
}
//---------------------------------------------------------------------------
} // namespace bitpacking
//---------------------------------------------------------------------------
} // namespace compression