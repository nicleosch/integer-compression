#pragma once
//---------------------------------------------------------------------------
#include <cmath>
//---------------------------------------------------------------------------
#include "bitpacking/simd32bit/BitPacking.hpp"
#include "bitpacking/simd64bit/BitPacking.hpp"
#include "common/Types.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
namespace bitpacking {
//---------------------------------------------------------------------------
/// @brief Bitpack the given data into arbitrary bit-widths.
/// Note: This function does NOT mask the input bits before packing it.
/// @param src The integers to be packed.
/// @param dest The destination to pack the data to.
/// @param pack_size The amount of bits to be used to pack an integer.
/// @tparam DataType The type of integer to be packed.
/// @tparam kSize The amount of integers to be unpacked.
/// @returns The size of the compressed data in bytes.
template <typename DataType, u32 kSize>
u32 pack(const DataType *src, u8 *dest, const u8 pack_size);
//---------------------------------------------------------------------------
/// @brief Bitpack the given data into arbitrary bit-widths.
/// Note: This function masks the input bits before packing it.
/// @param src The integers to be packed.
/// @param dest The destination to pack the data to.
/// @param pack_size The amount of bits to be used to pack an integer.
/// @tparam DataType The type of integer to be packed.
/// @tparam kSize The amount of integers to be unpacked.
/// @returns The size of the compressed data in bytes.
template <typename DataType, u32 kSize>
u32 packmask(const DataType *src, u8 *dest, const u8 pack_size);
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
  auto out = simd32::sse::packShortLength(reinterpret_cast<const u32 *>(src),
                                          64, reinterpret_cast<__m128i *>(dest),
                                          pack_size);
  return reinterpret_cast<u8 *>(out) - dest;
}
//---------------------------------------------------------------------------
template <>
inline u32 pack<INTEGER, 128>(const INTEGER *src, u8 *dest,
                              const u8 pack_size) {
  simd32::sse::pack(reinterpret_cast<const u32 *>(src),
                    reinterpret_cast<__m128i *>(dest), pack_size);
  return std::ceil(static_cast<double>(pack_size * 128) / 8);
}
//---------------------------------------------------------------------------
template <>
inline u32 pack<INTEGER, 256>(const INTEGER *src, u8 *dest,
                              const u8 pack_size) {
  simd32::avx::pack(reinterpret_cast<const u32 *>(src),
                    reinterpret_cast<__m256i *>(dest), pack_size);
  return std::ceil(static_cast<double>(pack_size * 256) / 8);
}
//---------------------------------------------------------------------------
template <>
inline u32 pack<INTEGER, 512>(const INTEGER *src, u8 *dest,
                              const u8 pack_size) {
  simd32::avx512::pack(reinterpret_cast<const u32 *>(src),
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
inline u32 packmask<INTEGER, 64>(const INTEGER *src, u8 *dest,
                                 const u8 pack_size) {
  return pack<INTEGER, 64>(src, dest, pack_size);
}
//---------------------------------------------------------------------------
template <>
inline u32 packmask<INTEGER, 128>(const INTEGER *src, u8 *dest,
                                  const u8 pack_size) {
  simd32::sse::packmask(reinterpret_cast<const u32 *>(src),
                        reinterpret_cast<__m128i *>(dest), pack_size);
  return std::ceil(static_cast<double>(pack_size * 128) / 8);
}
//---------------------------------------------------------------------------
template <>
inline u32 packmask<INTEGER, 256>(const INTEGER *src, u8 *dest,
                                  const u8 pack_size) {
  simd32::avx::packmask(reinterpret_cast<const u32 *>(src),
                        reinterpret_cast<__m256i *>(dest), pack_size);
  return std::ceil(static_cast<double>(pack_size * 256) / 8);
}
//---------------------------------------------------------------------------
template <>
inline u32 packmask<INTEGER, 512>(const INTEGER *src, u8 *dest,
                                  const u8 pack_size) {
  simd32::avx512::packmask(reinterpret_cast<const u32 *>(src),
                           reinterpret_cast<__m512i *>(dest), pack_size);
  return std::ceil(static_cast<double>(pack_size * 512) / 8);
}
//---------------------------------------------------------------------------
template <>
inline u32 packmask<BIGINT, 64>(const BIGINT *src, u8 *dest,
                                const u8 pack_size) {
  return pack<BIGINT, 64>(src, dest, pack_size);
}
//---------------------------------------------------------------------------
template <>
inline u32 packmask<BIGINT, 128>(const BIGINT *src, u8 *dest,
                                 const u8 pack_size) {
  simd64::simdpack(reinterpret_cast<const u64 *>(src),
                   reinterpret_cast<__m128i *>(dest), pack_size);
  return std::ceil(static_cast<double>(pack_size * 128) / 8);
}
//---------------------------------------------------------------------------
template <>
inline u32 packmask<BIGINT, 256>(const BIGINT *src, u8 *dest,
                                 const u8 pack_size) {
  simd64::avxpack(reinterpret_cast<const u64 *>(src),
                  reinterpret_cast<__m256i *>(dest), pack_size);
  return std::ceil(static_cast<double>(pack_size * 256) / 8);
}
//---------------------------------------------------------------------------
template <>
inline u32 packmask<BIGINT, 512>(const BIGINT *src, u8 *dest,
                                 const u8 pack_size) {
  simd64::avx512pack(reinterpret_cast<const u64 *>(src),
                     reinterpret_cast<__m512i *>(dest), pack_size);
  return std::ceil(static_cast<double>(pack_size * 512) / 8);
}
//---------------------------------------------------------------------------
template <>
inline void unpack<INTEGER, 64>(INTEGER *dest, const u8 *src,
                                const u8 pack_size) {
  simd32::sse::unpackShortLength(reinterpret_cast<const __m128i *>(src), 64,
                                 reinterpret_cast<u32 *>(dest), pack_size);
}
//---------------------------------------------------------------------------
template <>
inline void unpack<INTEGER, 128>(INTEGER *dest, const u8 *src,
                                 const u8 pack_size) {
  simd32::sse::unpack(reinterpret_cast<const __m128i *>(src),
                      reinterpret_cast<u32 *>(dest), pack_size);
}
//---------------------------------------------------------------------------
template <>
inline void unpack<INTEGER, 256>(INTEGER *dest, const u8 *src,
                                 const u8 pack_size) {
  simd32::avx::unpack(reinterpret_cast<const __m256i *>(src),
                      reinterpret_cast<u32 *>(dest), pack_size);
}
//---------------------------------------------------------------------------
template <>
inline void unpack<INTEGER, 512>(INTEGER *dest, const u8 *src,
                                 const u8 pack_size) {
  simd32::avx512::unpack(reinterpret_cast<const __m512i *>(src),
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
/// @brief Pack given integers using SIMD-BinaryPacking.
/// @param src The integers to be packed.
/// @param size The number of integers to be packed.
/// @param dest The destination to pack the data to.
/// @param pack_size The amount of bits to be used to pack an integer.
/// @tparam DataType The type of integer to be packed.
/// @returns The size of the compressed data in bytes.
template <typename DataType>
u32 pack(const DataType *src, const u16 size, u8 *dest, const u8 pack_size);
//---------------------------------------------------------------------------
/// @brief Unpack given integers using SIMD-BinaryPacking.
/// @param dest The resulting integers.
/// @param size The number of integers to be unpacked.
/// @param src The data to unpack.
/// @param pack_size The amount of bits used for packing.
/// @tparam DataType The type of integer to be unpacked.
template <typename DataType>
void unpack(DataType *dest, const u16 size, const u8 *src, const u8 pack_size);
//---------------------------------------------------------------------------
template <>
inline u32 pack<INTEGER>(const INTEGER *src, const u16 size, u8 *dest,
                         const u8 pack_size) {
  auto out =
      simd32::sse::packLength(reinterpret_cast<const u32 *>(src), size,
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
  simd32::sse::unpackLength(reinterpret_cast<const __m128i *>(src), size,
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