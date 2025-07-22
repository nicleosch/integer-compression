#pragma once
//---------------------------------------------------------------------------
#include <immintrin.h>
//---------------------------------------------------------------------------
#include "algebra/Predicate.hpp"
#include "common/Types.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
namespace bitpacking {
//---------------------------------------------------------------------------
namespace simd32 {
//---------------------------------------------------------------------------
namespace avx512 {
//---------------------------------------------------------------------------
/// @brief Reads 512 values from "in", writes "bit" 512-bit vectors to "out".
/// Note: This function does NOT mask the input bits before packing it.
void pack(const u32 *in, __m512i *out, const u8 bit);
//---------------------------------------------------------------------------
/// @brief Reads 512 values from "in", writes "bit" 512-bit vectors to "out".
/// Note: This adaptively packs the data using 8, 16 or 32 bit lanes, depending
/// on the given bit. This allows for faster filter evaluation while keeping
/// unpacking speed about constant.
void packfast(const u32 *in, __m512i *out, const u8 bit);
//---------------------------------------------------------------------------
/// @brief Reads 512 values from "in", writes "bit" 512-bit vectors to "out".
/// Note: This function masks the input bits before packing it.
void packmask(const u32 *in, __m512i *out, const u8 bit);
//---------------------------------------------------------------------------
/// @brief Reads "bit" 512-bit vectors from "in", writes 512 values to "out".
void unpack(const __m512i *in, u32 *out, const u8 bit);
//---------------------------------------------------------------------------
/// @brief Reads "bit" 512-bit vectors from "in", writes 512 values to "out".
/// Note: This adaptively unpacks the data using 8, 16 or 32 bit lanes,
/// depending on the given bit. This allows for faster filter evaluation while
/// keeping unpacking speed about constant.
void unpackfast(const __m512i *in, u32 *out, const u8 bit);
//---------------------------------------------------------------------------
/// @brief Reads "bit" 512-bit vectors from "in" and writes 512 values to
/// "matches". If there is a match with "predicate", the value will be larger
/// than 0, else 0.
void filter(const __m512i *in, u32 *matches, const u8 bit,
            const algebra::Predicate<INTEGER> &predicate);
//---------------------------------------------------------------------------
/// @brief Reads "bit" 512-bit vectors from "in" and writes 512 values to
/// "matches". If there is a match with "predicate", the value will be larger
/// than 0, else 0.
/// Note: This only works on the new layout, but using that layout this can
/// perform comparisons up to 5 times faster than filters used to (for bit sizes
/// <= 16 as otherwise the layout is the same).
void filterfast(const __m512i *in, u8 *matches, const u8 bit,
                const algebra::Predicate<INTEGER> &predicate);
//---------------------------------------------------------------------------
/// @brief Reads "bit" 512-bit vectors from "in" and writes a 512 value bit mask
/// to "matches". If there is a match with "predicate", the bit at respective
/// index will be set, else not.
void filterfastmask(const __m512i *in, __m512i *match_bitmap, const u8 bit,
                    const algebra::Predicate<INTEGER> &predicate);
} // namespace avx512
//---------------------------------------------------------------------------
} // namespace simd32
//---------------------------------------------------------------------------
} // namespace bitpacking
//---------------------------------------------------------------------------
} // namespace compression