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
namespace avx {
//---------------------------------------------------------------------------
/// @brief Reads 256 values from "in", writes "bit" 256-bit vectors to "out".
/// Note: This function does NOT mask the input bits before packing it.
void pack(const u32 *in, __m256i *out, const u8 bit);
//---------------------------------------------------------------------------
/// @brief Reads 256 values from "in", writes "bit" 256-bit vectors to "out".
/// Note: This function masks the input bits before packing it.
void packmask(const u32 *in, __m256i *out, const u8 bit);
//---------------------------------------------------------------------------
/// @brief Reads "bit" 256-bit vectors from "in", writes 256 values to "out".
void unpack(const __m256i *in, u32 *out, const u8 bit);
//---------------------------------------------------------------------------
/// @brief Reads "bit" 256-bit vectors from "in" and writes 256 values to
/// "matches". If there is a match with "predicate", the value will be larger
/// than 0, else 0.
void filter(const __m256i *in, u32 *matches, const u8 bit,
            const algebra::Predicate<INTEGER> &predicate);
//---------------------------------------------------------------------------
} // namespace avx
//---------------------------------------------------------------------------
} // namespace simd32
//---------------------------------------------------------------------------
} // namespace bitpacking
//---------------------------------------------------------------------------
} // namespace compression