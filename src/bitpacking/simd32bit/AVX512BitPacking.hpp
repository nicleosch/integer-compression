#pragma once
//---------------------------------------------------------------------------
#include <immintrin.h>
//---------------------------------------------------------------------------
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
/// Note: This function masks the input bits before packing it.
void packmask(const u32 *in, __m512i *out, const u8 bit);
//---------------------------------------------------------------------------
/// @brief Reads "bit" 512-bit vectors from "in", writes 512 values to "out".
void unpack(const __m512i *in, u32 *out, const u8 bit);
//---------------------------------------------------------------------------
} // namespace avx512
//---------------------------------------------------------------------------
} // namespace simd32
//---------------------------------------------------------------------------
} // namespace bitpacking
//---------------------------------------------------------------------------
} // namespace compression