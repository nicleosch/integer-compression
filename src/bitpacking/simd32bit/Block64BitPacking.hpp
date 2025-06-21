#pragma once
//---------------------------------------------------------------------------
#include <emmintrin.h>
//---------------------------------------------------------------------------
#include "common/Types.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
namespace bitpacking {
//---------------------------------------------------------------------------
namespace simd32 {
//---------------------------------------------------------------------------
namespace block64 {
//---------------------------------------------------------------------------
constexpr u32 kBlockSize = 64;
//---------------------------------------------------------------------------
/// @brief Reads 64 values from "in", writes ("bit"+1)/2 128-bit vectors to
/// "out". Note: This function does NOT mask the input bits before packing it.
void pack(const u32 *in, __m128i *out, const u8 bit);
//---------------------------------------------------------------------------
/// @brief Reads 64 values from "in", writes ("bit"+1)/2 128-bit vectors to
/// "out". Note: This function masks the input bits before packing it.
void packmask(const u32 *in, __m128i *out, const u8 bit);
//---------------------------------------------------------------------------
/// @brief Reads ("bit"+1)/2 128-bit vectors from "in", writes 64 values to
/// "out".
void unpack(const __m128i *in, u32 *out, const u8 bit);
//---------------------------------------------------------------------------
} // namespace block64
//---------------------------------------------------------------------------
} // namespace simd32
//---------------------------------------------------------------------------
} // namespace bitpacking
//---------------------------------------------------------------------------
} // namespace compression