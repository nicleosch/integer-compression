#include <immintrin.h>
//---------------------------------------------------------------------------
#include "common/Utils.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
namespace bitpacking {
//---------------------------------------------------------------------------
namespace simd64 {
//---------------------------------------------------------------------------
/// @brief Reads 256 values from "in", writes "bit" 256-bit vectors to "out".
void avxpack(const u64 *in, __m256i *out, const u8 bit);
//---------------------------------------------------------------------------
/// @brief Reads 256 values from "in", writes "bit" 256-bit vectors to "out".
/// Note: Values are not masked before being packed.
void avxpackwithoutmask(const u64 *in, __m256i *out, const u8 bit);
//---------------------------------------------------------------------------
/// @brief Reads "bit" 256-bit vectors from "in", writes 256 values to "out".
void avxunpack(const __m256i *in, u64 *out, const u8 bit);
//---------------------------------------------------------------------------
} // namespace simd64
//---------------------------------------------------------------------------
} // namespace bitpacking
//---------------------------------------------------------------------------
} // namespace compression