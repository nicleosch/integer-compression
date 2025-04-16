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
/// @brief Reads 512 values from "in", writes "bit" 512-bit vectors to "out".
void avx512pack(const u64 *in, __m512i *out, const u8 bit);
//---------------------------------------------------------------------------
/// @brief Reads 512 values from "in", writes "bit" 512-bit vectors to "out".
/// Note: Values are not masked before being packed.
void avx512packwithoutmask(const u64 *in, __m512i *out, const u8 bit);
//---------------------------------------------------------------------------
/// @brief Reads "bit" 512-bit vectors from "in", writes 512 values to "out".
void avx512unpack(const __m512i *in, u64 *out, const u8 bit);
//---------------------------------------------------------------------------
} // namespace simd64
//---------------------------------------------------------------------------
} // namespace bitpacking
//---------------------------------------------------------------------------
} // namespace compression