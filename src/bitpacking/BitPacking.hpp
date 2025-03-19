#pragma once
//---------------------------------------------------------------------------
#include "common/Units.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
namespace bitpacking {
//---------------------------------------------------------------------------
/// @brief Bitpack the given data into arbitrary bit-widths.
/// @param src The integers to be packed.
/// @param dest The destination to pack the data to.
/// @param size The amount of integers to be packed.
/// @param pack_size The amount of bits to be used to pack an integer.
void pack(const INTEGER *src, u8 *dest, const u32 length, const u8 pack_size);
//---------------------------------------------------------------------------
/// @brief Unpack the given data.
/// @param dest The resulting integers.
/// @param src The data to unpack.
/// @param size The amount of integers to be unpacked.
/// @param pack_size The amount of bits used for packing.
void unpack(INTEGER *dest, const u8 *src, const u32 length, const u8 pack_size);
//---------------------------------------------------------------------------
}  // namespace bitpacking
//---------------------------------------------------------------------------
}  // namespace compression