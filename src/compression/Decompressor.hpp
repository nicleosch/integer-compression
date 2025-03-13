#pragma once
//---------------------------------------------------------------------------
#include "common/Units.hpp"
#include "storage/Column.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
namespace decompressor {
//---------------------------------------------------------------------------
/// @brief Decompresses given Frame-Of-Reference compressed source data to
/// provided destination.
void decompressFOR(vector<INTEGER>& dest, u8* src, u32 total_size);
//---------------------------------------------------------------------------
/// @brief Decompresses given blockwise Frame-Of-Reference compressed source data
/// to provided destination.
void decompressFORn(vector<INTEGER>& dest, u8* src, u32 total_size, u32 block_size);
//---------------------------------------------------------------------------
/// @brief Decompresses given adaptive blockwise Frame-Of-Reference compressed source data
/// to provided destination.
void decompressAdaptiveFORn(vector<INTEGER>& dest, u8* src, u32 total_size, u32 block_size);
//---------------------------------------------------------------------------
}  // decompressor
//---------------------------------------------------------------------------
}  // compression