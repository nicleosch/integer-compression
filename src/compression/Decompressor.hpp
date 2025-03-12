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
void frameOfReference(vector<INTEGER>& dest, u8* src, u32 total_size);
//---------------------------------------------------------------------------
/// @brief Decompresses given blockwise Frame-Of-Reference compressed source data
/// to provided destination.
void frameOfReferenceN(vector<INTEGER>& dest, u8* src, u32 total_size, u32 block_size);
//---------------------------------------------------------------------------
}  // decompressor
//---------------------------------------------------------------------------
}  // compression