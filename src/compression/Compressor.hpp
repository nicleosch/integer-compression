#pragma once
//---------------------------------------------------------------------------
#include "common/Units.hpp"
#include "storage/Column.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
namespace compressor {
//---------------------------------------------------------------------------
/// @brief Compresses given column to provided destination using
/// Frame-Of-Reference encoding and subsequent bitpacking.
/// @return The size of the compressed data in bytes.
u32 frameOfReference(storage::Column col, std::unique_ptr<u8[]>& dest);
//---------------------------------------------------------------------------
/// @brief Compresses given column to provided destination using
/// blockwise Frame-Of-Reference encoding and subsequent blockwise bitpacking.
/// @return The size of the compressed data in bytes.
u32 frameOfReferenceN(storage::Column col, std::unique_ptr<u8[]>& dest, u32 block_size);
//---------------------------------------------------------------------------
}  // compressor
//---------------------------------------------------------------------------
}  // compression

