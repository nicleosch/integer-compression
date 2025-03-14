#pragma once
//---------------------------------------------------------------------------
#include <memory>
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
u32 compressFOR(storage::Column col, std::unique_ptr<u8[]> &dest);
//---------------------------------------------------------------------------
/// @brief Compresses given column to provided destination using
/// blockwise Frame-Of-Reference encoding and subsequent blockwise bitpacking.
/// @return The size of the compressed data in bytes.
u32 compressFORn(storage::Column col, std::unique_ptr<u8[]> &dest,
                 u32 block_size);
//---------------------------------------------------------------------------
/// @brief Compresses given column to provided destination using
/// blockwise Frame-Of-Reference encoding and subsequent blockwise bitpacking.
/// Compared to the regular FORn, this compression scheme packs data into
/// arbitrary bit widths.
/// @return The size of the compressed data in bytes.
u32 compressAdaptiveFORn(storage::Column col, std::unique_ptr<u8[]> &dest,
                         u32 block_size);
//---------------------------------------------------------------------------
} // namespace compressor
//---------------------------------------------------------------------------
} // namespace compression
