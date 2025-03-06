#pragma once
//---------------------------------------------------------------------------
#include "common/Units.hpp"
#include "statistics/Statistics.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
enum class CompressionSchemeType : u8 {
  kFOR = 1,
  kFORn = 2,
};
//---------------------------------------------------------------------------
class CompressionScheme {
  public:
    virtual ~CompressionScheme() = default;
    //---------------------------------------------------------------------------
    /// @brief Compresses the given data.
    /// @param src The data to be compressed.
    /// @param dest The compressed data.
    /// @param stats Statistics for each block in the source data.
    /// @param total_size The number of integers over all blocks.
    /// @param block_size The number of integers within a block.
    virtual void compress(
        const INTEGER* src,
        u8* dest,
        const Statistics* stats,
        const u32 total_size,
        const u16 block_size
    ) = 0;
    //---------------------------------------------------------------------------
    /// @brief Decompresses the given data.
    /// @param dest The decompressed data.
    /// @param src The data to be decompressed.
    /// @param total_size The number of integers over all blocks.
    /// @param block_size The number of integers within a block.
    virtual void decompress(
        INTEGER* dest,
        const u8* src,
        const u32 total_size,
        const u16 block_size
    ) = 0;
    //---------------------------------------------------------------------------
    virtual CompressionSchemeType getType() = 0;
};
//---------------------------------------------------------------------------
}  // namespace compression