#pragma once
//---------------------------------------------------------------------------
#include "common/Types.hpp"
#include "statistics/Statistics.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
enum class CompressionSchemeType {
  kBitPacking = 0,
  kDelta = 1,
  kFOR = 2,
  kFORn = 3,
  kRLE = 4,
  kTinyBlocks = 5,
  kUncompressed = 6,
  kLZ4 = 7,
  kZstd = 8,
  kSnappy = 9,
  kDataBlock = 10,
};
//---------------------------------------------------------------------------
struct CompressionDetails {
  u64 header_size;
  u64 payload_size;
};
//---------------------------------------------------------------------------
/// This class represents an interface for lightweight compression schemes.
template <typename DataType> class CompressionScheme {
public:
  //---------------------------------------------------------------------------
  /// Destructor.
  virtual ~CompressionScheme() = default;
  //---------------------------------------------------------------------------
  /// Compress data from src to dest.
  /// @param src The integers to be compressed.
  /// @param size The amount of integers to be compressed.
  /// @param dest The destination to compress the data to.
  /// @param stats Statistics on the data required to compress it.
  /// @return Some statistics on the compressed data.
  virtual CompressionDetails compress(const DataType *src, const u32 size,
                                      u8 *dest,
                                      const Statistics<DataType> *stats) = 0;
  //---------------------------------------------------------------------------
  /// Decompress data from src to dest.
  /// @param dest The decompressed integers.
  /// @param size The amount of integers to be decompressed.
  /// @param src The compressed data.
  virtual void decompress(DataType *dest, const u32 size, const u8 *src) = 0;
  //---------------------------------------------------------------------------
  /// Whether the scheme compresses partitions the data before compressing it.
  virtual bool isPartitioningScheme() = 0;
};
//---------------------------------------------------------------------------
} // namespace compression
