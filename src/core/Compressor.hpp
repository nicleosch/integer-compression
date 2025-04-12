#pragma once
//---------------------------------------------------------------------------
#include <memory>
//---------------------------------------------------------------------------
#include "schemes/CompressionScheme.hpp"
#include "storage/Column.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
using storage::Column;
//---------------------------------------------------------------------------
struct CompressionStats {
  double compression_rate;
  u64 uncompressed_size;
  u64 compressed_size;
};
//---------------------------------------------------------------------------
class Compressor {
public:
  //---------------------------------------------------------------------------
  /// Constructor.
  explicit Compressor(const Column &column)
      : column(column), scheme(CompressionSchemeType::kUncompressed) {}
  //---------------------------------------------------------------------------
  /// Constructor.
  Compressor(const Column &column, const CompressionSchemeType scheme)
      : column(column), scheme(scheme) {}
  //---------------------------------------------------------------------------
  /// Compress the column into given output buffer.
  /// @return The size of the compressed data.
  virtual CompressionStats compress(std::unique_ptr<u8[]> &dest) = 0;
  //---------------------------------------------------------------------------
  /// Decompress src data to provided destination.
  virtual void decompress(vector<INTEGER> &dest, u8 *src) = 0;
  //---------------------------------------------------------------------------
  /// Decompress src data.
  /// Note: Data is compressed into a local L1-resident buffer for benchmarking.
  virtual void decompress(u8 *src) = 0;
  //---------------------------------------------------------------------------
  void setScheme(const CompressionSchemeType scheme) { this->scheme = scheme; }

protected:
  /// The column to be compressed.
  Column column;
  /// The scheme used to compress the column.
  CompressionSchemeType scheme;
};
//---------------------------------------------------------------------------
} // namespace compression