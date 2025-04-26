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
/// @brief Some statistics on the compression.
struct CompressionStats {
  /// The size of the uncompressed data in bytes.
  u64 uncompressed_size;
  /// The size of the compressed data in bytes.
  u64 compressed_size;
  /// The compression ratio in bytes.
  double compression_rate;
  /// Details on the compressed data.
  CompressionDetails details;
};
//---------------------------------------------------------------------------
/// @brief Optional settings for a second (de)compression phase.
struct Phase2CompressionSettings {
  /// The scheme for the second compression phase.
  CompressionSchemeType scheme;
  /// Whether only the header should be compressed again.
  bool header_only;
  /// Whether only the payload should be compressed again.
  bool payload_only;
};
//---------------------------------------------------------------------------
/// @brief The compression interface.
/// @tparam T: The type of integer to be compressed.
template <typename T> class Compressor {
public:
  //---------------------------------------------------------------------------
  /// Constructor.
  explicit Compressor(const Column<T> &column)
      : column(column), scheme(CompressionSchemeType::kUncompressed),
        settings(nullptr) {}
  //---------------------------------------------------------------------------
  /// Constructor.
  Compressor(const Column<T> &column, const CompressionSchemeType scheme)
      : column(column), scheme(scheme), settings(nullptr) {}
  //---------------------------------------------------------------------------
  /// Constructor.
  Compressor(const Column<T> &column, const Phase2CompressionSettings *settings)
      : column(column), scheme(CompressionSchemeType::kUncompressed),
        settings(settings) {}
  //---------------------------------------------------------------------------
  /// Destructor.
  virtual ~Compressor() = default;
  //---------------------------------------------------------------------------
  /// Compress the column into given output buffer.
  /// @return The size of the compressed data.
  virtual CompressionStats compress(std::unique_ptr<u8[]> &dest) = 0;
  //---------------------------------------------------------------------------
  /// Decompress src data to provided destination.
  virtual void decompress(vector<T> &dest, u8 *src) = 0;
  //---------------------------------------------------------------------------
  /// Decompress src data.
  /// Note: Data is compressed into a local L1-resident buffer for benchmarking.
  virtual void decompress(u8 *src) = 0;
  //---------------------------------------------------------------------------
  void setScheme(const CompressionSchemeType scheme) { this->scheme = scheme; }

protected:
  /// The column to be compressed.
  Column<T> column;
  /// The scheme used to compress the column.
  CompressionSchemeType scheme;
  /// Settings for a second compression phase, if there is one.
  const Phase2CompressionSettings *settings;
};
//---------------------------------------------------------------------------
} // namespace compression