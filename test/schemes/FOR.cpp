#include <gtest/gtest.h>
//---------------------------------------------------------------------------
#include "core/BlockCompressor.hpp"
#include "core/ColumnCompressor.hpp"
#include "storage/Column.hpp"
//---------------------------------------------------------------------------
using namespace compression;
//---------------------------------------------------------------------------
// Verifies that the data remains unchanged after columnar compression and
// decompression.
TEST(FORTest, ColumnDecompressionInvariant) {
  constexpr uint16_t kBlockSize = 256;

  auto path = "../data/tpch/sf1/partsupp.tbl";
  auto column = storage::Column::fromFile(path, 0, '|');

  ColumnCompressor<kBlockSize> compressor(column, CompressionSchemeType::kFOR);

  // compress
  std::unique_ptr<compression::u8[]> compression_out;
  compressor.compress(compression_out);

  // decompress
  std::vector<compression::INTEGER> decompression_out;
  compressor.decompress(decompression_out, compression_out.get());

  // verify
  for (size_t i = 0; i < column.size(); ++i) {
    ASSERT_EQ(column.data()[i], decompression_out[i]);
  }
}
//---------------------------------------------------------------------------
// Verifies that the data remains unchanged after block-wise compression and
// decompression.
TEST(FORTest, BlockDecompressionInvariant) {
  constexpr uint16_t kBlockSize = 256;

  auto path = "../data/tpch/sf1/partsupp.tbl";
  auto column = storage::Column::fromFile(path, 0, '|');

  column.padToMultipleOf(kDefaultDataBlockSize);

  BlockCompressor<kDefaultDataBlockSize, kBlockSize> compressor(
      column, CompressionSchemeType::kFOR);

  // compress
  std::unique_ptr<compression::u8[]> compression_out;
  compressor.compress(compression_out);

  // decompress
  std::vector<compression::INTEGER> decompression_out;
  compressor.decompress(decompression_out, compression_out.get());

  // verify
  for (size_t i = 0; i < column.size(); ++i) {
    ASSERT_EQ(column.data()[i], decompression_out[i]);
  }
}