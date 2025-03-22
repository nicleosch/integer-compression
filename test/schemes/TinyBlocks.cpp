#include <gtest/gtest.h>
//---------------------------------------------------------------------------
#include "compression/Compressor.hpp"
#include "compression/Decompressor.hpp"
#include "statistics/Statistics.hpp"
#include "storage/Column.hpp"
//---------------------------------------------------------------------------
namespace cs = compression::storage;
namespace cc = compression::compressor;
namespace cd = compression::decompressor;
//---------------------------------------------------------------------------
// Verifies that the data remains unchanged after compression and decompression.
TEST(TinyBlocksTest, DecompressionInvariant) {
  auto path = "../data/tpch/partsupp.tbl";
  auto column = cs::Column::fromFile(path, 0, '|');
  constexpr uint16_t block_size = 256;
  column.padToMultipleOf(block_size);

  // compress
  std::unique_ptr<compression::u8[]> compression_out;
  cc::compressTinyBlocks<block_size>(column, compression_out);

  // decompress
  std::vector<compression::INTEGER> decompression_out;
  cd::decompressTinyBlocks<block_size>(decompression_out, column.size(),
                                       compression_out.get());

  // verify
  for (size_t i = 0; i < column.size(); ++i) {
    ASSERT_EQ(column.data()[i], decompression_out[i]);
  }
}