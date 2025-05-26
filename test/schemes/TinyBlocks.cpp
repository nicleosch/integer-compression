#include <gtest/gtest.h>
//---------------------------------------------------------------------------
#include "core/BlockCompressor.hpp"
#include "core/ColumnCompressor.hpp"
#include "storage/Column.hpp"
//---------------------------------------------------------------------------
using namespace compression;
//---------------------------------------------------------------------------
// Verifies that the data remains unchanged after columnar compression and
// decompression for 32 bit integers.
TEST(TinyBlocksTest, ColumnDecompressionInvariant32bit) {
  constexpr uint16_t kBlockSize = 256;

  auto path = "../data/tpch/sf1/partsupp.tbl";
  auto column = storage::Column<INTEGER>::fromFile(path, 0, '|');

  column.padToMultipleOf(kBlockSize);

  ColumnCompressor<INTEGER, kBlockSize> compressor(
      column, CompressionSchemeType::kTinyBlocks);

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
// decompression for 32 bit integers.
TEST(TinyBlocksTest, BlockDecompressionInvariant32bit) {
  constexpr uint16_t kBlockSize = 256;

  auto path = "../data/tpch/sf1/partsupp.tbl";
  auto column = storage::Column<INTEGER>::fromFile(path, 0, '|');

  column.padToMultipleOf(kDefaultDataBlockSize);

  BlockCompressor<INTEGER, kDefaultDataBlockSize, kBlockSize> compressor(
      column, CompressionSchemeType::kTinyBlocks);

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
// Verifies that the data remains unchanged after columnar compression and
// decompression for 64 bit integers.
TEST(TinyBlocksTest, ColumnDecompressionInvariant64bit) {
  constexpr uint16_t kBlockSize = 256;

  auto path = "../data/tpch/sf1/partsupp.tbl";
  auto column = storage::Column<BIGINT>::fromFile(path, 0, '|');

  column.padToMultipleOf(kBlockSize);

  ColumnCompressor<BIGINT, kBlockSize> compressor(
      column, CompressionSchemeType::kTinyBlocks);

  // compress
  std::unique_ptr<compression::u8[]> compression_out;
  compressor.compress(compression_out);

  // decompress
  std::vector<compression::BIGINT> decompression_out;
  compressor.decompress(decompression_out, compression_out.get());

  // verify
  for (size_t i = 0; i < column.size(); ++i) {
    ASSERT_EQ(column.data()[i], decompression_out[i]);
  }
}
//---------------------------------------------------------------------------
// Verifies that the data remains unchanged after block-wise compression and
// decompression for 64 bit integers.
TEST(TinyBlocksTest, BlockDecompressionInvariant64bit) {
  constexpr uint16_t kBlockSize = 256;

  auto path = "../data/tpch/sf1/partsupp.tbl";
  auto column = storage::Column<BIGINT>::fromFile(path, 0, '|');

  column.padToMultipleOf(kDefaultDataBlockSize);

  BlockCompressor<BIGINT, kDefaultDataBlockSize, kBlockSize> compressor(
      column, CompressionSchemeType::kTinyBlocks);

  // compress
  std::unique_ptr<compression::u8[]> compression_out;
  compressor.compress(compression_out);

  // decompress
  std::vector<compression::BIGINT> decompression_out;
  compressor.decompress(decompression_out, compression_out.get());

  // verify
  for (size_t i = 0; i < column.size(); ++i) {
    ASSERT_EQ(column.data()[i], decompression_out[i]);
  }
}
//---------------------------------------------------------------------------
// Verifies that the data remains unchanged after compression using TinyBlocks
// and a second round of compression using LZ4 on the header.
TEST(TinyBlocksTest, Phase2HeaderCompressionInvariant) {
  constexpr uint16_t kBlockSize = 256;

  auto path = "../data/tpch/sf1/partsupp.tbl";
  auto column = storage::Column<INTEGER>::fromFile(path, 0, '|');

  column.padToMultipleOf(kBlockSize);

  Phase2CompressionSettings settings{CompressionSchemeType::kLZ4, true, false};

  ColumnCompressor<INTEGER, kBlockSize> compressor(column, &settings);
  compressor.setScheme(CompressionSchemeType::kTinyBlocks);

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
// Verifies that the data remains unchanged after compression using TinyBlocks
// and a second round of compression using Zstd on the payload.
TEST(TinyBlocksTest, Phase2PayloadCompressionInvariant) {
  constexpr uint16_t kBlockSize = 256;

  auto path = "../data/tpch/sf1/partsupp.tbl";
  auto column = storage::Column<INTEGER>::fromFile(path, 0, '|');

  column.padToMultipleOf(kBlockSize);

  Phase2CompressionSettings settings{CompressionSchemeType::kZstd, false, true};

  ColumnCompressor<INTEGER, kBlockSize> compressor(column, &settings);
  compressor.setScheme(CompressionSchemeType::kTinyBlocks);

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
// Verifies that the data remains unchanged after compression using TinyBlocks
// and a second round of compression using Snappy on the whole compressed data.
TEST(TinyBlocksTest, Phase2CompressionInvariant) {
  constexpr uint16_t kBlockSize = 256;

  auto path = "../data/tpch/sf1/partsupp.tbl";
  auto column = storage::Column<INTEGER>::fromFile(path, 0, '|');

  column.padToMultipleOf(kBlockSize);

  Phase2CompressionSettings settings{CompressionSchemeType::kSnappy, false,
                                     false};

  ColumnCompressor<INTEGER, kBlockSize> compressor(column, &settings);
  compressor.setScheme(CompressionSchemeType::kTinyBlocks);

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
// Verifies that the data remains unchanged after RLE compression and
// decompression for 32 bit integers.
TEST(TinyBlocksTest, RLEDecompressionInvariant32bit) {
  constexpr uint16_t kBlockSize = 256;

  auto path = "../data/tpch/sf1/nation.tbl";
  auto column = storage::Column<INTEGER>::fromFile(path, 0, '|');

  column.padToMultipleOf(kBlockSize);

  ColumnCompressor<INTEGER, kBlockSize> compressor(
      column, CompressionSchemeType::kTinyBlocks);

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
// Verifies that the data remains unchanged after RLE compression and
// decompression for 64 bit integers.
TEST(TinyBlocksTest, RLEDecompressionInvariant64bit) {
  constexpr uint16_t kBlockSize = 256;

  auto path = "../data/tpch/sf1/nation.tbl";
  auto column = storage::Column<BIGINT>::fromFile(path, 0, '|');

  column.padToMultipleOf(kBlockSize);

  ColumnCompressor<BIGINT, kBlockSize> compressor(
      column, CompressionSchemeType::kTinyBlocks);

  // compress
  std::unique_ptr<compression::u8[]> compression_out;
  compressor.compress(compression_out);

  // decompress
  std::vector<compression::BIGINT> decompression_out;
  compressor.decompress(decompression_out, compression_out.get());

  // verify
  for (size_t i = 0; i < column.size(); ++i) {
    ASSERT_EQ(column.data()[i], decompression_out[i]);
  }
}
//---------------------------------------------------------------------------
// Verifies that the data remains unchanged after MonoInc compression and
// decompression for 32 bit integers.
TEST(TinyBlocksTest, MonoIncDecompressionInvariant) {
  constexpr uint16_t kBlockSize = 256;

  auto path = "../data/tpch/sf1/customer.tbl";
  auto column = storage::Column<INTEGER>::fromFile(path, 0, '|');

  column.padToMultipleOf(kBlockSize);

  ColumnCompressor<INTEGER, kBlockSize> compressor(
      column, CompressionSchemeType::kTinyBlocks);

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
// Verifies that the data remains unchanged after PFOR compression and
// decompression for 32 bit integers.
TEST(TinyBlocksTest, PFORDecompressionInvariant) {
  constexpr uint16_t kTinyBlocksSize = 256;
  using TinyBlocks = tinyblocks::TinyBlocks<INTEGER, kTinyBlocksSize>;
  //---------------------------------------------------------------------------
  auto path = "../data/tpch/sf1/partsupp.tbl";
  auto column = storage::Column<INTEGER>::fromFile(path, 0, '|');
  column.padToMultipleOf(kTinyBlocksSize);
  //---------------------------------------------------------------------------
  // compress
  auto compression_out =
      std::make_unique<u8[]>(column.size() * sizeof(INTEGER) * 2);
  //---------------------------------------------------------------------------
  // Calculate statistics on the data used for compression.
  vector<Statistics<INTEGER>> stats;
  auto block_count = column.size() / kTinyBlocksSize;
  for (size_t i = 0; i < block_count; ++i) {
    stats.push_back(Statistics<INTEGER>::generateFrom(
        column.data() + i * kTinyBlocksSize, kTinyBlocksSize));
  }
  //---------------------------------------------------------------------------
  TinyBlocks tb;
  tb.compress(column.data(), column.size(), compression_out.get(), stats.data(),
              tinyblocks::Scheme::PFOR);
  //---------------------------------------------------------------------------
  // decompress
  vector<INTEGER> decompression_out(column.size());
  tb.decompress(decompression_out.data(), column.size(), compression_out.get());
  //---------------------------------------------------------------------------
  // verify
  for (size_t i = 0; i < column.size(); ++i) {
    ASSERT_EQ(column.data()[i], decompression_out[i]);
  }
}
//---------------------------------------------------------------------------
// Verifies that the data remains unchanged after PFOREBP compression and
// decompression for 32 bit integers.
TEST(TinyBlocksTest, PFOREBPDecompressionInvariant) {
  constexpr uint16_t kTinyBlocksSize = 256;
  using TinyBlocks = tinyblocks::TinyBlocks<INTEGER, kTinyBlocksSize>;
  //---------------------------------------------------------------------------
  auto path = "../data/tpch/sf1/partsupp.tbl";
  auto column = storage::Column<INTEGER>::fromFile(path, 0, '|');
  column.padToMultipleOf(kTinyBlocksSize);
  //---------------------------------------------------------------------------
  // compress
  auto compression_out =
      std::make_unique<u8[]>(column.size() * sizeof(INTEGER) * 2);
  //---------------------------------------------------------------------------
  // Calculate statistics on the data used for compression.
  vector<Statistics<INTEGER>> stats;
  auto block_count = column.size() / kTinyBlocksSize;
  for (size_t i = 0; i < block_count; ++i) {
    stats.push_back(Statistics<INTEGER>::generateFrom(
        column.data() + i * kTinyBlocksSize, kTinyBlocksSize));
  }
  //---------------------------------------------------------------------------
  TinyBlocks tb;
  tb.compress(column.data(), column.size(), compression_out.get(), stats.data(),
              tinyblocks::Scheme::PFOREBP);
  //---------------------------------------------------------------------------
  // decompress
  vector<INTEGER> decompression_out(column.size());
  tb.decompress(decompression_out.data(), column.size(), compression_out.get());
  //---------------------------------------------------------------------------
  // verify
  for (size_t i = 0; i < column.size(); ++i) {
    ASSERT_EQ(column.data()[i], decompression_out[i]);
  }
}