#include <gtest/gtest.h>
//---------------------------------------------------------------------------
#include "schemes/Truncation.hpp"
//---------------------------------------------------------------------------
using namespace compression;
//---------------------------------------------------------------------------
// Verifies that the data remains unchanged after columnar compression and
// decompression.
TEST(TruncationTest, ColumnDecompressionInvariant32bit) {
  constexpr uint16_t kBlockSize = 512;
  //---------------------------------------------------------------------------
  /// Data.
  vector<INTEGER> data(kBlockSize << 4);
  for (u32 i = 0; i < data.size(); ++i) {
    data[i] = i % 4;
  }
  //---------------------------------------------------------------------------
  /// Statistics.
  auto stats = Statistics<INTEGER>::generateFrom(data.data(), data.size());
  //---------------------------------------------------------------------------
  auto compression_out = std::make_unique<compression::u8[]>(data.size() * 8);
  Truncation<INTEGER, kBlockSize> trunc;
  trunc.compress(data.data(), data.size(), compression_out.get(), &stats);
  //---------------------------------------------------------------------------
  vector<compression::INTEGER> decompression_out(data.size());
  trunc.decompress(decompression_out.data(), data.size(),
                   compression_out.get());
  //---------------------------------------------------------------------------
  // verify
  for (size_t i = 0; i < data.size(); ++i) {
    ASSERT_EQ(data.data()[i], decompression_out[i]);
  }
}
//---------------------------------------------------------------------------
// Verifies that the data remains unchanged after columnar compression and
// decompression.
TEST(TruncationTest, ColumnDecompressionInvariant64bit) {
  constexpr uint16_t kBlockSize = 512;
  //---------------------------------------------------------------------------
  //---------------------------------------------------------------------------
  /// Data.
  vector<BIGINT> data(kBlockSize << 4);
  for (u32 i = 0; i < data.size(); ++i) {
    data[i] = i % 4;
  }
  //---------------------------------------------------------------------------
  /// Statistics.
  auto stats = Statistics<BIGINT>::generateFrom(data.data(), data.size());
  //---------------------------------------------------------------------------
  auto compression_out = std::make_unique<compression::u8[]>(data.size() * 8);
  Truncation<BIGINT, kBlockSize> trunc;
  trunc.compress(data.data(), data.size(), compression_out.get(), &stats);
  //---------------------------------------------------------------------------
  vector<compression::BIGINT> decompression_out(data.size());
  trunc.decompress(decompression_out.data(), data.size(),
                   compression_out.get());
  //---------------------------------------------------------------------------
  // verify
  for (size_t i = 0; i < data.size(); ++i) {
    ASSERT_EQ(data.data()[i], decompression_out[i]);
  }
}